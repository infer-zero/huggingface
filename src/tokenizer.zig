const Self = @This();
const Parsed = std.json.Parsed(std.json.Value);

allocator: std.mem.Allocator,
parsed_tokenizer_ptr: *ParsedTokenizer,

parsed_tokenizer: ParsedTokenizer,

pub fn init(allocator: std.mem.Allocator, tokenizer_file: std.fs.File) !Self {
    const json_bytes = try tokenizer_file.readToEndAlloc(allocator, 128 * 1024 * 1024);
    defer allocator.free(json_bytes);

    const parsed: Parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_bytes, .{});
    defer parsed.deinit();

    const data = parsed.value;

    var encoding: ParsedTokenizer.EncodingMap = .empty;
    errdefer {
        var enc_it = encoding.iterator();
        while (enc_it.next()) |entry| allocator.free(entry.key_ptr.*);
        encoding.deinit(allocator);
    }

    var decoding: ParsedTokenizer.DecodingMap = .empty;
    errdefer {
        var dec_it = decoding.iterator();
        while (dec_it.next()) |entry| allocator.free(entry.value_ptr.*);
        decoding.deinit(allocator);
    }

    const model = try getObject(data, "model");

    const unknown_token: ?[]const u8 = blk: {
        if (model.object.get("unk_token")) |unk| {
            if (unk == .string) {
                break :blk try allocator.dupe(u8, unk.string);
            }
        }
        break :blk null;
    };
    errdefer if (unknown_token) |unk| allocator.free(unk);

    // Build merge pair index: "left\x00right" -> merge_idx
    var merge_index: ParsedTokenizer.MergePairIndex = .{};
    errdefer {
        var merge_it = merge_index.iterator();
        while (merge_it.next()) |entry| allocator.free(entry.key_ptr.*);
        merge_index.deinit(allocator);
    }

    var key_buf: std.ArrayListUnmanaged(u8) = .{};
    defer key_buf.deinit(allocator);

    const merges = try getArray(model, "merges");
    for (merges, 0..) |merge, idx| {
        var first: []const u8 = undefined;
        var second: []const u8 = undefined;

        if (merge == .string) {
            const split = std.mem.indexOfScalar(u8, merge.string, ' ') orelse {
                log.err("tokenizer.json: merge entry {d} missing space separator", .{idx});
                return error.InvalidTokenizer;
            };
            first = merge.string[0..split];
            second = merge.string[split + 1 ..];
        } else if (merge == .array) {
            if (merge.array.items.len != 2) {
                log.err("tokenizer.json: merge entry {d} array has {d} items (expected 2)", .{ idx, merge.array.items.len });
                return error.InvalidTokenizer;
            }
            const first_val = merge.array.items[0];
            const second_val = merge.array.items[1];
            if (first_val != .string or second_val != .string) {
                log.err("tokenizer.json: merge entry {d} array items are not strings", .{idx});
                return error.InvalidTokenizer;
            }
            first = first_val.string;
            second = second_val.string;
        } else {
            log.err("tokenizer.json: merge entry {d} has unexpected type", .{idx});
            return error.InvalidTokenizer;
        }

        key_buf.clearRetainingCapacity();
        try key_buf.appendSlice(allocator, first);
        try key_buf.append(allocator, 0);
        try key_buf.appendSlice(allocator, second);
        const key = try allocator.dupe(u8, key_buf.items);
        errdefer allocator.free(key);
        try merge_index.put(allocator, key, idx);
    }

    const vocabulary = try getObject(model, "vocab");

    var vocab_iter = vocabulary.object.iterator();
    while (vocab_iter.next()) |entry| {
        const raw_token = entry.key_ptr.*;
        const token_id: u32 = @intCast(entry.value_ptr.*.integer);

        const decoded_token = try decodeRawToken(allocator, raw_token);
        const subword_enc = try allocator.dupe(u8, raw_token);

        try encoding.put(allocator, subword_enc, token_id);
        try decoding.put(allocator, token_id, decoded_token);
    }

    var special_tokens: ParsedTokenizer.SpecialTokenMap = .empty;
    errdefer {
        var sp_it = special_tokens.iterator();
        while (sp_it.next()) |entry| allocator.free(entry.key_ptr.*);
        special_tokens.deinit(allocator);
    }

    const added_tokens = try getArray(data, "added_tokens");
    for (added_tokens) |token| {
        const token_id: u32 = @intCast(token.object.get("id").?.integer);
        const content = try getString(token, "content");

        if (!encoding.contains(content)) {
            const enc_token = try allocator.dupe(u8, content);
            const dec_token = try allocator.dupe(u8, content);

            try encoding.put(allocator, enc_token, token_id);
            try decoding.put(allocator, token_id, dec_token);
        }

        // Add ALL added_tokens to special_tokens (used to build special_tokens_sorted,
        // which the encoder splits on). The "special" flag in tokenizer.json is a
        // *decode-time* concern (whether to skip when skip_special_tokens=True),
        // NOT an encode-time concern. HF's tokenizer treats every added_token as
        // atomic during encoding regardless. Skipping non-special added tokens here
        // causes Qwen3-style architectures to BPE-split tokens like `<think>`,
        // `<tool_call>`, etc. into sub-tokens, which produces a token sequence the
        // model never saw during training.
        const sp_key = try allocator.dupe(u8, content);
        try special_tokens.put(allocator, sp_key, token_id);
    }

    // Build sorted special tokens list (longest first for greedy matching)
    const special_tokens_sorted = blk: {
        const sp_count = special_tokens.count();
        if (sp_count == 0) break :blk &[_]ParsedTokenizer.SpecialTokenEntry{};

        const sorted = try allocator.alloc(ParsedTokenizer.SpecialTokenEntry, sp_count);
        errdefer allocator.free(sorted);

        var sp_iter = special_tokens.iterator();
        var index: usize = 0;
        while (sp_iter.next()) |entry| : (index += 1) {
            sorted[index] = .{ .text = entry.key_ptr.*, .id = entry.value_ptr.* };
        }

        std.mem.sort(ParsedTokenizer.SpecialTokenEntry, sorted, {}, struct {
            fn lessThan(_: void, lhs: ParsedTokenizer.SpecialTokenEntry, rhs: ParsedTokenizer.SpecialTokenEntry) bool {
                // Sort by length descending (longer tokens first for greedy match)
                return lhs.text.len > rhs.text.len;
            }
        }.lessThan);

        break :blk sorted;
    };
    errdefer if (special_tokens_sorted.len > 0) allocator.free(special_tokens_sorted);

    const normalizer = try readNormalizer(allocator, data);
    errdefer if (normalizer) |norm| freeNormalizer(allocator, norm);

    const post_processor = try readPostProcessor(allocator, data, &encoding);
    errdefer if (post_processor) |proc| freePostProcessor(allocator, proc);

    const use_byte_level = blk: {
        if (data.object.get("decoder")) |decoder| {
            if (decoder == .object) {
                if (decoder.object.get("type")) |decoder_type| {
                    if (decoder_type == .string) {
                        if (std.mem.eql(u8, decoder_type.string, "ByteLevel")) break :blk true;
                        // Check inside Sequence decoders
                        if (std.mem.eql(u8, decoder_type.string, "Sequence")) {
                            if (decoder.object.get("decoders")) |decoders| {
                                if (decoders == .array) {
                                    for (decoders.array.items) |sub_decoder| {
                                        if (sub_decoder == .object) {
                                            if (sub_decoder.object.get("type")) |sub_type| {
                                                if (sub_type == .string and std.mem.eql(u8, sub_type.string, "ByteLevel"))
                                                    break :blk true;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        break :blk false;
    };

    const tok = try allocator.create(ParsedTokenizer);
    tok.* = .{
        .decoding = decoding,
        .encoding = encoding,
        .merge_index = merge_index,
        .unknown_token = unknown_token,
        .normalizer = normalizer,
        .post_processor = post_processor,
        .use_byte_level = use_byte_level,
        .special_tokens = special_tokens,
        .special_tokens_sorted = special_tokens_sorted,
    };

    return Self{
        .allocator = allocator,
        .parsed_tokenizer_ptr = tok,
        .parsed_tokenizer = tok.*,
    };
}

pub fn deinit(self: Self) void {
    // Free encoding keys
    var enc_iter = self.parsed_tokenizer_ptr.encoding.iterator();
    while (enc_iter.next()) |entry| {
        self.allocator.free(entry.key_ptr.*);
    }
    self.parsed_tokenizer_ptr.encoding.deinit(self.allocator);

    // Free decoding values
    var dec_iter = self.parsed_tokenizer_ptr.decoding.iterator();
    while (dec_iter.next()) |entry| {
        self.allocator.free(entry.value_ptr.*);
    }
    self.parsed_tokenizer_ptr.decoding.deinit(self.allocator);

    // Free merge index keys
    var merge_idx_iter = self.parsed_tokenizer_ptr.merge_index.iterator();
    while (merge_idx_iter.next()) |entry| {
        self.allocator.free(entry.key_ptr.*);
    }
    self.parsed_tokenizer_ptr.merge_index.deinit(self.allocator);

    if (self.parsed_tokenizer_ptr.unknown_token) |unk| {
        self.allocator.free(unk);
    }

    if (self.parsed_tokenizer_ptr.normalizer) |normalizer| {
        freeNormalizer(self.allocator, normalizer);
    }

    if (self.parsed_tokenizer_ptr.post_processor) |post_processor| {
        freePostProcessor(self.allocator, post_processor);
    }

    // Free special tokens
    var sp_iter = self.parsed_tokenizer_ptr.special_tokens.iterator();
    while (sp_iter.next()) |entry| {
        self.allocator.free(entry.key_ptr.*);
    }
    self.parsed_tokenizer_ptr.special_tokens.deinit(self.allocator);
    if (self.parsed_tokenizer_ptr.special_tokens_sorted.len > 0) {
        self.allocator.free(self.parsed_tokenizer_ptr.special_tokens_sorted);
    }

    self.allocator.destroy(self.parsed_tokenizer_ptr);
}

fn decodeRawToken(allocator: std.mem.Allocator, raw_token: []const u8) ![]const u8 {
    if (raw_token.len == 6 and
        raw_token[0] == '<' and
        raw_token[1] == '0' and
        raw_token[2] == 'x' and
        raw_token[5] == '>')
    {
        const hex_chars = raw_token[3..5];
        const byte_val = std.fmt.parseInt(u8, hex_chars, 16) catch {
            return try allocator.dupe(u8, raw_token);
        };
        const result = try allocator.alloc(u8, 1);
        result[0] = byte_val;
        return result;
    }
    return allocator.dupe(u8, raw_token);
}

fn readNormalizer(allocator: std.mem.Allocator, data: std.json.Value) !?ParsedTokenizer.Normalizer {
    const v_normalizer = data.object.get("normalizer") orelse return null;
    if (v_normalizer == .null) return null;
    return try readNormalizerValue(allocator, v_normalizer);
}

fn readNormalizerValue(allocator: std.mem.Allocator, v_normalizer: std.json.Value) !?ParsedTokenizer.Normalizer {
    const n_type = try getString(v_normalizer, "type");

    if (std.ascii.eqlIgnoreCase("Sequence", n_type)) {
        var result: std.ArrayListUnmanaged(ParsedTokenizer.Normalizer) = .{};
        defer result.deinit(allocator);

        const normalizers = try getArray(v_normalizer, "normalizers");
        for (normalizers) |v_norm| {
            if (try readNormalizerValue(allocator, v_norm)) |norm| {
                try result.append(allocator, norm);
            }
        }

        return .{ .sequence = try result.toOwnedSlice(allocator) };
    } else if (std.ascii.eqlIgnoreCase("Prepend", n_type)) {
        const value = try getString(v_normalizer, "prepend");
        return .{ .prepend = try allocator.dupe(u8, value) };
    } else if (std.ascii.eqlIgnoreCase("Replace", n_type)) {
        const content = try getString(v_normalizer, "content");
        const pattern = try getObject(v_normalizer, "pattern");
        const string = try getString(pattern, "String");

        return .{
            .replace = .{
                .pattern = try allocator.dupe(u8, string),
                .content = try allocator.dupe(u8, content),
            },
        };
    }
    return null;
}

fn readPostProcessor(
    allocator: std.mem.Allocator,
    data: std.json.Value,
    encoding: *ParsedTokenizer.EncodingMap,
) !?ParsedTokenizer.PostProcessor {
    const v_post = data.object.get("post_processor") orelse return null;
    if (v_post == .null) return null;
    return try readPostProcessorValue(allocator, v_post, encoding);
}

fn readPostProcessorValue(
    allocator: std.mem.Allocator,
    v_post: std.json.Value,
    encoding: *ParsedTokenizer.EncodingMap,
) !?ParsedTokenizer.PostProcessor {
    const post_type = try getString(v_post, "type");

    if (std.ascii.eqlIgnoreCase("Sequence", post_type)) {
        var result: std.ArrayListUnmanaged(ParsedTokenizer.PostProcessor) = .{};
        defer result.deinit(allocator);

        const processors = try getArray(v_post, "processors");
        for (processors) |v_proc| {
            if (try readPostProcessorValue(allocator, v_proc, encoding)) |proc| {
                try result.append(allocator, proc);
            }
        }

        return .{ .sequence = try result.toOwnedSlice(allocator) };
    } else if (std.ascii.eqlIgnoreCase("TemplateProcessing", post_type)) {
        var template: std.ArrayListUnmanaged(ParsedTokenizer.PostProcessor.TemplateProcessing) = .{};
        defer template.deinit(allocator);

        const single = try getArray(v_post, "single");
        for (single) |element| {
            if (element.object.get("SpecialToken")) |special_token| {
                const token = try getString(special_token, "id");
                const token_id = encoding.get(token).?;
                try template.append(allocator, .{ .special_token = token_id });
            } else if (element.object.get("Sequence")) |_| {
                try template.append(allocator, .{ .sequence = {} });
            }
        }

        return .{ .template = try template.toOwnedSlice(allocator) };
    }
    return null;
}

fn getObject(json_val: std.json.Value, key: []const u8) !std.json.Value {
    if (json_val != .object) {
        log.err("tokenizer.json: expected object when reading '{s}'", .{key});
        return error.InvalidJSON;
    }
    return json_val.object.get(key) orelse {
        log.err("tokenizer.json: missing required key '{s}'", .{key});
        return error.InvalidJSON;
    };
}

fn getString(json_val: std.json.Value, key: []const u8) ![]const u8 {
    if (json_val != .object) {
        log.err("tokenizer.json: expected object when reading '{s}'", .{key});
        return error.InvalidJSON;
    }
    const val = json_val.object.get(key) orelse {
        log.err("tokenizer.json: missing required key '{s}'", .{key});
        return error.InvalidJSON;
    };
    if (val != .string) {
        log.err("tokenizer.json: key '{s}' expected string, got different type", .{key});
        return error.InvalidJSON;
    }
    return val.string;
}

fn getArray(json_val: std.json.Value, key: []const u8) ![]const std.json.Value {
    if (json_val != .object) {
        log.err("tokenizer.json: expected object when reading '{s}'", .{key});
        return error.InvalidJSON;
    }
    const val = json_val.object.get(key) orelse {
        log.err("tokenizer.json: missing required key '{s}'", .{key});
        return error.InvalidJSON;
    };
    if (val != .array) {
        log.err("tokenizer.json: key '{s}' expected array, got different type", .{key});
        return error.InvalidJSON;
    }
    return val.array.items;
}

fn freeNormalizer(allocator: std.mem.Allocator, normalizer: ParsedTokenizer.Normalizer) void {
    switch (normalizer) {
        .sequence => |seq| {
            for (seq) |norm| freeNormalizer(allocator, norm);
            allocator.free(seq);
        },
        .replace => |replace| {
            allocator.free(replace.pattern);
            allocator.free(replace.content);
        },
        .prepend => |prepend| allocator.free(prepend),
    }
}

fn freePostProcessor(allocator: std.mem.Allocator, post_processor: ParsedTokenizer.PostProcessor) void {
    switch (post_processor) {
        .sequence => |seq| {
            for (seq) |processor| freePostProcessor(allocator, processor);
            allocator.free(seq);
        },
        .template => |template| allocator.free(template),
    }
}

test "load tokenizer.json and verify vocabulary" {
    const tokenizer_file = try std.fs.cwd().openFile("test_models/TinyStories-656K/tokenizer.json", .{});
    defer tokenizer_file.close();

    const tokenizer = try init(testing.allocator, tokenizer_file);
    defer tokenizer.deinit();

    const tok = tokenizer.parsed_tokenizer;

    try testing.expectEqual(26, tok.encoding.get("A").?);
    try testing.expectEqualStrings("A", tok.decoding.get(26).?);
    try testing.expect(tok.special_tokens.count() > 0);
    try testing.expect(tok.merge_index.count() > 0);
}

/// Parser-native tokenizer data extracted from a HuggingFace `tokenizer.json`.
/// All strings and hashmap keys are owned by the parser's allocator. Phase 6b
/// moved this out of `base.Vocabulary` so the parser no longer depends on
/// base; `harness/src/adapters.zig::vocabularyOwned` builds a fresh
/// `base.Vocabulary` from this struct on demand.
pub const ParsedTokenizer = struct {
    encoding: EncodingMap,
    decoding: DecodingMap,
    merge_index: MergePairIndex,
    special_tokens: SpecialTokenMap = .empty,
    special_tokens_sorted: []const SpecialTokenEntry = &.{},
    unknown_token: ?[]const u8 = null,
    normalizer: ?Normalizer = null,
    post_processor: ?PostProcessor = null,
    use_byte_level: bool = false,

    pub const TokenID = u32;
    pub const EncodingMap = std.StringHashMapUnmanaged(TokenID);
    pub const DecodingMap = std.AutoHashMapUnmanaged(TokenID, []const u8);
    pub const MergePairIndex = std.StringHashMapUnmanaged(usize);
    pub const SpecialTokenMap = std.StringHashMapUnmanaged(TokenID);

    pub const SpecialTokenEntry = struct {
        text: []const u8,
        id: TokenID,
    };

    pub const Normalizer = union(enum) {
        sequence: []const Normalizer,
        prepend: []const u8,
        replace: struct {
            pattern: []const u8,
            content: []const u8,
        },
    };

    pub const PostProcessor = union(enum) {
        sequence: []const PostProcessor,
        template: []const TemplateProcessing,

        pub const TemplateProcessing = union(enum) {
            sequence: void,
            special_token: TokenID,
        };
    };
};

const log = std.log.scoped(.infer);

const std = @import("std");
const testing = std.testing;
