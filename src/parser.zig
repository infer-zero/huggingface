//! A parsed HuggingFace model directory. `init` opens the directory and
//! loads its three artifacts — `config.json`, `tokenizer.json`, and every
//! `*.safetensors` file's metadata — into `config` (a `std.json.ObjectMap`
//! view of the parsed JSON), `tokenizer` (an `HfTokenizer`), and
//! `safetensors` (a `Safetensors` index).
//!
//! `config.json` is intentionally surfaced as a raw `ObjectMap` rather
//! than a typed struct: the shape is architecture-specific (llama, qwen3,
//! granite-hybrid, etc. all pull different fields) so each model variant
//! builds its own typed `Config.fromJson(parser.config)`. Use the helpers
//! in `json_config.zig` (`getUint`, `getFloat`, `getBool`) to extract
//! fields.
//!
//! Ownership: `Parser` owns everything it loaded — the directory handle,
//! the JSON arena, the tokenizer maps, and the safetensors metadata —
//! and tears them all down in `deinit`. `safetensors` keeps a reference
//! to the directory for later tensor reads, so `deinit` closes the
//! directory last.

const Self = @This();

io: std.Io,
allocator: std.mem.Allocator,
dir: std.Io.Dir,
parsed_config: Parsed,
config: std.json.ObjectMap,
tokenizer: HfTokenizer,
safetensors: Safetensors,

const Parsed = std.json.Parsed(std.json.Value);

const max_config_bytes = 10 * 1024 * 1024;
const max_tokenizer_bytes = 128 * 1024 * 1024;

pub fn init(io: std.Io, allocator: std.mem.Allocator, path: []const u8) !Self {
    var dir = try std.Io.Dir.cwd().openDir(io, path, .{ .iterate = true });
    errdefer dir.close(io);

    const json_bytes = try dir.readFileAlloc(io, "config.json", allocator, .limited(max_config_bytes));
    defer allocator.free(json_bytes);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_bytes, .{});
    errdefer parsed.deinit();
    if (parsed.value != .object) return error.InvalidConfig;

    const tokenizer_bytes = try dir.readFileAlloc(io, "tokenizer.json", allocator, .limited(max_tokenizer_bytes));
    defer allocator.free(tokenizer_bytes);

    var tokenizer = try HfTokenizer.init(allocator, tokenizer_bytes);
    errdefer tokenizer.deinit();

    var safetensors = try Safetensors.init(io, allocator, dir);
    errdefer safetensors.deinit();

    return .{
        .io = io,
        .allocator = allocator,
        .dir = dir,
        .parsed_config = parsed,
        .config = parsed.value.object,
        .tokenizer = tokenizer,
        .safetensors = safetensors,
    };
}

pub fn deinit(self: *Self) void {
    self.safetensors.deinit();
    self.tokenizer.deinit();
    self.parsed_config.deinit();
    self.dir.close(self.io);
}

test "Parser loads config, tokenizer, safetensors from test model" {
    const io = testing.io;
    var parser = try init(io, testing.allocator, "test_models/TinyStories-656K");
    defer parser.deinit();

    try testing.expectEqual(@as(usize, 2), json_config.getUint(parser.config, "num_hidden_layers").?);
    try testing.expectEqual(@as(usize, 128), json_config.getUint(parser.config, "hidden_size").?);
    try testing.expectEqual(@as(usize, 2048), json_config.getUint(parser.config, "vocab_size").?);

    try testing.expect(parser.tokenizer.encoding.count() > 0);

    try testing.expect(parser.safetensors.metadata.count() > 0);
}

test "Parser errors on missing config.json" {
    const io = testing.io;
    const result = init(io, testing.allocator, "test_models");
    try testing.expectError(error.FileNotFound, result);
}

const std = @import("std");
const testing = std.testing;

const HfTokenizer = @import("tokenizer.zig");
const Safetensors = @import("safetensors.zig");
const json_config = @import("json_config.zig");
