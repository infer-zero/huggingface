allocator: std.mem.Allocator,
metadata: MetadataIndex,
model_dir: std.fs.Dir,

pub const MetadataIndex = std.StringArrayHashMapUnmanaged(Metadata);

pub fn init(allocator: std.mem.Allocator, model_dir: std.fs.Dir) !@This() {
    var self: @This() = .{
        .allocator = allocator,
        .metadata = .empty,
        .model_dir = model_dir,
    };
    errdefer self.deinit();

    // iterate over all .safetensors files
    var dir_iter = model_dir.iterate();
    while (try dir_iter.next()) |file_entry| {
        const ext = std.fs.path.extension(file_entry.name);
        if (std.mem.eql(u8, ".safetensors", ext)) {
            const file = try model_dir.openFile(file_entry.name, .{});
            defer {
                file.seekTo(0) catch unreachable;
                file.close();
            }

            var buffer: [1024]u8 = undefined;
            var file_reader = file.reader(&buffer);
            const reader = &file_reader.interface;

            const metadata = try readMetadata(allocator, file_entry.name, reader);
            defer allocator.free(metadata);
            for (metadata) |item| {
                try self.metadata.put(allocator, try allocator.dupe(u8, item.name), item);
            }
        }
    }

    return self;
}

pub fn deinit(self: *@This()) void {
    var metadata_iter = self.metadata.iterator();
    while (metadata_iter.next()) |entry| {
        self.allocator.free(entry.key_ptr.*);
        entry.value_ptr.*.deinit(self.allocator);
    }
    self.metadata.deinit(self.allocator);
}

const Metadata = struct {
    file: []const u8,
    name: []const u8,
    dtype: []const u8,
    offset: usize,
    len: usize,

    pub fn deinit(self: @This(), allocator: std.mem.Allocator) void {
        allocator.free(self.file);
        allocator.free(self.name);
        allocator.free(self.dtype);
    }
};

fn readMetadata(
    allocator: std.mem.Allocator,
    file: []const u8,
    reader: *std.Io.Reader,
) ![]const Metadata {
    var weights_metadata: std.ArrayList(Metadata) = .empty;
    defer weights_metadata.deinit(allocator);
    errdefer for (weights_metadata.items) |item| item.deinit(allocator);

    const json_size = try reader.takeInt(u64, .little);
    if (json_size > 100 * 1024 * 1024) {
        log.err("safetensors: json header size {d} exceeds 100MB sanity limit", .{json_size});
        return error.IOError;
    }

    const json_bytes = try reader.readAlloc(allocator, json_size);
    defer allocator.free(json_bytes);

    const metadata = try std.json.parseFromSlice(std.json.Value, allocator, json_bytes, .{});
    defer metadata.deinit();

    var metadata_iter = metadata.value.object.iterator();
    while (metadata_iter.next()) |entry| {
        const name = entry.key_ptr.*;
        const info = entry.value_ptr.*;

        // Skip __metadata__ key
        if (std.mem.eql(u8, name, "__metadata__")) continue;

        // Only process tensor entries (those with data_offsets)
        if (info != .object) continue;
        const offsets_val = info.object.get("data_offsets") orelse continue;
        if (offsets_val != .array) continue;

        const dtype = info.object.get("dtype").?.string;
        const offsets = offsets_val.array.items;
        if (offsets.len < 2) continue;
        if (offsets[0].integer < 0 or offsets[1].integer < 0) return error.IOError;
        const offset: usize = @intCast(offsets[0].integer);
        const end: usize = @intCast(offsets[1].integer);
        if (end < offset) return error.IOError;
        const len = end - offset;

        const file_dup = try allocator.dupe(u8, file);
        errdefer allocator.free(file_dup);
        const name_dup = try allocator.dupe(u8, name);
        errdefer allocator.free(name_dup);
        const dtype_dup = try allocator.dupe(u8, dtype);
        errdefer allocator.free(dtype_dup);

        try weights_metadata.append(allocator, .{
            .file = file_dup,
            .name = name_dup,
            .dtype = dtype_dup,
            .offset = std.math.add(usize, std.math.add(usize, offset, 8) catch return error.IOError, @intCast(json_size)) catch return error.IOError,
            .len = len,
        });
    }

    return try weights_metadata.toOwnedSlice(allocator);
}

/// Parser-native data-type enum. Mirrors `runtime.Tensor.DataType` numerically
/// so the harness adapter's `rawToTensor` conversion is a single enum cast.
pub const DataType = enum(u8) {
    BF16 = 0,
    FP32 = 1,
    FP16 = 2,
    Q8_0 = 3,
    Q4_0 = 4,
    Q6_K = 5,
    Q4_1 = 6,
    Q5_0 = 7,
    Q4_K = 8,
    Q5_K = 9,
    _,

    pub fn fromString(dtype_str: []const u8) @This() {
        return std.meta.stringToEnum(@This(), dtype_str) orelse .BF16;
    }
};

/// Minimal parser-native tensor view: just the raw bytes plus the dtype.
/// No conversion methods, no dependency on runtime.
pub const RawTensor = struct {
    data_type: DataType,
    data: []const u8,
};

/// Raw-bytes counterpart to `getTensor`. Returns the tensor's on-disk bytes
/// plus its native `DataType` — no dependency on runtime. Caller owns the
/// returned buffer (free with `allocator.free(raw.data)`).
pub fn getTensorRaw(self: *@This(), name: []const u8) !?RawTensor {
    const meta = self.metadata.get(name) orelse return null;

    const file = self.model_dir.openFile(meta.file, .{}) catch {
        log.err("safetensors: failed to open '{s}' for tensor '{s}'", .{ meta.file, name });
        return error.IOError;
    };
    defer file.close();

    var buffer: [4 * 1024]u8 = undefined;
    var file_reader = file.reader(&buffer);
    const reader = &file_reader.interface;

    _ = reader.discard(.limited(meta.offset)) catch {
        log.err("safetensors: failed to seek to tensor '{s}' in '{s}'", .{ name, meta.file });
        return error.IOError;
    };

    const raw_data = reader.readAlloc(self.allocator, meta.len) catch {
        log.err("safetensors: failed to read tensor '{s}' ({d} bytes) from '{s}'", .{ name, meta.len, meta.file });
        return error.IOError;
    };

    return .{
        .data_type = DataType.fromString(meta.dtype),
        .data = raw_data,
    };
}

test "init loads safetensors metadata" {
    const path = "test_models/TinyStories-656K";
    var dir = try std.fs.cwd().openDir(path, .{ .iterate = true });
    defer dir.close();

    var sf = try init(testing.allocator, dir);
    defer sf.deinit();

    try testing.expect(sf.metadata.count() > 0);
}

test "getTensorRaw loads embeddings" {
    const path = "test_models/TinyStories-656K";
    var dir = try std.fs.cwd().openDir(path, .{ .iterate = true });
    defer dir.close();

    var sf = try init(testing.allocator, dir);
    defer sf.deinit();

    const embeddings = try sf.getTensorRaw("model.embed_tokens.weight");
    try testing.expect(embeddings != null);
    const raw = embeddings.?;
    defer testing.allocator.free(raw.data);

    try testing.expectEqual(DataType.BF16, raw.data_type);
    try testing.expect(raw.data.len > 0);
    try testing.expectEqual(@as(usize, 0), raw.data.len % 2);
}

test "getTensorRaw returns null for missing tensor" {
    const path = "test_models/TinyStories-656K";
    var dir = try std.fs.cwd().openDir(path, .{ .iterate = true });
    defer dir.close();

    var sf = try init(testing.allocator, dir);
    defer sf.deinit();

    const missing = try sf.getTensorRaw("nonexistent.weight");
    try testing.expect(missing == null);
}

const log = std.log.scoped(.infer);

const std = @import("std");
const testing = std.testing;
