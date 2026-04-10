/// Shared helpers for extracting typed values from a `std.json.ObjectMap`.
/// Used by BF16 (HuggingFace Safetensors) model variants to parse `config.json`.

pub fn getUint(obj: std.json.ObjectMap, key: []const u8) ?usize {
    const val = obj.get(key) orelse return null;
    return switch (val) {
        .integer => |v| @intCast(v),
        else => null,
    };
}

pub fn getFloat(obj: std.json.ObjectMap, key: []const u8) ?f32 {
    const val = obj.get(key) orelse return null;
    return switch (val) {
        .float => |v| @floatCast(v),
        .integer => |v| @floatFromInt(v),
        else => null,
    };
}

const std = @import("std");
