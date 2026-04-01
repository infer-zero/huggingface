const std = @import("std");
const huggingface = @import("huggingface");

pub fn main() !void {
    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .{};
    defer std.debug.assert(gpa.deinit() != .leak);
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 3) {
        const usage =
            \\Usage: hfget <repo_id> <target_folder> [file ...]
            \\
            \\Download a HuggingFace model repository to a local folder.
            \\
            \\Arguments:
            \\  repo_id        HuggingFace repo (e.g. Qwen/Qwen3-0.6B)
            \\  target_folder  Local directory to download into
            \\  file ...       Optional file names to download (default: all)
            \\
            \\Environment:
            \\  HF_TOKEN       Bearer token for gated/private models
            \\
        ;
        std.fs.File.stderr().writeAll(usage) catch {};
        std.process.exit(1);
    }

    const filter: ?[]const []const u8 = if (args.len > 3) args[3..] else null;

    const hf_token = std.process.getEnvVarOwned(allocator, "HF_TOKEN") catch |err| switch (err) {
        error.EnvironmentVariableNotFound => null,
        else => return err,
    };
    defer if (hf_token) |t| allocator.free(t);

    try huggingface.download.download(allocator, args[1], args[2], filter, hf_token);
}
