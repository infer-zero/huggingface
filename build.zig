const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const mod = b.addModule("huggingface", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Tests
    {
        const tests = b.addTest(.{
            .root_module = mod,
        });

        const run_tests = b.addRunArtifact(tests);
        run_tests.setCwd(b.path("."));
        const test_step = b.step("test", "Run huggingface tests");
        test_step.dependOn(&run_tests.step);
    }

}
