const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const base_dep = b.dependency("infer_base", .{ .target = target, .optimize = optimize });
    const base_mod = base_dep.module("infer_base");

    const mod = b.addModule("huggingface", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    mod.addImport("base", base_mod);

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

    // hfget executable
    {
        const exe = b.addExecutable(.{
            .name = "hfget",
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/bin/hfget.zig"),
                .target = target,
                .optimize = optimize,
            }),
        });
        exe.root_module.addImport("huggingface", mod);
        b.installArtifact(exe);

        const run_cmd = b.addRunArtifact(exe);
        if (b.args) |args| run_cmd.addArgs(args);
        const run_step = b.step("hfget", "Run hfget");
        run_step.dependOn(&run_cmd.step);
    }
}
