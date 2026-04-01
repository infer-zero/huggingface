const std = @import("std");
const log = std.log.scoped(.infer);

fn isUnsafeFilename(filename: []const u8) bool {
    if (filename.len == 0) return true;
    if (filename[0] == '/') return true;
    var it = std.mem.splitScalar(u8, filename, '/');
    while (it.next()) |component| {
        if (std.mem.eql(u8, component, "..")) return true;
    }
    return false;
}

pub fn download(
    allocator: std.mem.Allocator,
    repo_id: []const u8,
    target_folder: []const u8,
    file_filters: ?[]const []const u8,
    hf_token: ?[]const u8,
) !void {
    // When filters are provided, skip the API call if all filtered files already exist.
    if (file_filters) |filters| {
        var all_exist = true;
        for (filters) |filter| {
            const dest = try std.fs.path.join(allocator, &.{ target_folder, filter });
            defer allocator.free(dest);
            if (std.fs.cwd().statFile(dest)) |_| {
                log.debug("File {s} exists.", .{filter});
            } else |_| {
                all_exist = false;
                break;
            }
        }
        if (all_exist) {
            log.debug("All filtered files already exist, skipping download.", .{});
            return;
        }
    } else {
        // No filters: skip download if target folder already has a config.json (model already cached).
        const marker = try std.fs.path.join(allocator, &.{ target_folder, "config.json" });
        defer allocator.free(marker);
        if (std.fs.cwd().statFile(marker)) |_| {
            log.debug("Model already cached in {s}, skipping download.", .{target_folder});
            return;
        } else |_| {}
    }

    // Fetch repo file list from HF API.
    const all_filenames = try fetchFileList(allocator, repo_id, hf_token);
    defer {
        for (all_filenames) |filename| allocator.free(filename);
        allocator.free(all_filenames);
    }

    // Apply filter if provided.
    const filenames = filenames: {
        var filtered: std.ArrayList([]const u8) = .empty;

        for (all_filenames) |filename| {
            if (isUnsafeFilename(filename)) {
                log.warn("skipping unsafe filename from API: '{s}'", .{filename});
                continue;
            }
            const dest = try std.fs.path.join(allocator, &.{ target_folder, filename });
            defer allocator.free(dest);
            if (std.fs.cwd().statFile(dest)) |_| {
                log.debug("File {s} exists, skipping.", .{filename});
            } else |_| {
                if (file_filters) |filters| {
                    for (filters) |filter| {
                        if (std.ascii.eqlIgnoreCase(filename, filter)) {
                            log.debug("File {s} in filter list, included.", .{filename});
                            try filtered.append(allocator, filename);
                            break;
                        }
                    } else {
                        log.debug("File {s} NOT in filter list, skipped.", .{filename});
                    }
                } else {
                    log.debug("File {s} included.", .{filename});
                    try filtered.append(allocator, filename);
                }
            }
        }

        break :filenames try filtered.toOwnedSlice(allocator);
    };
    defer allocator.free(filenames);

    // Create target directory and all subdirectories before spawning threads.
    try std.fs.cwd().makePath(target_folder);
    for (filenames) |filename| {
        if (std.fs.path.dirnamePosix(filename)) |dir| {
            const sub_dir_path = try std.fs.path.join(allocator, &.{ target_folder, dir });
            defer allocator.free(sub_dir_path);
            try std.fs.cwd().makePath(sub_dir_path);
        }
    }

    // Download files in parallel using a thread pool.
    const results = try allocator.alloc(bool, filenames.len);
    defer allocator.free(results);

    const dest_paths = try allocator.alloc([]const u8, filenames.len);
    defer {
        for (dest_paths[0..filenames.len]) |path| allocator.free(path);
        allocator.free(dest_paths);
    }
    for (filenames, 0..) |filename, i| {
        dest_paths[i] = try std.fs.path.join(allocator, &.{ target_folder, filename });
    }

    var pool: std.Thread.Pool = undefined;
    try pool.init(.{ .allocator = allocator });
    defer pool.deinit();

    var wg: std.Thread.WaitGroup = .{};
    for (filenames, 0..) |filename, i| {
        log.debug("[{d}/{d}] {s} starting...\n", .{ i + 1, filenames.len, filename });
        pool.spawnWg(
            &wg,
            downloadWorker,
            .{
                allocator, repo_id, filename, dest_paths[i], hf_token, &results[i],
            },
        );
    }
    pool.waitAndWork(&wg);

    for (filenames, 0..) |filename, i| {
        if (results[i]) {
            log.debug("{s}: done.", .{filename});
        } else {
            log.err("{s}: failed.", .{filename});
            return error.DownloadFailed;
        }
    }

    log.debug("Done. Files saved to {s}\n", .{target_folder});
}

fn fetchFileList(allocator: std.mem.Allocator, repo_id: []const u8, hf_token: ?[]const u8) ![][]const u8 {
    var client: std.http.Client = .{ .allocator = allocator };
    defer client.deinit();

    const url = try std.fmt.allocPrint(allocator, "https://huggingface.co/api/models/{s}", .{repo_id});
    defer allocator.free(url);

    const uri = try std.Uri.parse(url);

    const auth_value = if (hf_token) |token|
        try std.fmt.allocPrint(allocator, "Bearer {s}", .{token})
    else
        null;
    defer if (auth_value) |value| allocator.free(value);

    const auth_headers: []const std.http.Header = if (auth_value) |value|
        &.{.{ .name = "authorization", .value = value }}
    else
        &.{};

    var req = try client.request(.GET, uri, .{
        .extra_headers = auth_headers,
        .headers = .{ .accept_encoding = .{ .override = "identity" } },
    });
    defer req.deinit();
    try req.sendBodiless();

    var redirect_buf: [8192]u8 = undefined;
    var response = try req.receiveHead(&redirect_buf);

    if (response.head.status != .ok) {
        log.err("hfget: API request for '{s}' failed with status {s}", .{ repo_id, @tagName(response.head.status) });
        return error.HttpRequestFailed;
    }

    var transfer_buf: [8192]u8 = undefined;
    const reader = response.reader(&transfer_buf);
    const body = try reader.allocRemaining(allocator, .unlimited);
    defer allocator.free(body);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{});
    defer parsed.deinit();

    const siblings = parsed.value.object.get("siblings") orelse {
        log.err("hfget: API response for '{s}' missing 'siblings' field", .{repo_id});
        return error.InvalidApiResponse;
    };

    var filenames: std.ArrayList([]const u8) = .empty;
    defer filenames.deinit(allocator);

    for (siblings.array.items) |sibling| {
        const rfilename = sibling.object.get("rfilename") orelse continue;
        try filenames.append(allocator, try allocator.dupe(u8, rfilename.string));
    }

    return try filenames.toOwnedSlice(allocator);
}

fn downloadWorker(
    allocator: std.mem.Allocator,
    repo_id: []const u8,
    filename: []const u8,
    dest_path: []const u8,
    hf_token: ?[]const u8,
    result: *bool,
) void {
    downloadFile(allocator, repo_id, filename, dest_path, hf_token) catch {
        result.* = false;
        return;
    };
    result.* = true;
}

fn downloadFile(
    allocator: std.mem.Allocator,
    repo_id: []const u8,
    filename: []const u8,
    dest_path: []const u8,
    hf_token: ?[]const u8,
) !void {
    var client: std.http.Client = .{ .allocator = allocator };
    defer client.deinit();

    const url = try std.fmt.allocPrint(allocator, "https://huggingface.co/{s}/resolve/main/{s}", .{ repo_id, filename });
    defer allocator.free(url);

    const uri = try std.Uri.parse(url);

    const auth_value = if (hf_token) |token|
        try std.fmt.allocPrint(allocator, "Bearer {s}", .{token})
    else
        null;
    defer if (auth_value) |value| allocator.free(value);

    const auth_headers: []const std.http.Header = if (auth_value) |value|
        &.{.{ .name = "authorization", .value = value }}
    else
        &.{};

    var req = try client.request(.GET, uri, .{
        .redirect_behavior = @enumFromInt(3),
        .privileged_headers = auth_headers,
        .headers = .{ .accept_encoding = .{ .override = "identity" } },
    });
    defer req.deinit();
    try req.sendBodiless();

    var redirect_buf: [8192]u8 = undefined;
    var response = try req.receiveHead(&redirect_buf);

    if (response.head.status != .ok) {
        log.err("hfget: download failed for '{s}': status {s}", .{ filename, @tagName(response.head.status) });
        return error.HttpRequestFailed;
    }

    var file = try std.fs.cwd().createFile(dest_path, .{});
    defer file.close();

    var transfer_buf: [8192]u8 = undefined;
    const reader = response.reader(&transfer_buf);

    var write_buf: [8192]u8 = undefined;
    var file_writer = file.writerStreaming(&write_buf);

    _ = reader.streamRemaining(&file_writer.interface) catch |err| switch (err) {
        error.ReadFailed => {
            log.err("hfget: HTTP read failed for '{s}'", .{filename});
            return error.HttpReadFailed;
        },
        error.WriteFailed => {
            log.err("hfget: file write failed for '{s}'", .{dest_path});
            return error.FileWriteFailed;
        },
    };
    try file_writer.interface.flush();
}
