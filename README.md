# huggingface

HuggingFace Hub integration: model downloading, safetensors parsing, and tokenizer loading.

## Modules

- **Safetensors** — Parser for the [safetensors](https://huggingface.co/docs/safetensors/) format. Scans `.safetensors` files in a directory, parses JSON headers, and provides a `MetadataIndex` mapping tensor names to their file location, dtype, offset, and size.
- **HfTokenizer** — BPE tokenizer loader for HuggingFace `tokenizer.json` format.
- **download** — Model downloader for the HuggingFace Hub API. Supports parallel downloads, file filtering, token-based authentication, and caching (skips download if `config.json` exists).

## CLI

Includes `hfget`, a standalone CLI tool for downloading models:

```bash
zig build hfget -- TinyLlama/TinyLlama-1.1B-Chat-v1.0 ./models/tinyllama
```

## Usage

```bash
zig fetch --save git+https://github.com/infer-zero/huggingface
```

Then in your `build.zig`:

```zig
const hf_dep = b.dependency("huggingface", .{ .target = target, .optimize = optimize });
my_mod.addImport("huggingface", hf_dep.module("huggingface"));
```

```zig
const hf = @import("huggingface");

// Load safetensors from a directory
var st = try hf.Safetensors.init(allocator, "/path/to/model/");
defer st.deinit();

// Download a model
try hf.download.download(allocator, "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "./cache", null, null);
```

## Dependencies

- [base](https://github.com/infer-zero/base) — Core inference abstractions

## License

MIT
