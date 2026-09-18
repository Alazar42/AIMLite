# AIMLite CLI Executable Build Directory

This directory hosts the compiled standalone executable for the **AIMLite Zero-Path CLI** along with automated build scripts.

---

## Artifacts in this Directory

- **`aimlite`**: A standalone, self-contained executable file compiled using Python's native `zipapp` packaging. It bundles the entire AIMLite library and CLI toolchain into a single executable with executable permissions (`+x`) and a `/usr/bin/env python3` interpreter shebang.
- **`build_cli.py`**: Python build script that stages `src/` packages and compiles `aimlite`.
- **`build_cli.sh`**: Shell convenience script that executes `build_cli.py`.

---

## Quickstart: Running the Standalone Executable

You can execute the binary directly from anywhere:

```bash
# View available commands
./build/aimlite --help

# Run system & hardware diagnostics
./build/aimlite doctor

# Initialize a new ML project
./build/aimlite init my_ai

# Serve inference server
./build/aimlite serve --port 8000
```

---

## Rebuilding the Executable

Whenever you make changes to `src/aimlite/` or `src/cli/`, rebuild the standalone executable with:

```bash
python3 build/build_cli.py
```
or
```bash
./build/build_cli.sh
```
