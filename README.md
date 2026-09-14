# ModelKit

A modular Python framework and toolkit with dedicated CLI capabilities and extensible modules.

---

## Authors & Collaborators

- **Mickyas Tesfaye** ([alazartesfaye42@gmail.com](mailto:alazartesfaye42@gmail.com))
- **Beamlak Tadesse** ([atocodes@gmail.com](mailto:atocodes@gmail.com))

---

## Quick Start & Setup

This project uses [uv](https://docs.astral.sh/uv/), a fast Python package and project manager.

### 1. Prerequisites
Ensure you have `uv` installed. If you do not have it installed yet:

```bash
# On Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using pip
pip install uv
```

Requires Python **>= 3.14**.

### 2. Clone and Setup Environment

```bash
# Clone the repository
git clone <repo-url>
cd ModelKit

# Create virtual environment and install all packages in editable mode
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

---

## Common Commands

All tasks can be run directly using `uv run`:

| Command | Description |
| :--- | :--- |
| `uv run modelkit` | Run the main ModelKit CLI entrypoint |
| `uv run cli` | Alias command for the CLI |
| `uv run test` | Run the test suite (`test/main.py`) |
| `uv sync` | Sync and update the virtual environment |
| `uv add <package>` | Add a new project dependency |
| `uv add --dev <package>` | Add a development dependency |

---

## Project Structure

```text
ModelKit/
├── src/
│   ├── cli/             # CLI application and script entry points
│   │   ├── __init__.py
│   │   ├── main.py      # Main CLI logic (`uv run modelkit` / `uv run cli`)
│   │   └── test.py      # Test runner hook (`uv run test`)
│   └── modelkit/        # Core library modules
│       ├── __init__.py  # Package exports and public API
│       └── headers.py   # (Upcoming/active modules)
├── test/                # Test suite directory
│   └── main.py          # Primary test runner script
├── pyproject.toml       # Project metadata, build configuration & scripts
└── README.md            # Developer guide & documentation
```

---

## Developer Collaboration Guide: What & How to Work

### 1. Adding Core Library Modules (`src/modelkit/`)
- Any new utility or feature module should be created inside `src/modelkit/` (e.g. `src/modelkit/headers.py`, `src/modelkit/database.py`).
- Export public classes or functions in `src/modelkit/__init__.py` so consumers can import them easily:
  ```python
  from modelkit import Headers
  ```
- Keep modules modular and decoupled.

### 2. Developing CLI Commands (`src/cli/`)
- All command-line tools live under `src/cli/`.
- The CLI entrypoint is `src/cli/main.py:main()`.
- You can add subcommands, argument parsing, or interactive commands here.
- Test your changes directly with:
  ```bash
  uv run modelkit
  # or
  uv run cli
  ```

### 3. Writing and Running Tests (`test/`)
- The test suite is located in the `test/` directory.
- `test/main.py` is the execution target when running:
  ```bash
  uv run test
  ```
- When adding new features in `src/modelkit/` or `src/cli/`, always add corresponding test checks into `test/main.py` (or new test files in `test/`).
- Make sure `uv run test` exits with code `0` before committing your code.

### 4. Git Collaboration Workflow
1. **Pull latest changes** before starting work:
   ```bash
   git pull origin master
   ```
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Develop & verify locally**:
   ```bash
   uv run test
   ```
4. **Commit & push**:
   ```bash
   git add .
   git commit -m "feat: description of changes"
   git push origin feature/your-feature-name
   ```
