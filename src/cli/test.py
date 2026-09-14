import sys
import subprocess
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    test_main = project_root / "test" / "main.py"
    args = sys.argv[1:]

    if "--pytest" in args:
        args.remove("--pytest")
        cmd = [sys.executable, "-m", "pytest", "test", *args]
        result = subprocess.run(cmd)
        sys.exit(result.returncode)

    if test_main.exists():
        cmd = [sys.executable, str(test_main), *args]
        result = subprocess.run(cmd)
        sys.exit(result.returncode)
    else:
        cmd = [sys.executable, "-m", "pytest", "test", *args]
        result = subprocess.run(cmd)
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
