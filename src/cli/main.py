"""AIMLite CLI Entrypoint: Zero-Path AI & ML Execution Engine."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from cli.commands import (
    run_data_validate,
    run_doctor,
    run_evaluate,
    run_init,
    run_install,
    run_serve,
    run_train,
)
from cli.ui import C, vite_header

VERSION = "0.1.2"


def print_custom_help() -> None:
    """Renders a Vite-style minimalist CLI help dashboard."""
    print(vite_header())
    print(f"  {C.BOLD}Usage:{C.RESET}")
    print(f"    $ {C.CYAN}aimlite{C.RESET} <command> [options]\n")

    print(f"  {C.BOLD}Commands:{C.RESET}")
    print(f"    {C.GREEN}init{C.RESET} [name]                 {C.DIM}Scaffold a new project layout & conventions{C.RESET}")
    print(f"    {C.GREEN}install{C.RESET} <pkgs...>           {C.DIM}Install dependencies into project .venv{C.RESET}")
    print(f"    {C.GREEN}data validate{C.RESET}             {C.DIM}Validate dataset schema & partition readiness{C.RESET}")
    print(f"    {C.GREEN}train{C.RESET} [class]               {C.DIM}Execute zero-path model training for specified class{C.RESET}")
    print(f"    {C.GREEN}evaluate{C.RESET} [class]            {C.DIM}Assess model performance against held-out splits{C.RESET}")
    print(f"    {C.GREEN}serve{C.RESET} [class] [options]     {C.DIM}Launch inference server & custom frontend{C.RESET}")
    print(f"    {C.GREEN}doctor{C.RESET}                    {C.DIM}Inspect runtime, accelerators & directory permissions{C.RESET}\n")

    print(f"  {C.BOLD}Options:{C.RESET}")
    print(f"    {C.DIM}-v, --version{C.RESET}               {C.DIM}Display version number{C.RESET}")
    print(f"    {C.DIM}-h, --help{C.RESET}                  {C.DIM}Display this message{C.RESET}\n")


def build_parser() -> argparse.ArgumentParser:
    """Constructs the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="aimlite",
        description="AIMLite CLI — The Django for AI & Machine Learning (Zero-Path Execution)",
        add_help=False,
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {VERSION}",
    )
    parser.add_argument(
        "-h",
        "--help",
        action="store_true",
        help="Show this help message and exit.",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # init
    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("name", nargs="?", default=None)
    init_parser.add_argument("--no-venv", action="store_true", help="Skip creating .venv and installing aimlite")
    init_parser.add_argument(
        "--type",
        "--template",
        dest="template_type",
        choices=["rag", "fine-tuning", "adapter", "scratch", "default"],
        default=None,
        help="System paradigm template: rag, fine-tuning, or scratch",
    )
    init_parser.add_argument(
        "--chat-provider",
        dest="chat_provider",
        choices=["openai", "gemini", "anthropic", "ollama", "local", "mock"],
        default=None,
        help="Chat provider for RAG LLM synthesis",
    )
    init_parser.add_argument(
        "--vector-db",
        dest="vector_db",
        choices=["postgres", "memory"],
        default=None,
        help="Vector database backend (postgres or memory)",
    )
    init_parser.add_argument(
        "--embedding",
        dest="embedding_engine",
        choices=["sentence-transformers", "api", "tfidf"],
        default=None,
        help="Embedding engine for vector generation",
    )
    init_parser.add_argument("-y", "--yes", "--non-interactive", dest="non_interactive", action="store_true", help="Non-interactive mode")

    # install
    install_parser = subparsers.add_parser("install")
    install_parser.add_argument("packages", nargs="*", default=[])
    install_parser.add_argument("--upgrade", action="store_true")

    # data
    data_parser = subparsers.add_parser("data")
    data_subparsers = data_parser.add_subparsers(dest="data_command")
    data_validate_parser = data_subparsers.add_parser("validate")
    data_validate_parser.add_argument("target", nargs="?", default=None, help="Optional Dataset class name or file to validate")

    # train
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("target", nargs="?", default=None, help="Model or Dataset class name to train (e.g. UserModel)")
    train_parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=None,
        help="Resume training from latest past checkpoint, or specify an explicit checkpoint path",
    )
    train_parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Custom destination directory to place checkpoints (defaults to models/)",
    )

    # evaluate
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("target", nargs="?", default=None, help="Model class name to evaluate (e.g. UserModel)")
    evaluate_parser.add_argument(
        "--checkpoint",
        default=None,
        help="Specific past checkpoint path to evaluate (defaults to discovering latest checkpoint)",
    )

    # serve
    serve_parser = subparsers.add_parser("serve")
    serve_parser.add_argument("target", nargs="?", default=None, help="Model class name to serve (e.g. UserModel)")
    serve_parser.add_argument(
        "--checkpoint",
        default=None,
        help="Specific past checkpoint path to load and serve (defaults to discovering latest checkpoint)",
    )
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    serve_parser.add_argument("--frontend", default=None, help="Custom frontend build directory (e.g. dist/, frontend/)")

    # doctor
    subparsers.add_parser("doctor")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI execution router."""
    raw_args = argv if argv is not None else sys.argv[1:]

    # If no arguments provided or help requested
    if not raw_args or "-h" in raw_args or "--help" in raw_args:
        if len(raw_args) > 1 and raw_args[0] not in ("-h", "--help"):
            parser = build_parser()
            parser.parse_args(raw_args)
            return 0

        print_custom_help()
        return 0

    parser = build_parser()
    args = parser.parse_args(raw_args)

    if not args.command:
        print_custom_help()
        return 0

    if args.command == "init":
        interactive_flag = False if getattr(args, "non_interactive", False) else (True if args.name is None else None)
        return run_init(
            project_name=args.name,
            create_venv=not getattr(args, "no_venv", False),
            template_type=getattr(args, "template_type", None),
            chat_provider=getattr(args, "chat_provider", None),
            vector_db=getattr(args, "vector_db", None),
            embedding_engine=getattr(args, "embedding_engine", None),
            interactive=interactive_flag,
        )

    if args.command == "install":
        return run_install(packages=args.packages, upgrade=args.upgrade)

    if args.command == "data":
        target = getattr(args, "target", None)
        return run_data_validate(target=target)

    if args.command == "train":
        return run_train(
            target=args.target,
            resume=getattr(args, "resume", None),
            checkpoint_dir=getattr(args, "checkpoint_dir", None),
        )

    if args.command == "evaluate":
        return run_evaluate(
            target=args.target,
            checkpoint=getattr(args, "checkpoint", None),
        )

    if args.command == "serve":
        return run_serve(
            target=args.target,
            port=args.port,
            host=args.host,
            frontend=args.frontend,
            checkpoint=getattr(args, "checkpoint", None),
        )

    if args.command == "doctor":
        return run_doctor()

    print_custom_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
