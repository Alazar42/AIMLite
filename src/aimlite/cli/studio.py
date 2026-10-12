"""Locally hosted AIMLite Studio: Web-based interactive project creation & architecture dashboard."""

from __future__ import annotations

import json
import mimetypes
import os
import socket
import sys
import threading
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Any, Dict, List, Optional

from aimlite.cli.commands.init import run_init
from aimlite.cli.ui import C, arrow, check, vite_header


class ThreadingStudioServer(ThreadingMixIn, HTTPServer):
    """Multi-threaded HTTP Server for AIMLite Studio."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address, RequestHandlerClass, auto_shutdown: bool = True):
        super().__init__(server_address, RequestHandlerClass)
        self.auto_shutdown = auto_shutdown


def _find_available_port(start_port: int = 8000, max_attempts: int = 100) -> int:
    """Finds the first available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return start_port


def _resolve_static_dist_dir() -> Optional[Path]:
    """Finds the built static assets directory for AIMLite Studio if packaged."""
    # Check packaged templates/studio_dist if present
    pkg_dist = Path(__file__).resolve().parent.parent / "templates" / "studio_dist"
    if pkg_dist.is_dir() and (pkg_dist / "index.html").is_file():
        return pkg_dist

    return None


def create_studio_handler(dist_dir: Optional[Path] = None):
    """Creates a configured RequestHandler for AIMLite Studio."""

    class StudioHTTPHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            code = args[1] if len(args) > 1 else "200"
            code_color = (
                C.GREEN
                if str(code).startswith("2")
                else (C.RED if str(code).startswith("4") or str(code).startswith("5") else C.YELLOW)
            )
            method = args[0].split()[0] if len(args) > 0 and " " in str(args[0]) else "REQ"
            url = args[0].split()[1] if len(args) > 0 and len(str(args[0]).split()) > 1 else ""
            print(f"  {C.DIM}[{self.log_date_time_string()}]{C.RESET} {C.CYAN}{method:<4}{C.RESET} {code_color}{code}{C.RESET} {url}")

        def _send_cors_headers(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self._send_cors_headers()
            self.end_headers()

        def _send_json(self, status_code: int, data_obj: Any) -> None:
            data = json.dumps(data_obj, indent=2).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(data)

        def _serve_file(self, file_path: Path) -> bool:
            if file_path.is_file():
                mime_type, _ = mimetypes.guess_type(str(file_path))
                mime_type = mime_type or "application/octet-stream"
                try:
                    data = file_path.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", mime_type)
                    self.send_header("Content-Length", str(len(data)))
                    self._send_cors_headers()
                    self.end_headers()
                    self.wfile.write(data)
                    return True
                except OSError:
                    pass
            return False

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            clean_path = parsed.path

            # API Status check
            if clean_path == "/api/status":
                self._send_json(
                    200,
                    {
                        "status": "ok",
                        "version": "2.2.2",
                        "cwd": str(Path.cwd()),
                        "python_version": sys.version,
                        "platform": sys.platform,
                    },
                )
                return

            # Static file serving
            if dist_dir and dist_dir.is_dir():
                rel_path = clean_path.lstrip("/")
                target_file = dist_dir / rel_path if rel_path else dist_dir / "index.html"

                if target_file.is_file():
                    if self._serve_file(target_file):
                        return

                # SPA fallback: if file not found, serve index.html
                index_file = dist_dir / "index.html"
                if index_file.is_file():
                    if self._serve_file(index_file):
                        return

            # Fallback inline studio HTML if dist is missing
            self._send_fallback_html()

        def do_POST(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            clean_path = parsed.path

            if clean_path == "/api/init":
                try:
                    content_len = int(self.headers.get("Content-Length", 0))
                    body = self.rfile.read(content_len) if content_len > 0 else b"{}"
                    data = json.loads(body.decode("utf-8")) if body else {}

                    cmd_run_dir = Path.cwd().resolve()

                    raw_name = str(data.get("project_name") or "my_ai").strip()
                    if raw_name != "." and raw_name.startswith("/"):
                        raw_name = raw_name.lstrip("/")
                    proj_name = raw_name or "my_ai"

                    raw_target = str(data.get("target_dir") or ".").strip()
                    target_path = Path(raw_target)
                    if target_path.is_absolute():
                        try:
                            resolved_target = target_path.resolve()
                            if (resolved_target.parent == Path("/") or resolved_target == Path("/")) and not os.access("/", os.W_OK):
                                effective_target = (cmd_run_dir / raw_target.lstrip("/")).resolve()
                            else:
                                effective_target = resolved_target
                        except Exception:
                            effective_target = (cmd_run_dir / raw_target.lstrip("/")).resolve()
                    else:
                        effective_target = (cmd_run_dir / raw_target).resolve()

                    template_type = data.get("template_type") or "scratch"
                    chat_provider = data.get("chat_provider")
                    model_name = data.get("model_name")
                    vector_db = data.get("vector_db")
                    embedding_engine = data.get("embedding_engine")
                    clean = bool(data.get("clean", False))
                    create_venv = bool(data.get("create_venv", True))
                    install_deps = bool(data.get("install_deps", False))

                    extra_packages = data.get("extra_packages") or []

                    # Execute zero-path project initialization anchored to command run dir
                    exit_code = run_init(
                        project_name=proj_name,
                        target_dir=str(effective_target),
                        create_venv=create_venv,
                        template_type=template_type,
                        chat_provider=chat_provider,
                        vector_db=vector_db,
                        embedding_engine=embedding_engine,
                        model_name=model_name,
                        interactive=False,
                        install_deps=install_deps,
                        clean=clean,
                        extra_packages=extra_packages,
                    )

                    if exit_code != 0:
                        self._send_json(
                            500,
                            {
                                "status": "error",
                                "message": f"Initialization failed with exit code {exit_code}",
                            },
                        )
                        return

                    # Compute dest directory
                    dest_dir = effective_target if proj_name == "." else (effective_target / proj_name)

                    # Enumerate generated files
                    scaffolded_files: List[str] = []
                    if dest_dir.is_dir():
                        for item in sorted(dest_dir.rglob("*")):
                            if ".venv" not in str(item) and "__pycache__" not in str(item):
                                rel = item.relative_to(dest_dir)
                                scaffolded_files.append(str(rel))

                    if dest_dir == cmd_run_dir:
                        cd_cmd = "# (already in project directory)"
                    else:
                        try:
                            rel_cd = dest_dir.relative_to(cmd_run_dir)
                            cd_cmd = f"cd {rel_cd}"
                        except ValueError:
                            cd_cmd = f"cd {dest_dir}"

                    self._send_json(
                        200,
                        {
                            "status": "success",
                            "project_name": proj_name,
                            "dest_dir": str(dest_dir),
                            "template": template_type,
                            "files": scaffolded_files,
                            "commands": [
                                cd_cmd,
                                "aimlite train",
                                "aimlite serve --port 8000",
                            ],
                        },
                    )

                    # Automatically close server and terminal upon project creation
                    if getattr(self.server, "auto_shutdown", True):
                        def _delayed_shutdown(srv):
                            time.sleep(0.5)
                            try:
                                srv.shutdown()
                            except Exception:
                                pass

                        threading.Thread(target=_delayed_shutdown, args=(self.server,), daemon=True).start()
                except Exception as e:
                    self._send_json(500, {"status": "error", "message": str(e)})
                return

            self._send_json(404, {"status": "error", "message": f"Endpoint {clean_path} not found"})

        def _send_fallback_html(self) -> None:
            """Serves complete multi-step studio HTML with all CLI steps."""
            html = _load_studio_html()
            data = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(data)

    return StudioHTTPHandler


def _load_studio_html() -> str:
    """Loads studio.html supporting zipapp archives, package resources, and file paths."""
    # 1. Package resources (supports wheels and zipapps)
    try:
        import pkgutil
        data = pkgutil.get_data("aimlite.templates", "studio.html")
        if data:
            return data.decode("utf-8")
    except Exception:
        pass

    try:
        from importlib import resources
        return resources.files("aimlite.templates").joinpath("studio.html").read_text(encoding="utf-8")
    except Exception:
        pass

    # 2. Filesystem paths
    cand = Path(__file__).resolve().parent.parent / "templates" / "studio.html"
    if cand.is_file():
        return cand.read_text(encoding="utf-8")

    cand2 = Path.cwd() / "src" / "aimlite" / "templates" / "studio.html"
    if cand2.is_file():
        return cand2.read_text(encoding="utf-8")

    return "<html><body><h1>AIMLite Studio</h1><p>Template not found.</p></body></html>"



def run_studio(
    host: str = "127.0.0.1",
    port: int = 8000,
    open_browser: bool = True,
    initial_params: Optional[Dict[str, Any]] = None,
) -> int:
    """Launches the locally hosted AIMLite Studio Web Server & Browser interface.

    Args:
        host: Host address (default 127.0.0.1).
        port: Requested port (default 8000). If occupied, automatically seeks next open port.
        open_browser: Whether to launch the default web browser automatically.
        initial_params: Optional pre-configured parameters to inject into the Studio URL.

    Returns:
        Process exit code (0 on clean shutdown).
    """
    actual_port = _find_available_port(port)
    dist_dir = _resolve_static_dist_dir()
    handler_class = create_studio_handler(dist_dir)

    server = ThreadingStudioServer((host, actual_port), handler_class)

    # Build launch URL
    query_params: Dict[str, str] = {"view": "studio"}
    if initial_params:
        if initial_params.get("name"):
            query_params["name"] = str(initial_params["name"])
        if initial_params.get("template_type"):
            query_params["type"] = str(initial_params["template_type"])
        if initial_params.get("chat_provider"):
            query_params["chat_provider"] = str(initial_params["chat_provider"])
        if initial_params.get("clean"):
            query_params["clean"] = "true"

    query_str = urllib.parse.urlencode(query_params)
    display_url = f"http://{host}:{actual_port}/?{query_str}"
    network_url = f"http://localhost:{actual_port}/?{query_str}"

    print(vite_header())
    print(f"  {C.BOLD}{C.GREEN}AIMLite Studio{C.RESET} {C.DIM}v2.2.2 — Locally Hosted Web Interface{C.RESET}\n")
    print(f"  {C.CYAN}➜{C.RESET}  {C.BOLD}Local:{C.RESET}   {C.UNDERLINE}{display_url}{C.RESET}")
    print(f"  {C.CYAN}➜{C.RESET}  {C.DIM}Network: use --host 0.0.0.0 to expose{C.RESET}")
    print(f"  {C.CYAN}➜{C.RESET}  {C.DIM}Mode:    Project Creator & Architectural Studio{C.RESET}\n")
    print(f"  {C.DIM}Press {C.RESET}{C.BOLD}Ctrl+C{C.RESET}{C.DIM} to stop the server{C.RESET}\n")

    if open_browser:
        # Launch browser in a background thread after a brief delay
        def _open():
            time.sleep(0.4)
            try:
                webbrowser.open(display_url)
            except Exception:
                pass

        threading.Thread(target=_open, daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\n  {C.YELLOW}Stopping AIMLite Studio...{C.RESET}")
    finally:
        server.server_close()

    print(f"\n  {check('Project scaffolding complete.')}")
    print(f"  {C.DIM}Studio session finished. Terminal closed.{C.RESET}\n")
    return 0
