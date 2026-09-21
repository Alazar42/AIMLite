"""Developer-Customizable HTTP Inference & Frontend Server for AIMLite."""

from __future__ import annotations

import json
import mimetypes
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

from cli.ui import C, arrow, cross, vite_header
from aimlite.lifecycle import BaseInference
from aimlite.models import Model


def _load_template(filename: str) -> str:
    """Loads HTML template from aimlite package or fallback directory."""
    try:
        from importlib import resources

        return resources.files("aimlite.templates").joinpath(filename).read_text(encoding="utf-8")
    except Exception:
        pass

    fallback_path = Path(__file__).resolve().parent.parent / "aimlite" / "templates" / filename
    if fallback_path.is_file():
        return fallback_path.read_text(encoding="utf-8")

    return f"<html><body><h1>AIMLite Server</h1><p>Template {filename} not found.</p></body></html>"


def get_openapi_schema(model_name: str, routes: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Generates OpenAPI 3.0 schema dynamically reflecting standard and custom endpoints."""
    paths: Dict[str, Any] = {
        "/predict": {
            "post": {
                "summary": "Execute Model Inference",
                "description": "Submits input features to execute live model predictions.",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"type": "object"},
                            "example": {"features": [1.0, 2.0, 3.0]},
                        },
                    },
                },
                "responses": {
                    "200": {"description": "Inference success"},
                    "400": {"description": "Malformed request"},
                },
            },
        },
        "/health": {
            "get": {
                "summary": "Server Health Check",
                "responses": {"200": {"description": "Server is healthy"}},
            },
        },
    }

    # Add custom endpoints from developer routes
    if routes:
        for route_sig in routes.keys():
            parts = route_sig.strip().split(" ", 1)
            if len(parts) == 2:
                method, path = parts[0].lower(), parts[1]
                if path not in paths:
                    paths[path] = {}
                paths[path][method] = {
                    "summary": f"Custom endpoint: {route_sig}",
                    "responses": {"200": {"description": "Success"}},
                }

    return {
        "openapi": "3.0.0",
        "info": {
            "title": f"AIMLite API - {model_name}",
            "version": "1.0.3",
            "description": "Developer-customizable inference server and frontend host.",
        },
        "paths": paths,
    }


class DefaultInference(BaseInference):
    """Fallback inference runner if none is declared by the developer."""

    def run(self, model: Model, raw_input: Any, **kwargs: Any) -> Any:
        if isinstance(raw_input, dict) and "features" in raw_input:
            inp = raw_input["features"]
        else:
            inp = raw_input
        preds = model.predict(inp)
        if hasattr(preds, "tolist"):
            return preds.tolist()
        return preds


def create_handler_class(
    model: Model,
    inference: Optional[BaseInference] = None,
    model_name: str = "model",
    frontend_dir: Optional[Path] = None,
):
    """Creates a configured RequestHandler supporting custom routes and frontend hosting."""
    active_inference = inference or DefaultInference()

    # Collect custom developer routes from BaseInference
    custom_routes: Dict[str, Callable] = {}
    if hasattr(active_inference, "get_routes"):
        try:
            custom_routes = active_inference.get_routes() or {}
        except Exception:
            custom_routes = {}

    class AIMLiteHTTPHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            code = args[1] if len(args) > 1 else "200"
            code_color = C.GREEN if str(code).startswith("2") else (C.RED if str(code).startswith("4") or str(code).startswith("5") else C.YELLOW)
            method = args[0].split()[0] if len(args) > 0 and " " in str(args[0]) else "REQ"
            url = args[0].split()[1] if len(args) > 0 and len(str(args[0]).split()) > 1 else ""
            print(f"  {C.DIM}[{self.log_date_time_string()}]{C.RESET} {C.CYAN}{method:<4}{C.RESET} {code_color}{code}{C.RESET} {url}")

        def _send_cors_headers(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With")

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self._send_cors_headers()
            self.end_headers()

        def _dispatch_handler(self, handler: Callable, payload: Any = None) -> None:
            """Executes developer handler and serializes response cleanly."""
            start_t = time.perf_counter()
            try:
                # Support both handler(model, payload) and handler(payload)
                try:
                    res = handler(model, payload)
                except TypeError:
                    res = handler(payload) if payload is not None else handler()
            except Exception as e:
                err = json.dumps({"status": "error", "message": str(e)}).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(err)))
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(err)
                return

            status_code = 200
            if isinstance(res, tuple) and len(res) == 2:
                res, status_code = res

            # Serialize output
            if isinstance(res, (dict, list)):
                data = json.dumps(res).encode("utf-8")
                content_type = "application/json; charset=utf-8"
            elif isinstance(res, str):
                data = res.encode("utf-8")
                content_type = "text/html; charset=utf-8" if "<html" in res.lower() else "text/plain; charset=utf-8"
            elif isinstance(res, bytes):
                data = res
                content_type = "application/octet-stream"
            else:
                data = json.dumps({"result": res}).encode("utf-8")
                content_type = "application/json; charset=utf-8"

            self.send_response(status_code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(data)

        def _serve_static_file(self, file_path: Path) -> bool:
            """Serves a static file if present."""
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
            path = self.path.split("?")[0]
            route_key = f"GET {path}"

            # 1. Check custom developer route
            if route_key in custom_routes:
                self._dispatch_handler(custom_routes[route_key])
                return

            # 2. Built-in Interactive Chat Playground
            if path in ["/chat", "/playground"]:
                html = _load_template("chat.html")
                data = html.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(data)
                return

            # 3. Built-in Swagger Docs & OpenAPI
            if path == "/docs":
                html = _load_template("swagger.html")
                data = html.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(data)
                return

            if path == "/openapi.json":
                schema = get_openapi_schema(model_name, custom_routes)
                data = json.dumps(schema, indent=2).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(data)
                return

            # 4. Default Health probe
            if path == "/health":
                payload = {"status": "healthy", "model": model_name}
                self._dispatch_handler(lambda: payload)
                return

            # 5. Custom Frontend Serving (if configured)
            if frontend_dir and frontend_dir.is_dir():
                rel_path = path.lstrip("/")
                target_file = frontend_dir / rel_path if rel_path else frontend_dir / "index.html"

                if self._serve_static_file(target_file):
                    return

                # SPA fallback: If no file extension, serve index.html
                if "." not in path.split("/")[-1]:
                    fallback_index = frontend_dir / "index.html"
                    if self._serve_static_file(fallback_index):
                        return

            # 6. Default API root (when no custom frontend is provided)
            if path in ["", "/"]:
                accept_header = self.headers.get("Accept", "")
                # If accessed via a web browser, serve the interactive Chat Playground
                if "text/html" in accept_header:
                    html = _load_template("chat.html")
                    data = html.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(data)))
                    self._send_cors_headers()
                    self.end_headers()
                    self.wfile.write(data)
                    return

                endpoints = {
                    "chat": f"GET /chat (Interactive Web Playground)",
                    "predict": f"POST /predict",
                    "health": f"GET /health",
                    "docs": f"GET /docs",
                    "openapi": f"GET /openapi.json",
                }
                payload = {
                    "name": model_name,
                    "status": "online",
                    "version": "1.0.3",
                    "endpoints": endpoints,
                }
                data = json.dumps(payload, indent=2).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(data)
                return

            if path == "/predict":
                payload = {
                    "endpoint": "POST /predict",
                    "description": "Submit a POST request with JSON payload to execute live model inference.",
                    "method": "POST",
                }
                data = json.dumps(payload, indent=2).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(data)
                return

            # 404
            msg = json.dumps({"error": f"Endpoint '{path}' not found"}).encode("utf-8")
            self.send_response(404)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(msg)))
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(msg)

        def do_POST(self) -> None:
            path = self.path.split("?")[0]
            route_key = f"POST {path}"

            # Read request body
            content_length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(content_length) if content_length > 0 else b"{}"

            content_type = self.headers.get("Content-Type", "")
            if "application/json" in content_type or raw_body.startswith(b"{") or raw_body.startswith(b"["):
                try:
                    payload = json.loads(raw_body.decode("utf-8")) if raw_body else {}
                except json.JSONDecodeError as e:
                    err = json.dumps({"error": f"Malformed JSON: {e}"}).encode("utf-8")
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(err)))
                    self._send_cors_headers()
                    self.end_headers()
                    self.wfile.write(err)
                    return
            else:
                try:
                    payload = raw_body.decode("utf-8")
                except UnicodeDecodeError:
                    payload = raw_body

            # 1. Custom developer route
            if route_key in custom_routes:
                self._dispatch_handler(custom_routes[route_key], payload)
                return

            # 2. Standard /predict endpoint
            if path == "/predict":
                start_t = time.perf_counter()
                try:
                    result = active_inference.run(model, payload)
                except Exception as e:
                    err = json.dumps({"status": "error", "message": str(e)}).encode("utf-8")
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(err)))
                    self._send_cors_headers()
                    self.end_headers()
                    self.wfile.write(err)
                    return

                latency = round((time.perf_counter() - start_t) * 1000, 2)
                response_payload = {
                    "status": "success",
                    "model": model_name,
                    "latency_ms": latency,
                    "result": result,
                }
                data = json.dumps(response_payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(data)
                return

            msg = json.dumps({"error": f"Endpoint '{path}' not found"}).encode("utf-8")
            self.send_response(404)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(msg)))
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(msg)

    return AIMLiteHTTPHandler


def run_inference_server(
    model: Model,
    inference: Optional[BaseInference] = None,
    model_name: str = "model",
    host: str = "127.0.0.1",
    port: int = 8000,
    frontend_dir: Optional[Path] = None,
    custom_app: Optional[Any] = None,
    checkpoint_path: Optional[Path] = None,
) -> int:
    """Starts the HTTP inference and frontend server.

    Args:
        model: Active Model instance.
        inference: Custom BaseInference instance.
        model_name: Project/model identifier.
        host: Listening host.
        port: Listening port.
        frontend_dir: Optional custom frontend directory (e.g. dist/, frontend/).
        custom_app: Optional WSGI app or custom server callable.
        checkpoint_path: Optional path to loaded model checkpoint.

    Returns:
        Exit code (0 on clean shutdown, 1 on error).
    """
    # If developer defined a custom serve callable
    if custom_app is not None:
        if callable(custom_app) and not hasattr(custom_app, "__call__"):
            try:
                custom_app(host=host, port=port)
                return 0
            except Exception:
                pass

        # If WSGI app (e.g. Flask or FastAPI/Starlette WSGI adapter)
        if callable(custom_app):
            try:
                from wsgiref.simple_server import make_server

                print(vite_header("ready in custom app mode"))
                print(arrow("Local", f"http://{host}:{port}/"))
                print(arrow("App", getattr(custom_app, "__name__", "Custom WSGI App")))
                print(f"\n  {C.DIM}press Ctrl+C to terminate{C.RESET}\n")
                wsgi_server = make_server(host, port, custom_app)
                wsgi_server.serve_forever()
                return 0
            except Exception as e:
                print(f"  {C.YELLOW}! Could not launch custom WSGI app ({e}); falling back to AIMLite server.{C.RESET}")

    handler_cls = create_handler_class(
        model=model,
        inference=inference,
        model_name=model_name,
        frontend_dir=frontend_dir,
    )

    try:
        server = HTTPServer((host, port), handler_cls)
    except OSError as e:
        if "Address already in use" in str(e) or e.errno == 98:
            print(f"\n{cross(f'Port {port} is already in use.')}")
            print(f"  {C.CYAN}Try a different port:{C.RESET} aimlite serve --port {port + 1}\n")
            return 1
        print(f"\n{cross(f'Failed to bind server to http://{host}:{port}: {e}')}\n")
        return 1

    print(vite_header("serve"))
    print(arrow("Target", model_name))
    if checkpoint_path:
        print(arrow("Checkpoint", str(checkpoint_path)))
    print(arrow("Local API", f"http://{host}:{port}/"))
    print(arrow("Chat UI", f"http://{host}:{port}/chat"))
    print(arrow("Inference", f"POST http://{host}:{port}/predict"))
    print(arrow("Health", f"http://{host}:{port}/health"))
    print(arrow("Docs", f"http://{host}:{port}/docs"))

    if frontend_dir and frontend_dir.is_dir():
        print(arrow("Frontend", f"{frontend_dir} (custom UI served at /)"))

    # Display custom developer routes if any
    if inference and hasattr(inference, "get_routes"):
        try:
            custom_routes = inference.get_routes()
            if custom_routes:
                for sig in custom_routes:
                    if sig not in ["POST /predict", "POST /", "GET /health", "GET /info"]:
                        print(arrow("Custom API", sig))
        except Exception:
            pass

    print(f"\n  {C.DIM}press Ctrl+C to terminate{C.RESET}\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\n  {C.DIM}Shutting down inference server...{C.RESET}\n")
    finally:
        server.server_close()

    return 0
