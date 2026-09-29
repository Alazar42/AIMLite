"""Developer-Customizable HTTP Inference & Frontend Server for AIMLite."""

from __future__ import annotations

import json
import mimetypes
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from cli.ui import C, arrow, cross, vite_header
from aimlite.lifecycle import BaseInference
from aimlite.models import Model


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """HTTP server that handles each request in a separate thread.

    This prevents a slow inference call from blocking other concurrent requests.
    ThreadingMixIn spawns a new thread for each incoming connection.
    """

    daemon_threads = True


def _load_template(filename: str, template_dir: Optional[Path] = None) -> str:
    """Loads HTML template with developer project override priority."""
    # 1. Developer's project templates directory (<project_root>/templates/<filename>)
    if template_dir is not None and template_dir.is_dir():
        custom_file = template_dir / filename
        if custom_file.is_file():
            return custom_file.read_text(encoding="utf-8")
        # Allow templates/index.html to satisfy app.html
        if filename in ("app.html", "chat.html"):
            index_custom = template_dir / "index.html"
            if index_custom.is_file():
                return index_custom.read_text(encoding="utf-8")

    # 2. Package resources
    try:
        from importlib import resources

        return resources.files("aimlite.templates").joinpath(filename).read_text(encoding="utf-8")
    except Exception:
        pass

    # 3. Direct relative path fallback
    fallback_path = Path(__file__).resolve().parent.parent / "aimlite" / "templates" / filename
    if fallback_path.is_file():
        return fallback_path.read_text(encoding="utf-8")

    # 4. Secondary fallback: if app.html requested but not found, try chat.html
    if filename == "app.html":
        chat_fallback = Path(__file__).resolve().parent.parent / "aimlite" / "templates" / "chat.html"
        if chat_fallback.is_file():
            return chat_fallback.read_text(encoding="utf-8")

    return f"<html><body><h1>AIMLite Server</h1><p>Template {filename} not found.</p></body></html>"


def get_openapi_schema(
    models: Dict[str, Dict[str, Any]],
    active_model_name: str,
    routes: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generates OpenAPI 3.0 schema dynamically reflecting registered models and endpoints."""
    paths: Dict[str, Any] = {
        "/predict": {
            "post": {
                "summary": f"Execute Default Model Inference ({active_model_name})",
                "description": "Submits input payload to the active default model.",
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
        "/models": {
            "get": {
                "summary": "List Registered Models",
                "description": "Returns metadata and endpoints for all registered and trained model classes.",
                "responses": {"200": {"description": "Model inventory list"}},
            },
        },
        "/health": {
            "get": {
                "summary": "Server Health Check",
                "responses": {"200": {"description": "Server is healthy"}},
            },
        },
    }

    # Add per-model endpoints
    for m_name, m_info in models.items():
        m_type = m_info.get("type", "ml")
        example_payload = (
            {"query": "Sample question for knowledge search", "top_k": 3}
            if m_type == "rag"
            else (
                {"prompt": "Generate response", "max_tokens": 128}
                if m_type == "adapter"
                else {"features": [1.0, 2.0, 3.0]}
            )
        )
        paths[f"/models/{m_name}/predict"] = {
            "post": {
                "summary": f"Inference for {m_name} ({m_type.upper()})",
                "description": m_info.get("description", f"Executes inference using model class {m_name}."),
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"type": "object"},
                            "example": example_payload,
                        },
                    },
                },
                "responses": {
                    "200": {"description": "Inference success"},
                    "404": {"description": "Model not found"},
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
            "title": f"AIMLite API — {active_model_name}",
            "version": "2.0.0",
            "description": "Developer-customizable zero-path multi-model inference server.",
        },
        "paths": paths,
    }


class DefaultInference:
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
    template_dir: Optional[Path] = None,
    registered_models: Optional[Dict[str, Dict[str, Any]]] = None,
    active_model_name: Optional[str] = None,
    api_only: bool = False,
):
    """Creates a configured RequestHandler supporting multi-model routing, custom routes and frontend hosting."""
    active_inference = inference or DefaultInference()
    models_dict = registered_models or {
        model_name: {
            "class": type(model),
            "instance": model,
            "type": "ml",
            "description": type(model).__doc__ or f"{model_name} model",
            "checkpoint": None,
            "features": [],
            "inference": active_inference,
        }
    }
    primary_name = active_model_name or model_name

    # Collect custom developer routes from BaseInference
    custom_routes: Dict[str, Callable] = {}
    if hasattr(active_inference, "get_routes"):
        try:
            custom_routes = active_inference.get_routes() or {}
        except Exception:
            custom_routes = {}

    def get_model_entry(name: Optional[str]) -> Optional[Dict[str, Any]]:
        if not name:
            return models_dict.get(primary_name)
        if name in models_dict:
            return models_dict[name]
        # Case-insensitive search
        name_lower = name.lower()
        for k, v in models_dict.items():
            if k.lower() == name_lower or k.lower().replace("model", "") == name_lower:
                return v
        return None

    def execute_model_inference(m_entry: Dict[str, Any], payload: Any) -> Any:
        m_inst = m_entry["instance"]
        m_inf = m_entry.get("inference") or active_inference
        # Allow inference handler or direct model predict
        if hasattr(m_inf, "run"):
            return m_inf.run(m_inst, payload)
        # Direct fallback
        if isinstance(payload, dict) and "features" in payload:
            return m_inst.predict(payload["features"])
        return m_inst.predict(payload)

    class AIMLiteHTTPHandler(BaseHTTPRequestHandler):
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
            self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With")

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self._send_cors_headers()
            self.end_headers()

        def _send_json(self, status_code: int, data_obj: Any) -> None:
            data = json.dumps(data_obj, indent=2 if status_code == 200 else None).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(data)

        def _send_html(self, html_content: str, status_code: int = 200) -> None:
            data = html_content.encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(data)

        def _dispatch_handler(self, handler: Callable, payload: Any = None) -> None:
            """Executes developer handler and serializes response cleanly."""
            try:
                try:
                    res = handler(model, payload)
                except TypeError:
                    res = handler(payload) if payload is not None else handler()
            except Exception as e:
                self._send_json(500, {"status": "error", "message": str(e)})
                return

            status_code = 200
            if isinstance(res, tuple) and len(res) == 2:
                res, status_code = res

            if isinstance(res, (dict, list)):
                self._send_json(status_code, res)
            elif isinstance(res, str):
                content_type = "text/html; charset=utf-8" if "<html" in res.lower() else "text/plain; charset=utf-8"
                data = res.encode("utf-8")
                self.send_response(status_code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(data)
            elif isinstance(res, bytes):
                self.send_response(status_code)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(res)))
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(res)
            else:
                self._send_json(status_code, {"result": res})

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

            # 1. Custom developer route
            if route_key in custom_routes:
                self._dispatch_handler(custom_routes[route_key])
                return

            # 2. Multi-model list endpoint: GET /models
            if path == "/models":
                models_list = []
                for name, info in models_dict.items():
                    models_list.append({
                        "name": name,
                        "type": info.get("type", "ml"),
                        "description": info.get("description", ""),
                        "checkpoint": info.get("checkpoint"),
                        "features": info.get("features", []),
                        "active": (name == primary_name),
                    })
                self._send_json(200, models_list)
                return

            # 2b. Specific model metadata: GET /models/<name>
            if path.startswith("/models/") and len(path.split("/")) == 3:
                req_model = path.split("/")[2]
                entry = get_model_entry(req_model)
                if entry:
                    self._send_json(200, {
                        "name": req_model,
                        "type": entry.get("type", "ml"),
                        "description": entry.get("description", ""),
                        "checkpoint": entry.get("checkpoint"),
                        "features": entry.get("features", []),
                        "active": (req_model == primary_name),
                    })
                    return
                self._send_json(404, {"error": f"Model '{req_model}' not found"})
                return

            # 3. Headless API Mode: if --api is enabled, disallow web UI
            if api_only:
                if path in ["/chat", "/playground", "/app"]:
                    self._send_json(403, {
                        "status": "headless",
                        "message": "Web UI is disabled in headless --api mode. Use REST API endpoints.",
                        "endpoints": {
                            "models": "GET /models",
                            "predict": "POST /predict",
                            "health": "GET /health",
                            "docs": "GET /docs",
                            "openapi": "GET /openapi.json",
                        },
                    })
                    return

                if path in ["", "/"]:
                    payload = {
                        "name": "AIMLite Server",
                        "status": "online",
                        "mode": "headless_api",
                        "active_model": primary_name,
                        "models": list(models_dict.keys()),
                        "endpoints": {
                            "models": "GET /models",
                            "predict": "POST /predict",
                            "health": "GET /health",
                            "docs": "GET /docs",
                            "openapi": "GET /openapi.json",
                        },
                    }
                    self._send_json(200, payload)
                    return

            # 4. Built-in or Overridden Web Application: GET /chat, /app, /playground
            if path in ["/chat", "/playground", "/app"]:
                html = _load_template("app.html", template_dir=template_dir)
                self._send_html(html)
                return

            # 5. Built-in Swagger Docs & OpenAPI
            if path == "/docs":
                html = _load_template("swagger.html", template_dir=template_dir)
                self._send_html(html)
                return

            if path == "/openapi.json":
                schema = get_openapi_schema(models_dict, primary_name, custom_routes)
                self._send_json(200, schema)
                return

            # 6. Default Health probe
            if path == "/health":
                payload = {
                    "status": "healthy",
                    "model": primary_name,
                    "type": models_dict.get(primary_name, {}).get("type", "ml"),
                    "models": list(models_dict.keys()),
                    "api_only": api_only,
                }
                self._send_json(200, payload)
                return

            # 7. Custom Frontend Serving (if configured by developer)
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

            # 8. Root URL (GET /)
            if path in ["", "/"]:
                accept_header = self.headers.get("Accept", "")
                # If accessed via a browser, serve the adaptive web app
                if "text/html" in accept_header:
                    html = _load_template("app.html", template_dir=template_dir)
                    self._send_html(html)
                    return

                # If requested via curl/json API client, return service directory
                endpoints = {
                    "app": "GET /app (Interactive Web Interface)",
                    "models": "GET /models (Registered Models Inventory)",
                    "predict": "POST /predict",
                    "health": "GET /health",
                    "docs": "GET /docs",
                    "openapi": "GET /openapi.json",
                }
                payload = {
                    "name": primary_name,
                    "status": "online",
                    "version": "2.0.0",
                    "active_model": primary_name,
                    "models": list(models_dict.keys()),
                    "endpoints": endpoints,
                }
                self._send_json(200, payload)
                return

            if path == "/predict":
                payload = {
                    "endpoint": "POST /predict",
                    "description": "Submit a POST request with JSON payload to execute live model inference.",
                    "method": "POST",
                }
                self._send_json(200, payload)
                return

            # 404
            self._send_json(404, {"error": f"Endpoint '{path}' not found"})

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
                    self._send_json(400, {"error": f"Malformed JSON: {e}"})
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

            # 2. Specific Model Inference: POST /models/<model_name>/predict
            if path.startswith("/models/") and path.endswith("/predict"):
                parts = path.strip("/").split("/")
                if len(parts) == 3:
                    target_model_name = parts[1]
                    m_entry = get_model_entry(target_model_name)
                    if not m_entry:
                        self._send_json(404, {"error": f"Model '{target_model_name}' not found"})
                        return

                    start_t = time.perf_counter()
                    try:
                        result = execute_model_inference(m_entry, payload)
                    except Exception as e:
                        self._send_json(500, {"status": "error", "model": target_model_name, "message": str(e)})
                        return

                    latency = round((time.perf_counter() - start_t) * 1000, 2)
                    self._send_json(200, {
                        "status": "success",
                        "model": target_model_name,
                        "type": m_entry.get("type", "ml"),
                        "latency_ms": latency,
                        "result": result,
                    })
                    return

            # 3. Standard /predict endpoint (dispatches to target model or default model)
            if path == "/predict":
                # Check if specific model was requested in body payload: {"model": "...", ...}
                target_m_name = primary_name
                if isinstance(payload, dict) and "model" in payload and payload["model"] in models_dict:
                    target_m_name = payload["model"]

                m_entry = get_model_entry(target_m_name)
                if not m_entry:
                    self._send_json(404, {"error": f"Target model '{target_m_name}' not found"})
                    return

                start_t = time.perf_counter()
                try:
                    result = execute_model_inference(m_entry, payload)
                except Exception as e:
                    self._send_json(500, {"status": "error", "model": target_m_name, "message": str(e)})
                    return

                latency = round((time.perf_counter() - start_t) * 1000, 2)
                self._send_json(200, {
                    "status": "success",
                    "model": target_m_name,
                    "type": m_entry.get("type", "ml"),
                    "latency_ms": latency,
                    "result": result,
                })
                return

            self._send_json(404, {"error": f"Endpoint '{path}' not found"})

    return AIMLiteHTTPHandler


def run_inference_server(
    model: Model,
    inference: Optional[BaseInference] = None,
    model_name: str = "model",
    host: str = "127.0.0.1",
    port: int = 8000,
    frontend_dir: Optional[Path] = None,
    template_dir: Optional[Path] = None,
    custom_app: Optional[Any] = None,
    checkpoint_path: Optional[Path] = None,
    registered_models: Optional[Dict[str, Dict[str, Any]]] = None,
    active_model_name: Optional[str] = None,
    api_only: bool = False,
) -> int:
    """Starts the HTTP inference and frontend server.

    Args:
        model: Active Model instance.
        inference: Custom BaseInference instance.
        model_name: Project/model identifier.
        host: Listening host.
        port: Listening port.
        frontend_dir: Optional custom frontend directory (e.g. dist/, frontend/).
        template_dir: Optional custom templates directory (<project_root>/templates/).
        custom_app: Optional WSGI app or custom server callable.
        checkpoint_path: Optional path to loaded model checkpoint.
        registered_models: Dictionary of all ready model entries.
        active_model_name: Name of the primary active model.
        api_only: If True, operates in headless mode without HTML pages.

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

    all_models = registered_models or {
        model_name: {
            "class": type(model),
            "instance": model,
            "type": "ml",
            "description": type(model).__doc__ or f"{model_name} model",
            "checkpoint": str(checkpoint_path) if checkpoint_path else None,
            "features": [],
            "inference": inference,
        }
    }
    primary_name = active_model_name or model_name

    handler_cls = create_handler_class(
        model=model,
        inference=inference,
        model_name=model_name,
        frontend_dir=frontend_dir,
        template_dir=template_dir,
        registered_models=all_models,
        active_model_name=primary_name,
        api_only=api_only,
    )

    try:
        server = ThreadingHTTPServer((host, port), handler_cls)
    except OSError as e:
        if "Address already in use" in str(e) or e.errno == 98:
            print(f"\n{cross(f'Port {port} is already in use.')}")
            print(f"  {C.CYAN}Try a different port:{C.RESET} aimlite serve --port {port + 1}\n")
            return 1
        print(f"\n{cross(f'Failed to bind server to http://{host}:{port}: {e}')}\n")
        return 1

    header_title = "serve (api only)" if api_only else "serve"
    print(vite_header(header_title))
    print(arrow("Primary Model", f"{primary_name} ({all_models.get(primary_name, {}).get('type', 'ml').upper()})"))

    if len(all_models) > 1:
        model_summary = ", ".join(f"{k} [{v.get('type', 'ml')}]" for k, v in all_models.items())
        print(arrow("Registered", model_summary))

    if checkpoint_path:
        print(arrow("Checkpoint", str(checkpoint_path)))

    if api_only:
        print(arrow("Mode", "Headless REST API (--api)"))
        print(arrow("API Directory", f"http://{host}:{port}/"))
        print(arrow("Models List", f"http://{host}:{port}/models"))
        print(arrow("Inference", f"POST http://{host}:{port}/predict"))
    else:
        print(arrow("Web App", f"http://{host}:{port}/"))
        print(arrow("Models API", f"http://{host}:{port}/models"))
        print(arrow("Inference", f"POST http://{host}:{port}/predict"))
        print(arrow("Docs", f"http://{host}:{port}/docs"))

    if template_dir and template_dir.is_dir():
        print(arrow("Custom Templates", f"{template_dir} (active overrides)"))

    if frontend_dir and frontend_dir.is_dir():
        print(arrow("Custom Frontend", f"{frontend_dir} (served at /)"))

    # Display custom developer routes if any
    if inference and hasattr(inference, "get_routes"):
        try:
            custom_routes = inference.get_routes()
            if custom_routes:
                for sig in custom_routes:
                    if sig not in ["POST /predict", "POST /", "GET /health", "GET /info"]:
                        print(arrow("Custom Route", sig))
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
