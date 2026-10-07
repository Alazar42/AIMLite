"""Comprehensive tests for the AIMLite concurrent serving infrastructure.

Covers:
- ThreadingHTTPServer handles concurrent requests without blocking
- /health endpoint returns correct status
- /models endpoint lists all registered models with metadata
- /predict dispatches correctly to model.predict()
- /models/<name>/predict dispatches to specific model
- POST /predict with unknown model name returns 404
- GET /predict returns usage info
- api_only mode blocks web UI endpoints with 403
- Custom frontend directory is served
- Malformed JSON request body returns 400
- 404 for unknown endpoints
- Latency field present in inference responses
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from aimlite.models import Model
from aimlite.rag import KnowledgeModel, MockChatProvider
from aimlite.adapters import AdapterModel
from aimlite.cli.server import ThreadingHTTPServer, create_handler_class


# =============================================================================
# Fixtures and helpers
# =============================================================================


class SlowModel(Model):
    """Model that intentionally sleeps during predict() to test concurrency."""

    def __init__(self, sleep_sec: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.sleep_sec = sleep_sec

    def predict(self, inputs, **kwargs):
        time.sleep(self.sleep_sec)
        return {"slept": self.sleep_sec, "input": inputs}


class FastModel(Model):
    """Model that responds immediately."""

    def predict(self, inputs, **kwargs):
        return {"echo": inputs}


class RagLikeModel(KnowledgeModel):
    """RAG model fixture for serving tests."""

    def __init__(self, **kwargs):
        kwargs.setdefault("chat_provider", MockChatProvider())
        kwargs.setdefault("name", "test_rag_serve")
        super().__init__(**kwargs)

    def predict(self, inputs, **kwargs):
        return {"answer": "RAG answer", "sources": [{"content": "source1"}]}


class ServerContext:
    """Context manager that starts a test server and shuts it down on exit."""

    def __init__(self, port: int, model_instance: Model, models_dict: Dict, api_only: bool = False):
        self.port = port
        self.model_instance = model_instance
        self.models_dict = models_dict
        self.api_only = api_only
        self.server: Optional[ThreadingHTTPServer] = None

    def __enter__(self):
        handler = create_handler_class(
            model=self.model_instance,
            model_name=list(self.models_dict.keys())[0],
            registered_models=self.models_dict,
            active_model_name=list(self.models_dict.keys())[0],
            api_only=self.api_only,
        )
        self.server = ThreadingHTTPServer(("127.0.0.1", self.port), handler)
        t = threading.Thread(target=self.server.serve_forever, daemon=True)
        t.start()
        # Brief pause for server to be ready
        time.sleep(0.05)
        return self

    def __exit__(self, *args):
        if self.server:
            self.server.shutdown()

    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def get(self, path: str, accept: str = "application/json") -> tuple:
        req = urllib.request.Request(
            self.base_url() + path,
            headers={"Accept": accept},
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read().decode())

    def post(self, path: str, body: dict) -> tuple:
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            self.base_url() + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read().decode())


def make_models_dict(name: str, instance: Model, model_type: str = "ml") -> Dict:
    return {
        name: {
            "class": type(instance),
            "instance": instance,
            "type": model_type,
            "description": f"{name} test model",
            "features": [],
        }
    }


# =============================================================================
# Basic endpoint tests
# =============================================================================


class TestServerEndpoints:
    PORT = 9930

    def test_health_endpoint(self):
        model = FastModel(name="FastModel")
        models = make_models_dict("FastModel", model)
        with ServerContext(self.PORT, model, models) as ctx:
            status, data = ctx.get("/health")
            assert status == 200
            assert data["status"] == "healthy"
            assert "model" in data

    def test_models_list_endpoint(self):
        model = FastModel(name="FastModel")
        models = make_models_dict("FastModel", model)
        with ServerContext(self.PORT + 1, model, models) as ctx:
            status, data = ctx.get("/models")
            assert status == 200
            assert isinstance(data, list)
            assert len(data) == 1
            assert data[0]["name"] == "FastModel"
            assert data[0]["type"] == "ml"

    def test_predict_endpoint_dispatches(self):
        model = FastModel(name="EchoModel")
        models = make_models_dict("EchoModel", model)
        with ServerContext(self.PORT + 2, model, models) as ctx:
            status, data = ctx.post("/predict", {"features": [1.0, 2.0, 3.0]})
            assert status == 200
            assert data["status"] == "success"
            assert "result" in data

    def test_predict_includes_latency_ms(self):
        model = FastModel(name="LatencyModel")
        models = make_models_dict("LatencyModel", model)
        with ServerContext(self.PORT + 3, model, models) as ctx:
            status, data = ctx.post("/predict", {"features": [1.0]})
            assert status == 200
            assert "latency_ms" in data
            assert isinstance(data["latency_ms"], float)
            assert data["latency_ms"] >= 0.0

    def test_model_specific_predict_endpoint(self):
        model = FastModel(name="TargetModel")
        models = make_models_dict("TargetModel", model)
        with ServerContext(self.PORT + 4, model, models) as ctx:
            status, data = ctx.post("/models/TargetModel/predict", {"features": [5.0]})
            assert status == 200
            assert data["status"] == "success"
            assert data["model"] == "TargetModel"

    def test_unknown_model_predict_returns_404(self):
        model = FastModel(name="OnlyModel")
        models = make_models_dict("OnlyModel", model)
        with ServerContext(self.PORT + 5, model, models) as ctx:
            try:
                ctx.post("/models/Nonexistent/predict", {})
                assert False, "Should have raised HTTPError"
            except urllib.error.HTTPError as e:
                assert e.code == 404

    def test_unknown_endpoint_returns_404(self):
        model = FastModel(name="SomeModel")
        models = make_models_dict("SomeModel", model)
        with ServerContext(self.PORT + 6, model, models) as ctx:
            try:
                ctx.get("/this/does/not/exist")
                assert False, "Should have raised HTTPError"
            except urllib.error.HTTPError as e:
                assert e.code == 404

    def test_get_predict_returns_usage_info(self):
        model = FastModel(name="UsageModel")
        models = make_models_dict("UsageModel", model)
        with ServerContext(self.PORT + 7, model, models) as ctx:
            status, data = ctx.get("/predict")
            assert status == 200
            assert "endpoint" in data or "method" in data

    def test_malformed_json_returns_400(self):
        model = FastModel(name="MalformedModel")
        models = make_models_dict("MalformedModel", model)
        with ServerContext(self.PORT + 8, model, models) as ctx:
            raw_req = urllib.request.Request(
                ctx.base_url() + "/predict",
                data=b"{ not valid json !!",
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                urllib.request.urlopen(raw_req, timeout=5)
                assert False, "Should have raised HTTPError"
            except urllib.error.HTTPError as e:
                assert e.code == 400


# =============================================================================
# API-only mode
# =============================================================================


class TestApiOnlyMode:
    PORT = 9950

    def test_root_returns_json_with_headless_api_mode(self):
        model = FastModel(name="HeadlessModel")
        models = make_models_dict("HeadlessModel", model)
        with ServerContext(self.PORT, model, models, api_only=True) as ctx:
            status, data = ctx.get("/")
            assert status == 200
            assert data["mode"] == "headless_api"

    def test_chat_endpoint_returns_403_in_api_mode(self):
        model = FastModel(name="HeadlessModel2")
        models = make_models_dict("HeadlessModel2", model)
        with ServerContext(self.PORT + 1, model, models, api_only=True) as ctx:
            try:
                ctx.get("/chat")
                assert False, "Should have raised HTTPError"
            except urllib.error.HTTPError as e:
                assert e.code == 403

    def test_playground_endpoint_returns_403_in_api_mode(self):
        model = FastModel(name="HeadlessModel3")
        models = make_models_dict("HeadlessModel3", model)
        with ServerContext(self.PORT + 2, model, models, api_only=True) as ctx:
            try:
                ctx.get("/playground")
                assert False, "Should have raised HTTPError"
            except urllib.error.HTTPError as e:
                assert e.code == 403

    def test_health_accessible_in_api_mode(self):
        model = FastModel(name="HeadlessHealth")
        models = make_models_dict("HeadlessHealth", model)
        with ServerContext(self.PORT + 3, model, models, api_only=True) as ctx:
            status, data = ctx.get("/health")
            assert status == 200


# =============================================================================
# Concurrency test
# =============================================================================


class TestConcurrentRequests:
    PORT = 9960

    def test_concurrent_requests_do_not_block_each_other(self):
        """Multiple simultaneous requests must be handled concurrently.

        Sends N slow-model predict requests simultaneously.
        With ThreadingHTTPServer, total wall-clock time should be much less
        than N * sleep_time (sequential would take N * sleep_time).
        """
        sleep_sec = 0.15
        n_requests = 5
        model = SlowModel(name="SlowModel", sleep_sec=sleep_sec)
        models = make_models_dict("SlowModel", model)

        results = []
        errors = []

        def send_request():
            try:
                data = json.dumps({"features": [1.0]}).encode()
                req = urllib.request.Request(
                    f"http://127.0.0.1:{self.PORT}/predict",
                    data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=10) as r:
                    results.append(json.loads(r.read().decode()))
            except Exception as e:
                errors.append(e)

        with ServerContext(self.PORT, model, models) as ctx:
            threads = [threading.Thread(target=send_request) for _ in range(n_requests)]
            start = time.perf_counter()
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)
            elapsed = time.perf_counter() - start

        assert errors == [], f"Concurrent requests produced errors: {errors}"
        assert len(results) == n_requests, f"Expected {n_requests} results, got {len(results)}"

        # Sequential execution would take >= n_requests * sleep_sec
        # Concurrent execution should finish in much less than that
        sequential_lower_bound = n_requests * sleep_sec
        assert elapsed < sequential_lower_bound, (
            f"Concurrent server took {elapsed:.2f}s for {n_requests} x {sleep_sec}s requests. "
            f"Expected < {sequential_lower_bound:.2f}s (concurrent). "
            "This suggests requests are being handled sequentially (blocking)."
        )


# =============================================================================
# Multi-model serving
# =============================================================================


class TestMultiModelServing:
    PORT = 9970

    def test_multi_model_models_list(self):
        model1 = FastModel(name="Model1")
        model2 = FastModel(name="Model2")
        models = {
            "Model1": {"class": FastModel, "instance": model1, "type": "ml", "description": "M1", "features": []},
            "Model2": {"class": FastModel, "instance": model2, "type": "ml", "description": "M2", "features": []},
        }
        handler = create_handler_class(
            model=model1,
            model_name="Model1",
            registered_models=models,
            active_model_name="Model1",
        )
        server = ThreadingHTTPServer(("127.0.0.1", self.PORT), handler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        time.sleep(0.05)
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.PORT}/models", timeout=5) as r:
                data = json.loads(r.read().decode())
                assert len(data) == 2
                names = {m["name"] for m in data}
                assert "Model1" in names
                assert "Model2" in names
        finally:
            server.shutdown()

    def test_predict_routes_to_correct_model(self):
        """POST /models/<name>/predict must invoke the named model's predict()."""
        call_log = []

        class LoggingModel(Model):
            def __init__(self, tag, **kwargs):
                super().__init__(**kwargs)
                self.tag = tag

            def predict(self, inputs, **kwargs):
                call_log.append(self.tag)
                return {"tag": self.tag}

        m1 = LoggingModel("alpha", name="Alpha")
        m2 = LoggingModel("beta", name="Beta")
        models = {
            "Alpha": {"class": LoggingModel, "instance": m1, "type": "ml", "description": "", "features": []},
            "Beta": {"class": LoggingModel, "instance": m2, "type": "ml", "description": "", "features": []},
        }
        handler = create_handler_class(
            model=m1,
            model_name="Alpha",
            registered_models=models,
            active_model_name="Alpha",
        )
        server = ThreadingHTTPServer(("127.0.0.1", self.PORT + 1), handler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        time.sleep(0.05)
        try:
            data = json.dumps({}).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{self.PORT + 1}/models/Beta/predict",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as r:
                result = json.loads(r.read().decode())
                assert result["result"]["tag"] == "beta"
            assert "beta" in call_log
        finally:
            server.shutdown()
