"""Tests for AIMLite multi-model serving, type detection, and headless --api mode."""

import json
import tempfile
import urllib.request
import threading
from http.server import HTTPServer
from pathlib import Path

from cli.commands.serve import detect_model_type, discover_feature_names
from cli.server import _load_template, create_handler_class, get_openapi_schema
from aimlite.models import Model
from aimlite.rag import KnowledgeModel
from aimlite.adapters import AdapterModel


class MockClassifier(Model):
    """Mock ML classifier with feature inputs."""
    def predict(self, inputs, **kwargs):
        return [1]


class MockRAG(KnowledgeModel):
    """Mock RAG conversational assistant."""
    def predict(self, inputs, **kwargs):
        return {"answer": "Grounded response", "sources": [{"content": "Source doc 1"}]}


class MockLoRA(AdapterModel):
    """Mock LoRA fine-tuned model."""
    def predict(self, inputs, **kwargs):
        return "Generated token response"


def test_detect_model_type():
    assert detect_model_type(MockClassifier) == "ml"
    assert detect_model_type(MockRAG) == "rag"
    assert detect_model_type(MockLoRA) == "adapter"


def test_template_loading_priority():
    with tempfile.TemporaryDirectory() as td:
        custom_dir = Path(td)
        (custom_dir / "app.html").write_text("<div>Developer Custom App</div>", encoding="utf-8")
        
        loaded = _load_template("app.html", template_dir=custom_dir)
        assert loaded == "<div>Developer Custom App</div>"


def test_multi_model_server_endpoints():
    clf = MockClassifier(name="MockClassifier")
    rag = MockRAG(name="MockRAG")
    
    models = {
        "MockClassifier": {
            "class": MockClassifier,
            "instance": clf,
            "type": "ml",
            "description": "Mock ML Classifier",
            "features": ["feature_a", "feature_b"],
        },
        "MockRAG": {
            "class": MockRAG,
            "instance": rag,
            "type": "rag",
            "description": "Mock Knowledge Engine",
            "features": [],
        }
    }
    
    handler = create_handler_class(
        model=clf,
        model_name="MockClassifier",
        registered_models=models,
        active_model_name="MockClassifier",
        api_only=False,
    )
    
    server = HTTPServer(("127.0.0.1", 9922), handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    
    try:
        # 1. GET /models
        with urllib.request.urlopen("http://127.0.0.1:9922/models") as r:
            assert r.status == 200
            m_list = json.loads(r.read().decode())
            assert len(m_list) == 2
            assert m_list[0]["name"] == "MockClassifier"
            assert m_list[0]["type"] == "ml"
            assert m_list[1]["name"] == "MockRAG"
            assert m_list[1]["type"] == "rag"

        # 2. POST /models/MockClassifier/predict
        req_ml = urllib.request.Request(
            "http://127.0.0.1:9922/models/MockClassifier/predict",
            data=json.dumps({"features": [1.0, 2.0]}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_ml) as r:
            assert r.status == 200
            res = json.loads(r.read().decode())
            assert res["status"] == "success"
            assert res["result"] == [1]

        # 3. POST /models/MockRAG/predict
        req_rag = urllib.request.Request(
            "http://127.0.0.1:9922/models/MockRAG/predict",
            data=json.dumps({"query": "What is AIMLite?"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_rag) as r:
            assert r.status == 200
            res = json.loads(r.read().decode())
            assert res["status"] == "success"
            assert res["result"]["answer"] == "Grounded response"

        # 4. GET / (Browser HTML)
        req_html = urllib.request.Request(
            "http://127.0.0.1:9922/",
            headers={"Accept": "text/html"},
        )
        with urllib.request.urlopen(req_html) as r:
            assert r.status == 200
            assert "text/html" in r.headers.get("Content-Type", "")

    finally:
        server.shutdown()


def test_headless_api_mode():
    clf = MockClassifier(name="MockClassifier")
    models = {
        "MockClassifier": {
            "class": MockClassifier,
            "instance": clf,
            "type": "ml",
            "description": "Mock ML Classifier",
            "features": [],
        }
    }
    
    handler = create_handler_class(
        model=clf,
        model_name="MockClassifier",
        registered_models=models,
        active_model_name="MockClassifier",
        api_only=True,
    )
    
    server = HTTPServer(("127.0.0.1", 9923), handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    
    try:
        # GET / returns JSON in api_only mode
        with urllib.request.urlopen("http://127.0.0.1:9923/") as r:
            assert r.status == 200
            data = json.loads(r.read().decode())
            assert data["mode"] == "headless_api"
            
        # GET /chat returns 403 in api_only mode
        try:
            urllib.request.urlopen("http://127.0.0.1:9923/chat")
            assert False, "Should have raised HTTPError"
        except urllib.error.HTTPError as e:
            assert e.code == 403
    finally:
        server.shutdown()
