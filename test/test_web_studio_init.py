"""Tests for locally hosted AIMLite web studio and CLI --cli flag routing."""

import json
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from aimlite.cli.main import build_parser, main
from aimlite.cli.studio import (
    _find_available_port,
    _resolve_static_dist_dir,
    create_studio_handler,
    ThreadingStudioServer,
)


def test_parser_init_cli_flag():
    """Verify --cli flag is registered on the init subcommand."""
    parser = build_parser()
    
    # Without --cli
    args1 = parser.parse_args(["init", "my_proj"])
    assert args1.command == "init"
    assert args1.name == "my_proj"
    assert args1.cli is False

    # With --cli
    args2 = parser.parse_args(["init", "my_proj", "--cli"])
    assert args2.command == "init"
    assert args2.name == "my_proj"
    assert args2.cli is True

    # With --cli and paradigm template
    args3 = parser.parse_args(["init", "my_rag", "--cli", "--type", "rag"])
    assert args3.command == "init"
    assert args3.name == "my_rag"
    assert args3.cli is True
    assert args3.template_type == "rag"


def test_parser_studio_top_level_flags():
    """Verify top-level studio flags (--port, --host, --no-browser)."""
    parser = build_parser()
    args = parser.parse_args(["--port", "8888", "--host", "0.0.0.0", "--no-browser"])
    assert args.port == 8888
    assert args.host == "0.0.0.0"
    assert args.no_browser is True


def test_main_bare_aimlite_routes_to_studio():
    """Running bare 'aimlite' must launch the locally hosted web studio."""
    with patch("aimlite.cli.main.run_studio", return_value=0) as mock_studio:
        code = main([])
        assert code == 0
        mock_studio.assert_called_once_with(host="127.0.0.1", port=8000, open_browser=True)


def test_main_bare_aimlite_with_flags_routes_to_studio():
    """Running 'aimlite --port 9000 --no-browser' routes to studio with options."""
    with patch("aimlite.cli.main.run_studio", return_value=0) as mock_studio:
        code = main(["--port", "9000", "--no-browser"])
        assert code == 0
        mock_studio.assert_called_once_with(host="127.0.0.1", port=9000, open_browser=False)


def test_main_init_without_cli_routes_to_studio():
    """Running 'aimlite init' without --cli must launch the web studio."""
    with patch("aimlite.cli.main.run_studio", return_value=0) as mock_studio, \
         patch("aimlite.cli.main.run_init") as mock_init:
        code = main(["init", "my_studio_proj", "--type", "rag"])
        assert code == 0
        mock_init.assert_not_called()
        mock_studio.assert_called_once()
        _, kwargs = mock_studio.call_args
        assert kwargs["initial_params"]["name"] == "my_studio_proj"
        assert kwargs["initial_params"]["template_type"] == "rag"


def test_main_init_with_cli_routes_to_cli_init(tmp_path: Path):
    """Running 'aimlite init <name> --cli' must execute creation steps from the CLI."""
    with patch("aimlite.cli.main.run_studio") as mock_studio, \
         patch("aimlite.cli.main.run_init", return_value=0) as mock_init:
        code = main(["init", "my_cli_proj", "--cli", "--clean", "--type", "scratch"])
        assert code == 0
        mock_studio.assert_not_called()
        mock_init.assert_called_once()
        _, kwargs = mock_init.call_args
        assert kwargs["project_name"] == "my_cli_proj"
        assert kwargs["clean"] is True
        assert kwargs["template_type"] == "scratch"


@pytest.fixture
def test_studio_server(tmp_path: Path):
    """Fixture that boots up an ephemeral ThreadingStudioServer on an open port."""
    port = _find_available_port(start_port=18500)
    dist_dir = _resolve_static_dist_dir()
    handler_class = create_studio_handler(dist_dir)
    server = ThreadingStudioServer(("127.0.0.1", port), handler_class)

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    time.sleep(0.1)

    base_url = f"http://127.0.0.1:{port}"
    yield base_url, tmp_path

    server.shutdown()
    server.server_close()


def test_studio_api_status(test_studio_server):
    """Verify GET /api/status returns 200 with runtime diagnostics."""
    base_url, _ = test_studio_server
    req = urllib.request.Request(f"{base_url}/api/status")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["status"] == "ok"
        assert data["version"] == "2.2.1"
        assert "cwd" in data
        assert "python_version" in data


def test_studio_get_index_html(test_studio_server):
    """Verify GET / serves index.html."""
    base_url, _ = test_studio_server
    req = urllib.request.Request(f"{base_url}/")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        content = resp.read().decode("utf-8")
        assert "<html" in content.lower()
        assert "AIMLite" in content


def test_studio_api_init_scratch(test_studio_server):
    """Verify POST /api/init successfully scaffolds a scratch project."""
    base_url, tmp_dir = test_studio_server

    payload = {
        "project_name": "web_scratch_test",
        "target_dir": str(tmp_dir),
        "template_type": "scratch",
        "clean": True,
        "create_venv": False,
        "install_deps": False,
    }

    req = urllib.request.Request(
        f"{base_url}/api/init",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["status"] == "success"
        assert data["project_name"] == "web_scratch_test"
        assert "aimlite train" in " ".join(data["commands"])

    # Verify project exists on disk
    target = tmp_dir / "web_scratch_test"
    assert target.is_dir()
    manifest = json.loads((target / "aimlite.json").read_text(encoding="utf-8"))
    assert manifest["name"] == "web_scratch_test"
    assert manifest["template"] == "scratch"
    assert manifest["clean"] is True


def test_studio_api_init_rag(test_studio_server):
    """Verify POST /api/init successfully scaffolds a RAG project with configs."""
    base_url, tmp_dir = test_studio_server

    payload = {
        "project_name": "web_rag_test",
        "target_dir": str(tmp_dir),
        "template_type": "rag",
        "chat_provider": "gemini",
        "model_name": "gemini-1.5-flash",
        "vector_db": "memory",
        "embedding_engine": "sentence-transformers",
        "clean": True,
        "create_venv": False,
        "install_deps": False,
    }

    req = urllib.request.Request(
        f"{base_url}/api/init",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["status"] == "success"
        assert data["template"] == "rag"

    target = tmp_dir / "web_rag_test"
    assert target.is_dir()
    manifest = json.loads((target / "aimlite.json").read_text(encoding="utf-8"))
    assert manifest["template"] == "rag"
    assert manifest["config"]["rag"]["chat_provider"] == "gemini"
    assert manifest["config"]["rag"]["model_name"] == "gemini-1.5-flash"
