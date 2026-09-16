"""Comprehensive test suite for ModelKit CLI commands and HTTP server."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.client import HTTPConnection
from pathlib import Path
from typing import Any, Dict

from cli.commands import (
    run_data_validate,
    run_doctor,
    run_evaluate,
    run_init,
    run_serve,
    run_train,
)
from cli.discovery import find_project_root, load_project_manifest
from cli.main import main as cli_main
from cli.server import HTTPServer, create_handler_class
from modelkit.lifecycle import BaseInference
from modelkit.models import Model


class DummyModel(Model):
    """Simple test model."""

    def predict(self, inputs, **kwargs):
        if isinstance(inputs, list):
            return [x * 2 for x in inputs]
        return inputs


class CustomInference(BaseInference):
    """Developer-customized inference with custom routes."""

    def run(self, model: Model, raw_input: Any, **kwargs: Any) -> Any:
        return {"custom_prediction": model.predict(raw_input)}

    def custom_status(self, model: Model, payload: Any = None) -> Dict[str, Any]:
        return {"app_status": "all_systems_go", "active_model": model.name}

    def get_routes(self) -> Dict[str, Any]:
        return {
            "POST /predict": self.run,
            "GET /status": self.custom_status,
            "GET /health": self.health,
        }


class TestModelKitCLI(unittest.TestCase):
    """Tests for zero-path CLI execution, custom frontend, and custom routes."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="modelkit_cli_test_")
        self.test_root = Path(self.temp_dir)
        self.orig_cwd = os.getcwd()

    def tearDown(self):
        os.chdir(self.orig_cwd)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cli_help(self):
        """CLI main returns 0 on --help or no args."""
        self.assertEqual(cli_main([]), 0)

    def test_cli_doctor(self):
        """CLI doctor returns 0 and inspects environment."""
        self.assertEqual(cli_main(["doctor"]), 0)
        self.assertEqual(run_doctor(project_root=self.test_root), 0)

    def test_init_scaffolds_project(self):
        """CLI init creates convention folders, starter code, and mlkit.json without task_type."""
        proj_name = "test_ai"
        os.chdir(self.test_root)

        code = run_init(project_name=proj_name)
        self.assertEqual(code, 0)

        proj_dir = self.test_root / proj_name
        self.assertTrue((proj_dir / "mlkit.json").is_file())

        manifest = load_project_manifest(proj_dir)
        self.assertEqual(manifest["name"], proj_name)
        self.assertEqual(manifest["entrypoint"], proj_name)
        self.assertNotIn("type", manifest)
        self.assertNotIn("apps", manifest)

        # Convention directories must exist
        for d in ["data", "models", "experiments", "artifacts", "checkpoints"]:
            self.assertTrue((proj_dir / d).is_dir(), f"Missing convention directory: {d}")

        # Starter files must exist
        package_dir = proj_dir / proj_name
        for fname in ["config.py", "data.py", "model.py", "trainer.py", "evaluator.py", "inference.py"]:
            self.assertTrue((package_dir / fname).is_file(), f"Missing starter file: {fname}")

    def test_init_dot_uses_current_directory_name(self):
        """CLI init . initializes project directly inside current folder with folder name as package."""
        target_folder = self.test_root / "my_custom_workspace"
        target_folder.mkdir(parents=True, exist_ok=True)
        os.chdir(target_folder)

        code = run_init(project_name=".")
        self.assertEqual(code, 0)

        # Manifest must be directly in target_folder
        self.assertTrue((target_folder / "mlkit.json").is_file())
        manifest = load_project_manifest(target_folder)
        self.assertEqual(manifest["name"], "my_custom_workspace")
        self.assertEqual(manifest["entrypoint"], "my_custom_workspace")

        # Package directory must exist inside target_folder
        pkg_dir = target_folder / "my_custom_workspace"
        self.assertTrue(pkg_dir.is_dir())
        for fname in ["config.py", "data.py", "model.py", "trainer.py", "evaluator.py", "inference.py"]:
            self.assertTrue((pkg_dir / fname).is_file())

    def test_train_excludes_framework_classes(self):
        """CLI train strictly excludes framework classes (AdapterModel, RAGModel) from discovered models."""
        proj_name = "leak_test_proj"
        os.chdir(self.test_root)
        run_init(project_name=proj_name)
        proj_dir = self.test_root / proj_name
        os.chdir(proj_dir)

        # Put valid data
        data_file = proj_dir / "data" / "dataset.csv"
        with open(data_file, "w", encoding="utf-8") as f:
            f.write("f1,f2,target\n1,2,0\n3,4,1\n")

        # Empty out model.py so no custom model is defined
        model_file = proj_dir / proj_name / "model.py"
        with open(model_file, "w", encoding="utf-8") as f:
            f.write("# No model defined here\n")

        # Import RAGModel and AdapterModel into Python runtime to simulate them being loaded
        from modelkit import AdapterModel, RAGModel  # noqa: F401

        # Run train: it must fail with exit code 1 because no project model is found,
        # and NOT fail with 'Multiple models found in model.py: AdapterModel, RAGModel'
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            code = run_train(project_root=proj_dir)

        output = buf.getvalue()
        self.assertEqual(code, 1)
        self.assertIn("No Model subclass found in model.py", output)
        self.assertNotIn("AdapterModel", output)
        self.assertNotIn("RAGModel", output)

    def test_data_validate_fails_when_empty(self):
        """CLI data validate fails with exit code 1 when data directory is empty."""
        proj_name = "empty_data_proj"
        os.chdir(self.test_root)
        run_init(project_name=proj_name)
        proj_dir = self.test_root / proj_name
        os.chdir(proj_dir)

        code = run_data_validate(project_root=proj_dir)
        self.assertEqual(code, 1)

    def test_train_fails_when_no_data(self):
        """CLI train halts cleanly with exit code 1 if data is missing."""
        proj_name = "no_data_train_proj"
        os.chdir(self.test_root)
        run_init(project_name=proj_name)
        proj_dir = self.test_root / proj_name
        os.chdir(proj_dir)

        code = run_train(project_root=proj_dir)
        self.assertEqual(code, 1)

    def test_full_train_and_evaluate_cycle(self):
        """Full zero-path pipeline: data validate, train, evaluate."""
        proj_name = "lifecycle_proj"
        os.chdir(self.test_root)
        run_init(project_name=proj_name)
        proj_dir = self.test_root / proj_name
        os.chdir(proj_dir)

        # Add mock dataset to data/
        data_file = proj_dir / "data" / "dataset.csv"
        with open(data_file, "w", encoding="utf-8") as f:
            f.write("feature1,feature2,label\n")
            for i in range(20):
                f.write(f"{i * 0.1},{i * 0.2},{i % 2}\n")

        # 1. Validate data
        val_code = run_data_validate(project_root=proj_dir)
        self.assertEqual(val_code, 0)

        # 2. Run train
        train_code = run_train(project_root=proj_dir)
        self.assertEqual(train_code, 0)

        # Assert checkpoint and weights exist
        self.assertTrue((proj_dir / "models" / "model.pkl").is_file())
        self.assertTrue((proj_dir / "experiments" / "experiment_snapshot.json").is_file())

        # 3. Run evaluate
        eval_code = run_evaluate(project_root=proj_dir)
        self.assertEqual(eval_code, 0)

    def test_custom_developer_routes(self):
        """Tests that developer-defined custom routes and endpoints in BaseInference work."""
        model = DummyModel(name="custom_model")
        inference = CustomInference()
        handler_cls = create_handler_class(model=model, inference=inference, model_name="custom_model")
        server = HTTPServer(("127.0.0.1", 0), handler_cls)
        port = server.server_address[1]

        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        time.sleep(0.1)

        try:
            conn = HTTPConnection("127.0.0.1", port, timeout=5)

            # 1. GET /status (developer custom route)
            conn.request("GET", "/status")
            res = conn.getresponse()
            self.assertEqual(res.status, 200)
            data = json.loads(res.read().decode())
            self.assertEqual(data["app_status"], "all_systems_go")
            self.assertEqual(data["active_model"], "custom_model")

            # 2. POST /predict (custom inference handler)
            payload = json.dumps([1, 2, 3])
            conn.request("POST", "/predict", body=payload, headers={"Content-Type": "application/json"})
            res = conn.getresponse()
            self.assertEqual(res.status, 200)
            pred_data = json.loads(res.read().decode())
            self.assertEqual(pred_data["custom_prediction"], [2, 4, 6])
        finally:
            server.shutdown()
            server.server_close()

    def test_custom_frontend_serving(self):
        """Tests that modelkit serve can host a developer's custom frontend."""
        # Create mock frontend directory with index.html and style.css
        frontend_dir = self.test_root / "custom_frontend"
        frontend_dir.mkdir(parents=True, exist_ok=True)
        (frontend_dir / "index.html").write_text("<!DOCTYPE html><html><body><h1>Developer Frontend</h1></body></html>", encoding="utf-8")
        (frontend_dir / "app.js").write_text("console.log('custom');", encoding="utf-8")

        model = DummyModel(name="test_model")
        handler_cls = create_handler_class(model=model, model_name="test_model", frontend_dir=frontend_dir)
        server = HTTPServer(("127.0.0.1", 0), handler_cls)
        port = server.server_address[1]

        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        time.sleep(0.1)

        try:
            conn = HTTPConnection("127.0.0.1", port, timeout=5)

            # 1. GET / -> serves developer's custom index.html
            conn.request("GET", "/")
            res = conn.getresponse()
            self.assertEqual(res.status, 200)
            self.assertIn("text/html", res.headers.get("Content-Type", ""))
            body = res.read().decode()
            self.assertIn("Developer Frontend", body)

            # 2. GET /app.js -> serves static asset
            conn.request("GET", "/app.js")
            res = conn.getresponse()
            self.assertEqual(res.status, 200)
            self.assertIn("custom", res.read().decode())

            # 3. GET /dashboard -> SPA route fallback to index.html
            conn.request("GET", "/dashboard")
            res = conn.getresponse()
            self.assertEqual(res.status, 200)
            self.assertIn("Developer Frontend", res.read().decode())

            # 4. GET /health -> API still accessible alongside custom frontend
            conn.request("GET", "/health")
            res = conn.getresponse()
            self.assertEqual(res.status, 200)
        finally:
            server.shutdown()
            server.server_close()

    def test_serve_fails_when_no_checkpoint_or_data(self):
        """CLI serve fails cleanly with exit code 1 if model is not trained and data is missing."""
        proj_name = "uninitialized_serve_proj"
        os.chdir(self.test_root)
        run_init(project_name=proj_name)
        proj_dir = self.test_root / proj_name

        code = run_serve(project_root=proj_dir)
        self.assertEqual(code, 1)

    def test_default_api_root_returns_json_not_html(self):
        """When no custom frontend is provided, serve returns JSON API info, not a web page."""
        model = DummyModel(name="api_model")
        handler_cls = create_handler_class(model=model, model_name="api_model")
        server = HTTPServer(("127.0.0.1", 0), handler_cls)
        port = server.server_address[1]

        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        time.sleep(0.1)

        try:
            conn = HTTPConnection("127.0.0.1", port, timeout=5)

            # GET / -> pure JSON, not HTML
            conn.request("GET", "/")
            res = conn.getresponse()
            self.assertEqual(res.status, 200)
            self.assertIn("application/json", res.headers.get("Content-Type", ""))
            data = json.loads(res.read().decode())
            self.assertEqual(data["name"], "api_model")
            self.assertEqual(data["status"], "online")
            self.assertIn("predict", data["endpoints"])

            # GET /predict -> JSON instructions, not HTML
            conn.request("GET", "/predict")
            res2 = conn.getresponse()
            self.assertEqual(res2.status, 200)
            self.assertIn("application/json", res2.headers.get("Content-Type", ""))
            info = json.loads(res2.read().decode())
            self.assertEqual(info["method"], "POST")
        finally:
            server.shutdown()
            server.server_close()

    def test_empty_csv_file_gracefully_fails_without_traceback(self):
        """CLI train and data validate handle 0-byte or empty CSV files gracefully without tracebacks."""
        proj_name = "empty_csv_proj"
        os.chdir(self.test_root)
        run_init(project_name=proj_name)
        proj_dir = self.test_root / proj_name
        os.chdir(proj_dir)

        # Create an empty 0-byte CSV file in data/
        empty_csv = proj_dir / "data" / "dataset.csv"
        empty_csv.touch()

        # Both commands must return exit code 1 cleanly without raising unhandled EmptyDataError
        val_code = run_data_validate(project_root=proj_dir)
        self.assertEqual(val_code, 1)

        train_code = run_train(project_root=proj_dir)
        self.assertEqual(train_code, 1)

    def test_multiple_datasets_and_file_types(self):
        """Tests projects with multiple data files (JSON, TSV, CSV) and multiple Dataset classes."""
        proj_name = "multi_data_proj"
        os.chdir(self.test_root)
        run_init(project_name=proj_name)
        proj_dir = self.test_root / proj_name
        os.chdir(proj_dir)

        # 1. Add users.csv
        users_file = proj_dir / "data" / "users.csv"
        users_file.write_text("id,name,score\n1,Alice,95\n2,Bob,88\n", encoding="utf-8")

        # 2. Add metrics.json
        metrics_file = proj_dir / "data" / "metrics.json"
        metrics_file.write_text('[{"metric": "loss", "val": 0.1}, {"metric": "acc", "val": 0.99}]', encoding="utf-8")

        # Update data.py to define multiple dataset classes
        data_py = proj_dir / proj_name / "data.py"
        data_py.write_text('''from modelkit import Dataset

class UsersDataset(Dataset):
    filename = "users.csv"

class MetricsDataset(Dataset):
    filename = "metrics.json"
''', encoding="utf-8")

        # Update model.py to bind UsersDataset
        model_py = proj_dir / proj_name / "model.py"
        model_py.write_text(f'''from typing import Any
from modelkit import Model
from {proj_name}.data import UsersDataset

class MultiModel(Model):
    dataset = UsersDataset

    def predict(self, inputs: Any, **kwargs: Any) -> Any:
        return inputs
''', encoding="utf-8")

        # Validate datasets
        val_code = run_data_validate(project_root=proj_dir)
        self.assertEqual(val_code, 0)

        # Train pipeline
        train_code = run_train(project_root=proj_dir)
        self.assertEqual(train_code, 0)

        # Verify weights were written
        self.assertTrue((proj_dir / "models" / "model.pkl").is_file())

    def test_train_fails_when_custom_dataset_has_no_matching_model(self):
        """If developer created a custom dataset in data.py but no model in model.py, train fails cleanly."""
        proj_name = "no_model_proj"
        os.chdir(self.test_root)
        run_init(project_name=proj_name)
        proj_dir = self.test_root / proj_name
        os.chdir(proj_dir)

        # 1. Add valid CSV data
        (proj_dir / "data" / "regression.csv").write_text("x,y\n1,2\n2,4\n", encoding="utf-8")

        # 2. Define custom dataset in data.py
        data_py = proj_dir / proj_name / "data.py"
        data_py.write_text('''from modelkit import Dataset

class UiRegressionDataset(Dataset):
    filename = "regression.csv"
''', encoding="utf-8")

        # Note: model.py only contains default AppModel without binding UiRegressionDataset
        # Running train without target must fail and suggest defining UiRegressionModel
        code = run_train(project_root=proj_dir)
        self.assertEqual(code, 1)

        # Trying to train non-existent model class by name must also fail cleanly
        non_existent_code = run_train(target="UiRegressionModel", project_root=proj_dir)
        self.assertEqual(non_existent_code, 1)

    def test_targeted_train_by_class_name(self):
        """Tests training a specific model class by passing class name to modelkit train."""
        proj_name = "targeted_train_proj"
        os.chdir(self.test_root)
        run_init(project_name=proj_name)
        proj_dir = self.test_root / proj_name
        os.chdir(proj_dir)

        # 1. Add dataset
        (proj_dir / "data" / "users.csv").write_text("id,val\n1,10\n2,20\n", encoding="utf-8")

        # 2. Define dataset
        data_py = proj_dir / proj_name / "data.py"
        data_py.write_text('''from modelkit import Dataset

class UserDataset(Dataset):
    filename = "users.csv"
''', encoding="utf-8")

        # 3. Define specific model
        model_py = proj_dir / proj_name / "model.py"
        model_py.write_text(f'''from typing import Any
from modelkit import Model
from {proj_name}.data import UserDataset

class UserModel(Model):
    dataset = UserDataset

    def predict(self, inputs: Any, **kwargs: Any) -> Any:
        return inputs
''', encoding="utf-8")

        # Train specifically UserModel
        code = run_train(target="UserModel", project_root=proj_dir)
        self.assertEqual(code, 0)
        self.assertTrue((proj_dir / "models" / "model.pkl").is_file())

    def test_train_fails_when_multiple_models_without_target(self):
        """When multiple custom models exist in model.py, modelkit train without args halts and asks for class name."""
        proj_name = "multi_model_proj"
        os.chdir(self.test_root)
        run_init(project_name=proj_name)
        proj_dir = self.test_root / proj_name
        os.chdir(proj_dir)

        (proj_dir / "data" / "dataset.csv").write_text("a,b\n1,2\n", encoding="utf-8")

        model_py = proj_dir / proj_name / "model.py"
        model_py.write_text('''from typing import Any
from modelkit import Model

class ModelA(Model):
    def predict(self, inputs: Any, **kwargs: Any) -> Any:
        return inputs

class ModelB(Model):
    def predict(self, inputs: Any, **kwargs: Any) -> Any:
        return inputs
''', encoding="utf-8")

        # Running train without target must fail and ask which model to train
        code = run_train(project_root=proj_dir)
        self.assertEqual(code, 1)

        # Running train ModelA specifically succeeds
        code_a = run_train(target="ModelA", project_root=proj_dir)
        self.assertEqual(code_a, 0)

    def test_standalone_executable(self):
        """Tests that build/modelkit runs and executes CLI commands."""
        project_root = Path(__file__).resolve().parent.parent
        executable = project_root / "build" / "modelkit"
        self.assertTrue(executable.is_file(), "Executable build/modelkit does not exist")
        self.assertTrue(os.access(executable, os.X_OK), "build/modelkit is not executable")

        # Test --version
        res = subprocess.run([str(executable), "--version"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("modelkit 0.1.0", res.stdout)

        # Test doctor
        res = subprocess.run([str(executable), "doctor"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("ModelKit Doctor", res.stdout)


if __name__ == "__main__":
    unittest.main()
