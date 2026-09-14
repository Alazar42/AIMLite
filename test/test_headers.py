"""Tests for ModelKit Core Headers and Contract Specifications."""

import tempfile
import unittest
from pathlib import Path

import modelkit
from modelkit.config import BaseConfig
from modelkit.data import Dataset
from modelkit.lifecycle import BaseEvaluator, BaseInference, BaseTrainer
from modelkit.models import Model
from modelkit.registry import clear_registry, get, register


class DummyDataset(Dataset):
    def load(self, source=None, **kwargs):
        self._data = [1, 2, 3, 4, 5]
        return self

    def split(self, train=0.8, validation=0.1, test=0.1, **kwargs):
        return ([1, 2, 3, 4], [5], [])


class DummyModel(Model):
    def __init__(self, name="dummy_model", config=None):
        super().__init__(name, config)
        self.weights = {"param": 1.0}

    def predict(self, inputs, **kwargs):
        return [x * 2 for x in inputs]

    def save(self, destination, **kwargs):
        dest = Path(destination)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "weights.txt").write_text("1.0")

    def load(self, source, **kwargs):
        self.weights = {"param": 1.0}


class DummyTrainer(BaseTrainer):
    def fit(self, model, dataset, **kwargs):
        return {"epochs": 1, "loss": 0.05}


class DummyEvaluator(BaseEvaluator):
    def evaluate(self, model, dataset, **kwargs):
        return {"accuracy": 0.98, "loss": 0.05}


class DummyInference(BaseInference):
    def run(self, model, raw_input, **kwargs):
        return {"prediction": model.predict(raw_input)}


class TestModelKitHeaders(unittest.TestCase):
    def setUp(self):
        clear_registry()

    def test_top_level_exports(self):
        self.assertTrue(hasattr(modelkit, "BaseConfig"))
        self.assertTrue(hasattr(modelkit, "Dataset"))
        self.assertTrue(hasattr(modelkit, "Model"))
        self.assertTrue(hasattr(modelkit, "BaseTrainer"))
        self.assertTrue(hasattr(modelkit, "BaseEvaluator"))
        self.assertTrue(hasattr(modelkit, "BaseInference"))
        self.assertTrue(hasattr(modelkit, "register"))
        self.assertTrue(hasattr(modelkit, "get"))

    def test_base_config(self):
        cfg = BaseConfig()
        device = cfg.resolve_device()
        self.assertIn(device, ["cuda", "mps", "cpu"])
        data_dict = cfg.to_dict()
        self.assertIsInstance(data_dict, dict)
        self.assertIn("name", data_dict)

    def test_dataset_abstract(self):
        with self.assertRaises(TypeError):
            Dataset("raw_dataset")

        ds = DummyDataset("test_data", source="dummy.csv")
        self.assertEqual(ds.name, "test_data")
        self.assertEqual(ds.source, Path("dummy.csv"))
        self.assertTrue(ds.validate())
        loaded = ds.load()
        self.assertIsNotNone(loaded._data)
        train_p, val_p, test_p = ds.split(0.8, 0.1, 0.1)
        self.assertEqual(train_p, [1, 2, 3, 4])

    def test_model_abstract(self):
        with self.assertRaises(TypeError):
            Model("raw_model")

        m = DummyModel("test_model", {"lr": 0.001})
        self.assertEqual(m.name, "test_model")
        self.assertEqual(m.config, {"lr": 0.001})
        self.assertEqual(m.predict([1, 2, 3]), [2, 4, 6])
        self.assertIsInstance(m.evaluate([1, 2]), dict)

    def test_lifecycle_trainer(self):
        with self.assertRaises(TypeError):
            BaseTrainer()

        trainer = DummyTrainer()
        m = DummyModel()
        ds = DummyDataset("test_data")
        metrics = trainer.fit(m, ds)
        self.assertEqual(metrics["loss"], 0.05)

        with tempfile.TemporaryDirectory() as tmpdir:
            chk_path = trainer.checkpoint(m, tmpdir, step=1)
            self.assertTrue(chk_path.exists())
            self.assertTrue((chk_path / "experiment_snapshot.json").exists())
            self.assertTrue((chk_path / "weights.txt").exists())

    def test_lifecycle_evaluator(self):
        with self.assertRaises(TypeError):
            BaseEvaluator()

        evaluator = DummyEvaluator()
        m = DummyModel()
        ds = DummyDataset("test_data")
        res = evaluator.evaluate(m, ds)
        self.assertIn("accuracy", res)
        self.assertEqual(res["accuracy"], 0.98)

    def test_lifecycle_inference(self):
        with self.assertRaises(TypeError):
            BaseInference()

        infer = DummyInference()
        m = DummyModel()
        out = infer.run(m, [5, 10])
        self.assertEqual(out, {"prediction": [10, 20]})

    def test_registry(self):
        @register("model", "dummy")
        class RegisteredModel(DummyModel):
            pass

        retrieved = get("model", "dummy")
        self.assertIs(retrieved, RegisteredModel)

        with self.assertRaises(KeyError):
            get("model", "unknown")

        with self.assertRaises(KeyError):
            get("unknown_category", "dummy")


if __name__ == "__main__":
    unittest.main()
