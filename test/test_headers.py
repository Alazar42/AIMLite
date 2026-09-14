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

    def test_dataset_interface(self):
        raw_ds = Dataset("raw_dataset")
        self.assertEqual(raw_ds.name, "raw_dataset")
        self.assertFalse(raw_ds.validate())

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

    def test_csv_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            csv_path.write_text("col1,col2,target\n1,2,0\n3,4,1\n5,6,0\n7,8,1\n", encoding="utf-8")

            ds = modelkit.Dataset(name="my_csv", source=csv_path)
            self.assertTrue(ds.validate())
            data = ds.load()
            self.assertEqual(len(data), 4)
            self.assertEqual(ds.columns, ["col1", "col2", "target"])

            train, val, test = ds.split(0.5, 0.25, 0.25)
            self.assertGreater(len(train), 0)

    def test_csv_multiple_files_handling(self):
        """Tests handling of multiple CSV files: specific filename, combine_all, and pre-split."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()

            # Create multiple CSVs
            (data_dir / "housing.csv").write_text("price,rooms\n100,2\n200,3\n", encoding="utf-8")
            (data_dir / "customers.csv").write_text("age,income\n25,50\n40,90\n", encoding="utf-8")

            # 1. Target specific file via class attribute
            class HousingDataset(modelkit.Dataset):
                filename = "housing.csv"

            ds_housing = HousingDataset("housing", config=modelkit.BaseConfig(tmpdir))
            data_h = ds_housing.load()
            self.assertEqual(len(data_h), 2)
            self.assertEqual(ds_housing.columns, ["price", "rooms"])

            # 2. Combine all CSV files
            (data_dir / "shard1.csv").write_text("feat,val\n1,10\n", encoding="utf-8")
            (data_dir / "shard2.csv").write_text("feat,val\n2,20\n", encoding="utf-8")

            class ShardedDataset(modelkit.Dataset):
                combine_all = True

            with tempfile.TemporaryDirectory() as shard_dir:
                s_data = Path(shard_dir) / "data"
                s_data.mkdir()
                (s_data / "a.csv").write_text("v\n1\n2\n", encoding="utf-8")
                (s_data / "b.csv").write_text("v\n3\n4\n", encoding="utf-8")
                ds_shard = ShardedDataset("shards", config=modelkit.BaseConfig(shard_dir))
                data_s = ds_shard.load()
                self.assertEqual(len(data_s), 4)

            # 3. Pre-split convention: train.csv and test.csv
            with tempfile.TemporaryDirectory() as presplit_dir:
                ps_data = Path(presplit_dir) / "data"
                ps_data.mkdir()
                (ps_data / "train.csv").write_text("x,y\n1,1\n2,2\n3,3\n4,4\n", encoding="utf-8")
                (ps_data / "test.csv").write_text("x,y\n5,5\n6,6\n", encoding="utf-8")

                ds_presplit = modelkit.Dataset("presplit", config=modelkit.BaseConfig(presplit_dir))
                ds_presplit.load()
                train_p, val_p, test_p = ds_presplit.split(validation=0.25)
                self.assertEqual(len(test_p), 2)  # from test.csv
                self.assertGreater(len(train_p), 0)

    def test_default_model_persistence(self):
        class DefaultModel(Model):
            def predict(self, inputs, **kwargs):
                return inputs

        m = DefaultModel("default_m")
        m.weights = {"alpha": 0.5}

        with tempfile.TemporaryDirectory() as tmpdir:
            m.save(tmpdir)
            self.assertTrue((Path(tmpdir) / "model.pkl").is_file())

            m2 = DefaultModel("default_m")
            m2.load(tmpdir)
            self.assertEqual(m2.weights["alpha"], 0.5)

    def test_model_dataset_binding(self):
        class CustomerDataset(Dataset):
            source = "customers.csv"

        class ChurnModel(Model):
            dataset = CustomerDataset

            def predict(self, inputs, **kwargs):
                return [1]

        m = ChurnModel("churn")
        self.assertEqual(m.dataset, CustomerDataset)

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

    def test_automatic_subclass_registration_under_the_hood(self):
        """Validates that subclasses are automatically registered into the registry without @register."""
        from modelkit import get_all

        class AutoRegisteredDataset(Dataset):
            pass

        class AutoRegisteredModel(Model):
            dataset = "AutoRegisteredDataset"

            def predict(self, inputs, **kwargs):
                return inputs

        class AutoRegisteredTrainer(BaseTrainer):
            def fit(self, model, dataset, **kwargs):
                return {"loss": 0.0}

        class AutoRegisteredEvaluator(BaseEvaluator):
            def evaluate(self, model, dataset, **kwargs):
                return {"accuracy": 1.0}

        class AutoRegisteredInference(BaseInference):
            def run(self, model, raw_input, **kwargs):
                return {"res": 1}

        # 1. Verify automatic registration without @register
        self.assertIs(get("dataset", "AutoRegisteredDataset"), AutoRegisteredDataset)
        self.assertIs(get("model", "AutoRegisteredModel"), AutoRegisteredModel)
        self.assertIs(get("trainer", "AutoRegisteredTrainer"), AutoRegisteredTrainer)
        self.assertIs(get("evaluator", "AutoRegisteredEvaluator"), AutoRegisteredEvaluator)
        self.assertIs(get("inference", "AutoRegisteredInference"), AutoRegisteredInference)

        # 2. Verify case-insensitive retrieval
        self.assertIs(get("model", "autoregisteredmodel"), AutoRegisteredModel)
        self.assertIs(get("dataset", "autoregistereddataset"), AutoRegisteredDataset)

        # 3. Verify get_all
        all_models = get_all("model")
        self.assertIn("AutoRegisteredModel", all_models)

        # 4. Verify Model.get_dataset resolves dataset string via registry
        m = AutoRegisteredModel("auto_m")
        ds = m.get_dataset()
        self.assertIsInstance(ds, AutoRegisteredDataset)


if __name__ == "__main__":
    unittest.main()

