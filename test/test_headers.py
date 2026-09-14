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

    def test_multi_app_config(self):
        """Validates multi-app discovery and resolution in BaseConfig."""
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "mlkit.json"
            manifest_data = {
                "name": "enterprise_ai",
                "apps": ["knowledge", "classifier"],
                "app_configs": {
                    "knowledge": {"type": "rag", "retriever_k": 5},
                    "classifier": {"type": "fine_tune", "learning_rate": 0.001},
                },
            }
            manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")

            cfg = BaseConfig(config_path=manifest_path)
            self.assertTrue(cfg.is_multi_app)
            self.assertEqual(cfg.apps, ["knowledge", "classifier"])
            self.assertEqual(cfg.get_app_dir("knowledge"), Path(tmpdir) / "knowledge")
            self.assertEqual(cfg.get_app_config("knowledge")["type"], "rag")
            self.assertEqual(cfg.get_app_config("classifier")["retriever_k" if "retriever_k" in cfg.get_app_config("classifier") else "type"], "fine_tune")

        # Flat root fallback
        default_cfg = BaseConfig()
        self.assertFalse(default_cfg.is_multi_app)
        self.assertEqual(default_cfg.apps, [])

    def test_rag_pipeline(self):
        """Validates first-class RAG components: Document, TextSplitter, Embedding, VectorStore, Retriever, RAGModel."""
        from modelkit.rag import (
            Document,
            DocumentLoader,
            MemoryVectorStore,
            RAGModel,
            TextSplitter,
            TfidfEmbedding,
            VectorRetriever,
        )

        # 1. Document & TextSplitter
        doc = Document(content="ModelKit is the Django for AI. It unifies RAG, fine-tuning, and training.", metadata={"author": "team"})
        splitter = TextSplitter(chunk_size=35, chunk_overlap=10)
        chunks = splitter.split_documents([doc])
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0].metadata["parent_id"], doc.id)

        # 2. Embedding & VectorStore
        embedder = TfidfEmbedding(dim=128)
        vec = embedder.embed_text("Django for AI")
        self.assertEqual(len(vec), 128)

        store = MemoryVectorStore(embedding_fn=embedder)
        docs = [
            Document(content="Refunds are processed within 14 business days.", metadata={"topic": "billing"}),
            Document(content="Remote work policy requires manager pre-approval.", metadata={"topic": "hr"}),
        ]
        store.add_documents(docs)

        # 3. VectorRetriever
        retriever = VectorRetriever(vector_store=store, embedding_fn=embedder)
        results = retriever.retrieve("How do I get a refund?", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertIn("Refunds", results[0].content)

        # 4. RAGModel
        class PolicyRAG(RAGModel):
            pass

        rag_model = PolicyRAG("policy_rag", retriever=retriever)
        out = rag_model.predict("How do I get a refund?", top_k=1)
        self.assertEqual(out["query"], "How do I get a refund?")
        self.assertIn("Refunds are processed", out["answer"])
        self.assertEqual(len(out["sources"]), 1)
        self.assertEqual(out["sources"][0]["metadata"]["topic"], "billing")

        # 5. Under-the-hood auto-registration under "model" and "rag"
        self.assertIs(get("rag", "PolicyRAG"), PolicyRAG)
        self.assertIs(get("model", "PolicyRAG"), PolicyRAG)

    def test_adapter_fine_tuning_pipeline(self):
        """Validates adapter fine-tuning: AdapterConfig, AdapterModel, delta persistence, AdapterTrainer."""
        from modelkit.adapters import (
            AdapterConfig,
            AdapterModel,
            AdapterTrainer,
        )

        # 1. AdapterConfig
        cfg = AdapterConfig(r=16, alpha=32.0, base_model_path="meta-llama/Llama-3-8B")
        cfg_dict = cfg.to_dict()
        self.assertEqual(cfg_dict["r"], 16)
        self.assertEqual(cfg_dict["alpha"], 32.0)

        # 2. AdapterModel
        class MockBaseLLM:
            def predict(self, x, **kwargs):
                return f"BaseLLM output: {x}"

        class SummarizationAdapter(AdapterModel):
            pass

        base_llm = MockBaseLLM()
        adapter_model = SummarizationAdapter("summary_lora", adapter_config=cfg, base_model=base_llm)
        pred = adapter_model.predict("Summarize document")
        self.assertEqual(pred, "BaseLLM output: Summarize document")

        # 3. Lightweight delta persistence (saving only adapter weights)
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_model.adapter_weights["lora_A"]["layer_0"] = [0.1, 0.2]
            adapter_model.save(tmpdir)

            dest_path = Path(tmpdir)
            self.assertTrue((dest_path / "adapter_config.json").is_file())
            self.assertTrue((dest_path / "adapter_model.pkl").is_file())
            self.assertTrue((dest_path / "adapter_metadata.json").is_file())

            # Load into fresh adapter
            restored = SummarizationAdapter("restored_lora", base_model=base_llm)
            restored.load(tmpdir)
            self.assertEqual(restored.adapter_weights["lora_A"]["layer_0"], [0.1, 0.2])
            self.assertEqual(restored.adapter_config.r, 16)

        # 4. AdapterTrainer
        class DummyDataset2(Dataset):
            def load(self, source=None, **kwargs):
                return self

        trainer = AdapterTrainer()
        res = trainer.fit(adapter_model, DummyDataset2("d"), epochs=2)
        self.assertEqual(res["status"], "completed")
        self.assertEqual(res["epochs"], 2)

        # 5. Under-the-hood auto-registration under "model" and "adapter"
        self.assertIs(get("adapter", "SummarizationAdapter"), SummarizationAdapter)
        self.assertIs(get("model", "SummarizationAdapter"), SummarizationAdapter)


if __name__ == "__main__":
    unittest.main()


