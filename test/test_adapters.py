"""Unit tests for AIMLite's LoRA & Parameter-Efficient Fine-Tuning (PEFT) subsystem."""

import json
import tempfile
import unittest
from pathlib import Path

from aimlite.adapters import (
    AdapterConfig,
    AdapterModel,
    AdapterTrainer,
    LoRALayer,
    MultiAdapterManager,
)
from aimlite.data import Dataset


class DummyInstructionDataset(Dataset):
    """Simple test dataset for instruction tuning."""

    filename = "instructions.json"

    def load(self, **kwargs):
        return [
            {"instruction": "Summarize this bug report.", "response": "Fixed crash on startup."},
            {"instruction": "Translate hello to French.", "response": "Bonjour."},
        ]


class TestLoRAEngine(unittest.TestCase):
    """Test suite covering low-rank math, PEFT configs, multi-adapter routing, and trainers."""

    def test_adapter_config_peft_interop(self):
        """Tests that AdapterConfig serializes to and parses standard Hugging Face PEFT configs."""
        cfg = AdapterConfig(
            r=8,
            alpha=16.0,
            target_modules=["q_proj", "v_proj"],
            dropout=0.05,
            base_model_path="meta-llama/Llama-3-8B",
            task_type="CAUSAL_LM",
        )

        # 1. Scaling calculation: alpha / r = 16.0 / 8 = 2.0
        self.assertEqual(cfg.scaling, 2.0)

        # 2. Export to PEFT dictionary
        peft_dict = cfg.to_peft_dict()
        self.assertEqual(peft_dict["peft_type"], "LORA")
        self.assertEqual(peft_dict["r"], 8)
        self.assertEqual(peft_dict["lora_alpha"], 16.0)
        self.assertEqual(peft_dict["base_model_name_or_path"], "meta-llama/Llama-3-8B")
        self.assertEqual(peft_dict["target_modules"], ["q_proj", "v_proj"])

        # 3. Round-trip from PEFT dictionary
        loaded_cfg = AdapterConfig.from_dict(peft_dict)
        self.assertEqual(loaded_cfg.r, 8)
        self.assertEqual(loaded_cfg.alpha, 16.0)
        self.assertEqual(loaded_cfg.base_model_path, "meta-llama/Llama-3-8B")
        self.assertEqual(loaded_cfg.scaling, 2.0)

        # 4. Save and load JSON file
        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / "adapter_config.json"
            cfg.save(dest)
            self.assertTrue(dest.is_file())

            reloaded = AdapterConfig.load(dest)
            self.assertEqual(reloaded.r, 8)
            self.assertEqual(reloaded.alpha, 16.0)

    def test_lora_layer_mathematics(self):
        """Tests low-rank matrix decomposition, initial zero delta, and weight merging."""
        in_dim = 4
        out_dim = 4
        r = 2
        alpha = 4.0

        # Create known base weights W0 (identity matrix)
        base_w = [
            [1.0 if i == j else 0.0 for j in range(in_dim)]
            for i in range(out_dim)
        ]

        layer = LoRALayer(
            in_features=in_dim,
            out_features=out_dim,
            r=r,
            alpha=alpha,
            base_weights=base_w,
        )

        # Scaling: alpha / r = 4.0 / 2 = 2.0
        self.assertEqual(layer.scaling, 2.0)

        # 1. At step 0, B is initialized to zeros -> delta W must be all zeros
        delta_0 = layer.delta_weight()
        for row in delta_0:
            for val in row:
                self.assertEqual(val, 0.0)

        # Forward pass on x = [1.0, 2.0, 3.0, 4.0] must equal W0 * x initially
        x = [1.0, 2.0, 3.0, 4.0]
        h_init = layer.forward(x)
        self.assertEqual(h_init, [1.0, 2.0, 3.0, 4.0])

        # 2. Simulate training: set non-zero values in B
        layer.lora_B = [
            [0.1, 0.2],
            [0.3, 0.4],
            [0.0, 0.1],
            [0.2, 0.0],
        ]

        # Delta must now be non-zero
        delta_trained = layer.delta_weight()
        self.assertTrue(any(v != 0.0 for row in delta_trained for v in row))

        # Forward pass output must now reflect base + low-rank delta
        h_adapted = layer.forward(x)
        self.assertNotEqual(h_adapted, h_init)

        # 3. Test weight merging: folds delta into W0 for zero-overhead inference
        layer.merge_weights()
        self.assertTrue(layer.merged)

        # Forward pass after merging must produce identical adapted representations
        h_merged = layer.forward(x)
        for val_a, val_m in zip(h_adapted, h_merged):
            self.assertAlmostEqual(val_a, val_m, places=6)

        # 4. Test weight unmerging: restores base weights
        layer.unmerge_weights()
        self.assertFalse(layer.merged)
        for i in range(out_dim):
            for j in range(in_dim):
                expected = 1.0 if i == j else 0.0
                self.assertAlmostEqual(layer.base_weights[i][j], expected, places=6)

    def test_multi_adapter_management(self):
        """Tests multi-adapter registration, dynamic hot-swapping, and disabling."""
        manager = MultiAdapterManager()

        cfg_support = AdapterConfig(r=8, alpha=16.0, base_model_path="llama-3")
        cfg_coder = AdapterConfig(r=16, alpha=32.0, base_model_path="llama-3")

        manager.add_adapter("support", cfg_support)
        manager.add_adapter("coder", cfg_coder)

        # Default active adapter is the first one added
        self.assertEqual(manager.active_adapter_name, "support")
        self.assertEqual(manager.get_active_adapter()[0], "support")
        self.assertEqual(manager.list_adapters(), ["support", "coder"])

        # Hot-swap to coding adapter
        manager.set_active_adapter("coder")
        self.assertEqual(manager.get_active_adapter()[0], "coder")
        self.assertEqual(manager.get_active_adapter()[1].r, 16)

        # Disable adapters -> returns None (pure base model mode)
        manager.disable_adapters()
        self.assertIsNone(manager.get_active_adapter())

        # Re-enable adapters
        manager.enable_adapters()
        self.assertEqual(manager.get_active_adapter()[0], "coder")

        # Switching to non-existent adapter raises KeyError
        with self.assertRaises(KeyError):
            manager.set_active_adapter("non_existent")

    def test_adapter_model_lifecycle_and_diagnostics(self):
        """Tests AdapterModel parameter diagnostics, freezing, and delta serialization."""
        model = AdapterModel(name="test_peft_model", lora_rank=8, lora_alpha=16.0)

        # 1. Base model is frozen by default
        self.assertTrue(model.base_model_frozen)

        # 2. Parameter efficiency diagnostics
        stats = model.get_trainable_parameters()
        self.assertIn("trainable_params", stats)
        self.assertIn("all_params", stats)
        self.assertIn("trainable_percent", stats)
        self.assertIn("memory_reduction_percent", stats)
        self.assertGreater(stats["trainable_params"], 0)
        self.assertGreater(stats["memory_reduction_percent"], 0.0)

        # Console string verification
        msg = model.print_trainable_parameters()
        self.assertIn("trainable params:", msg)
        self.assertIn("memory savings", msg)

        # 3. Delta-only checkpointing
        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / "checkpoint"
            model.save(dest)

            self.assertTrue((dest / "adapter_config.json").is_file())
            self.assertTrue((dest / "adapter_model.pkl").is_file())
            self.assertTrue((dest / "adapter_metadata.json").is_file())

            # Verify metadata contains parameter stats
            with open(dest / "adapter_metadata.json", "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.assertEqual(meta["name"], "test_peft_model")
            self.assertTrue(meta["is_adapter"])
            self.assertEqual(meta["active_adapter"], "default")

            # Restore into a fresh model
            restored_model = AdapterModel(name="restored_model")
            restored_model.load(dest)
            self.assertEqual(restored_model.adapter_config.r, model.adapter_config.r)

    def test_adapter_trainer_fit(self):
        """Tests AdapterTrainer training loop, loss convergence, and metric reporting."""
        model = AdapterModel(name="train_lora_model")
        dataset = DummyInstructionDataset(name="dummy_ds")

        trainer = AdapterTrainer()
        results = trainer.fit(model, dataset, epochs=4, lr=1e-3)

        self.assertEqual(results["status"], "completed")
        self.assertEqual(results["epochs"], 4)
        self.assertEqual(len(results["loss_history"]), 4)
        self.assertLess(results["final_loss"], results["loss_history"][0])
        self.assertIn("parameter_stats", results)


if __name__ == "__main__":
    unittest.main()
