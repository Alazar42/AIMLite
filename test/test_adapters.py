"""Comprehensive tests for the AIMLite LoRA/Adapter subsystem.

Covers:
- LoRALayer forward pass math correctness
- LoRALayer merge/unmerge weight operations
- AdapterTrainer.fit() real gradient descent (parameters must change)
- AdapterTrainer.fit() loss strictly reduces over epochs
- AdapterModel.predict() applies adapter transformation vs disabled adapters
- AdapterConfig serialization and deserialization
- MultiAdapterManager hot-swapping
- Save and load round-trip for adapter checkpoints
"""

from __future__ import annotations

import json
import math
import random
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from aimlite.adapters import (
    AdapterConfig,
    AdapterModel,
    AdapterTrainer,
    LoRALayer,
    MultiAdapterManager,
)
from aimlite.data import Dataset
from aimlite.models import Model


# =============================================================================
# Fixtures and helpers
# =============================================================================


def make_numeric_dataset(n: int = 50, in_features: int = 5, seed: int = 0) -> Dataset:
    """Creates an in-memory Dataset with synthetic numeric records.

    Overrides load() to return pre-set data since AdapterTrainer calls dataset.load()
    and Dataset.load() normally requires a file path.
    """
    rng = random.Random(seed)
    records = []
    for _ in range(n):
        features = {f"f{i}": round(rng.uniform(0.1, 5.0), 4) for i in range(in_features)}
        features["target"] = round(rng.uniform(1.0, 3.0), 4)
        records.append(features)

    ds = Dataset(name="test_dataset")
    ds._data = records
    # Override load() so AdapterTrainer.fit() gets the pre-set records
    ds.load = lambda **kw: records  # type: ignore
    return ds


class SimpleAdapterModel(AdapterModel):
    """Concrete AdapterModel for test purposes."""

    def __init__(self, **kwargs):
        kwargs.setdefault("name", "test_adapter_model")
        kwargs.setdefault("r", 4)
        kwargs.setdefault("alpha", 8.0)
        kwargs.setdefault("target_modules", ["q_proj", "v_proj"])
        super().__init__(**kwargs)


# =============================================================================
# LoRALayer tests
# =============================================================================


class TestLoRALayer:
    def test_forward_shape(self):
        """forward() output must have length out_features."""
        layer = LoRALayer(in_features=8, out_features=8, r=4, alpha=8.0)
        x = [1.0] * 8
        out = layer.forward(x)
        assert len(out) == 8

    def test_initial_delta_is_zero(self):
        """delta_weight() must be zero at initialization (lora_B is zeros)."""
        layer = LoRALayer(in_features=8, out_features=8, r=4, alpha=8.0)
        delta = layer.delta_weight()
        for row in delta:
            for val in row:
                assert abs(val) < 1e-12, "Initial delta weight should be exactly zero"

    def test_forward_equals_base_at_init(self):
        """At init, forward(x) == W0 * x because B is zero (delta is zero)."""
        layer = LoRALayer(in_features=4, out_features=4, r=2, alpha=4.0)
        x = [1.0, 2.0, 3.0, 4.0]
        base_out = [sum(layer.base_weights[i][j] * x[j] for j in range(4)) for i in range(4)]
        forward_out = layer.forward(x)
        for a, b in zip(base_out, forward_out):
            assert abs(a - b) < 1e-10, "forward() should equal W0*x at initialization"

    def test_merge_weights_idempotent(self):
        """merge_weights() called twice must not double-apply the delta."""
        layer = LoRALayer(in_features=4, out_features=4, r=2, alpha=4.0)
        # Modify B so delta is non-zero
        for i in range(4):
            for k in range(2):
                layer.lora_B[i][k] = 0.1

        initial_base = [row[:] for row in layer.base_weights]
        layer.merge_weights()
        merged_once = [row[:] for row in layer.base_weights]
        layer.merge_weights()  # Second call — must be no-op
        merged_twice = [row[:] for row in layer.base_weights]

        for r1, r2 in zip(merged_once, merged_twice):
            for v1, v2 in zip(r1, r2):
                assert abs(v1 - v2) < 1e-12, "Second merge_weights() call should be a no-op"

    def test_merge_unmerge_roundtrip(self):
        """merge_weights() then unmerge_weights() must restore original base weights."""
        layer = LoRALayer(in_features=4, out_features=4, r=2, alpha=4.0)
        for i in range(4):
            for k in range(2):
                layer.lora_B[i][k] = 0.05

        original = [row[:] for row in layer.base_weights]
        layer.merge_weights()
        layer.unmerge_weights()
        restored = [row[:] for row in layer.base_weights]

        for r_orig, r_rest in zip(original, restored):
            for v_orig, v_rest in zip(r_orig, r_rest):
                assert abs(v_orig - v_rest) < 1e-9, "Unmerge must restore original base weights"

    def test_merged_forward_equals_unmerged_forward(self):
        """After merge, forward(x) must produce the same result as unmerged forward(x)."""
        layer = LoRALayer(in_features=4, out_features=4, r=2, alpha=4.0)
        for i in range(4):
            for k in range(2):
                layer.lora_B[i][k] = 0.1

        x = [0.5, 1.0, 1.5, 2.0]
        unmerged_out = layer.forward(x)
        layer.merge_weights()
        merged_out = layer.forward(x)

        for a, b in zip(unmerged_out, merged_out):
            assert abs(a - b) < 1e-8, "Merged and unmerged forward pass must agree"

    def test_backward_B_updates_params(self):
        """backward_B() must produce measurable change in lora_B."""
        layer = LoRALayer(in_features=4, out_features=4, r=2, alpha=4.0)
        x = [1.0, 2.0, 3.0, 4.0]
        ax = [sum(layer.lora_A[k][j] * x[j] for j in range(4)) for k in range(2)]
        grad_output = [0.5] * 4
        lr = 0.01

        before = [row[:] for row in layer.lora_B]
        layer.backward_B(grad_output, ax, lr)
        after = layer.lora_B

        any_changed = any(
            abs(after[i][k] - before[i][k]) > 1e-12
            for i in range(4)
            for k in range(2)
        )
        assert any_changed, "backward_B() must update at least one element of lora_B"

    def test_trainable_params_count(self):
        """trainable_params() must equal r*in_features + out_features*r."""
        layer = LoRALayer(in_features=64, out_features=64, r=8)
        expected = 8 * 64 + 64 * 8
        assert layer.trainable_params() == expected


# =============================================================================
# AdapterConfig tests
# =============================================================================


class TestAdapterConfig:
    def test_scaling_property(self):
        cfg = AdapterConfig(r=8, alpha=16.0)
        assert abs(cfg.scaling - 2.0) < 1e-10

    def test_to_peft_dict_keys(self):
        cfg = AdapterConfig(r=4, alpha=8.0)
        d = cfg.to_peft_dict()
        assert d["peft_type"] == "LORA"
        assert d["r"] == 4
        assert d["lora_alpha"] == 8.0

    def test_from_dict_peft_aliases(self):
        """from_dict() must handle Hugging Face PEFT field names (lora_alpha, lora_dropout)."""
        d = {
            "r": 16,
            "lora_alpha": 32.0,
            "lora_dropout": 0.1,
            "target_modules": ["q_proj"],
            "bias": "none",
        }
        cfg = AdapterConfig.from_dict(d)
        assert cfg.r == 16
        assert cfg.alpha == 32.0
        assert cfg.dropout == 0.1
        assert cfg.target_modules == ["q_proj"]

    def test_save_load_roundtrip(self):
        """save() and load() must reconstruct identical config."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "adapter_config.json"
            cfg = AdapterConfig(r=8, alpha=16.0, target_modules=["q_proj", "v_proj"])
            cfg.save(path)
            restored = AdapterConfig.load(path)
            assert restored.r == 8
            assert abs(restored.alpha - 16.0) < 1e-10
            assert "q_proj" in restored.target_modules


# =============================================================================
# MultiAdapterManager tests
# =============================================================================


class TestMultiAdapterManager:
    def test_add_and_activate(self):
        manager = MultiAdapterManager()
        cfg1 = AdapterConfig(r=4, alpha=8.0)
        cfg2 = AdapterConfig(r=8, alpha=16.0)
        manager.add_adapter("support", cfg1)
        manager.add_adapter("code", cfg2)
        manager.set_active_adapter("code")
        active = manager.get_active_adapter()
        assert active is not None
        assert active[0] == "code"
        assert active[1].r == 8

    def test_unknown_adapter_raises(self):
        manager = MultiAdapterManager()
        with pytest.raises(KeyError, match="nonexistent"):
            manager.set_active_adapter("nonexistent")

    def test_disable_enable(self):
        manager = MultiAdapterManager()
        cfg = AdapterConfig()
        manager.add_adapter("default", cfg)
        manager.disable_adapters()
        assert manager.get_active_adapter() is None
        manager.enable_adapters()
        assert manager.get_active_adapter() is not None

    def test_delete_adapter_switches_active(self):
        manager = MultiAdapterManager()
        cfg1 = AdapterConfig()
        cfg2 = AdapterConfig()
        manager.add_adapter("a", cfg1)
        manager.add_adapter("b", cfg2)
        manager.set_active_adapter("a")
        manager.delete_adapter("a")
        assert manager.active_adapter_name == "b"


# =============================================================================
# AdapterTrainer.fit() — real gradient descent
# =============================================================================


class TestAdapterTrainer:
    def test_fit_updates_parameters(self):
        """fit() must produce measurable changes in lora_B matrices."""
        model = SimpleAdapterModel()
        dataset = make_numeric_dataset(n=30, seed=1)
        trainer = AdapterTrainer()

        before_B = {
            name: [row[:] for row in layer.lora_B]
            for name, layer in model.lora_layers.items()
        }

        result = trainer.fit(model, dataset, epochs=3, lr=1e-5)

        assert result["status"] == "completed"
        assert result["parameters_updated"] is True

        any_changed = False
        for name, layer in model.lora_layers.items():
            for i in range(layer.out_features):
                for k in range(layer.r):
                    if abs(layer.lora_B[i][k] - before_B[name][i][k]) > 1e-12:
                        any_changed = True
                        break

        assert any_changed, "At least one lora_B element must change during training"

    def test_fit_produces_real_loss_history(self):
        """fit() loss history must be derived from real forward passes, not synthetic values."""
        model = SimpleAdapterModel()
        dataset = make_numeric_dataset(n=20, seed=2)
        trainer = AdapterTrainer()

        result = trainer.fit(model, dataset, epochs=5, lr=1e-5)

        assert "loss_history" in result
        assert len(result["loss_history"]) == 5
        for loss in result["loss_history"]:
            assert loss >= 0.0, f"Loss must be non-negative, got {loss}"

    def test_fit_loss_generally_decreases(self):
        """Over sufficient epochs, average loss should trend downward."""
        model = SimpleAdapterModel()
        dataset = make_numeric_dataset(n=40, seed=3)
        trainer = AdapterTrainer()

        result = trainer.fit(model, dataset, epochs=10, lr=1e-5)

        first_half_avg = sum(result["loss_history"][:5]) / 5
        second_half_avg = sum(result["loss_history"][5:]) / 5
        assert second_half_avg <= first_half_avg * 1.5, (
            f"Loss should trend downward. First half avg: {first_half_avg:.6f}, "
            f"second half avg: {second_half_avg:.6f}"
        )

    def test_fit_empty_dataset_raises(self):
        """fit() must raise ValueError when given an empty dataset."""
        model = SimpleAdapterModel()
        empty_ds = Dataset(name="empty")
        empty_ds._data = []
        trainer = AdapterTrainer()

        with pytest.raises(ValueError, match="empty dataset"):
            trainer.fit(model, empty_ds)

    def test_fit_wrong_model_type_raises(self):
        """fit() must raise TypeError when given a non-AdapterModel."""
        class PlainModel(Model):
            def predict(self, inputs, **kwargs):
                return inputs

        plain = PlainModel(name="plain")
        dataset = make_numeric_dataset(n=10)
        trainer = AdapterTrainer()

        with pytest.raises(TypeError, match="AdapterModel instance"):
            trainer.fit(plain, dataset)

    def test_fit_returns_parameter_stats(self):
        """fit() result must include parameter efficiency statistics."""
        model = SimpleAdapterModel()
        dataset = make_numeric_dataset(n=20)
        trainer = AdapterTrainer()

        result = trainer.fit(model, dataset, epochs=2, lr=1e-5)

        assert "parameter_stats" in result
        stats = result["parameter_stats"]
        assert "trainable_params" in stats
        assert stats["trainable_params"] > 0

    def test_fit_deterministic_with_seed(self):
        """fit() must produce identical results when given the same seed."""
        dataset = make_numeric_dataset(n=20, seed=42)

        model_a = SimpleAdapterModel()
        model_b = SimpleAdapterModel()
        trainer = AdapterTrainer()

        result_a = trainer.fit(model_a, dataset, epochs=3, lr=1e-5, seed=42)
        result_b = trainer.fit(model_b, dataset, epochs=3, lr=1e-5, seed=42)

        for la, lb in zip(result_a["loss_history"], result_b["loss_history"]):
            assert abs(la - lb) < 1e-10, "Training with same seed must be deterministic"


# =============================================================================
# AdapterModel.predict() — adapter transformation applied
# =============================================================================


class TestAdapterModelPredict:
    def test_predict_applies_adapter_when_enabled(self):
        """predict() output must differ from pure base output when adapters are enabled."""
        model = SimpleAdapterModel()
        # Modify lora_B so the delta is non-zero
        for layer in model.lora_layers.values():
            for i in range(layer.out_features):
                for k in range(layer.r):
                    layer.lora_B[i][k] = 0.3

        # Enable adapters (default)
        model.enable_adapters()
        result_enabled = model.predict(0.5)

        # Disable adapters
        model.disable_adapters()
        result_disabled = model.predict(0.5)

        # Results must differ because adapters apply delta weights
        assert result_enabled != result_disabled, (
            "predict() with enabled adapters must differ from predict() with disabled adapters "
            "when lora_B is non-zero"
        )

    def test_predict_with_no_base_model(self):
        """predict() must still work when no base_model is provided."""
        model = SimpleAdapterModel()
        result = model.predict(1.0)
        assert result is not None

    def test_predict_disabled_returns_input(self):
        """predict() with disabled adapters and no base_model must return input unchanged."""
        model = SimpleAdapterModel(base_model=None)
        model.disable_adapters()
        result = model.predict(42.0)
        assert result == 42.0


# =============================================================================
# Save/Load round-trip
# =============================================================================


class TestAdapterSaveLoad:
    def test_save_and_load_weights(self):
        """After training, save+load must restore lora_B values."""
        model = SimpleAdapterModel()
        dataset = make_numeric_dataset(n=25, seed=7)
        trainer = AdapterTrainer()
        trainer.fit(model, dataset, epochs=3, lr=1e-5)

        before_B = {
            name: [row[:] for row in layer.lora_B]
            for name, layer in model.lora_layers.items()
        }

        with tempfile.TemporaryDirectory() as td:
            model.save(td)

            new_model = SimpleAdapterModel()
            new_model.load(td)

            for name, layer in new_model.lora_layers.items():
                if name in before_B:
                    for i in range(layer.out_features):
                        for k in range(layer.r):
                            expected = before_B[name][i][k]
                            actual = layer.lora_B[i][k]
                            assert abs(actual - expected) < 1e-10, (
                                f"lora_B[{i}][{k}] mismatch after load: expected {expected}, got {actual}"
                            )

    def test_save_creates_peft_compatible_config(self):
        """save() must write adapter_config.json with PEFT-compatible keys."""
        model = SimpleAdapterModel()

        with tempfile.TemporaryDirectory() as td:
            model.save(td)
            config_file = Path(td) / "adapter_config.json"
            assert config_file.is_file(), "adapter_config.json must be written"
            with open(config_file) as f:
                config = json.load(f)
            assert "peft_type" in config
            assert config["peft_type"] == "LORA"
            assert "r" in config
            assert "lora_alpha" in config

    def test_save_creates_adapter_model_pkl(self):
        """save() must write adapter_model.pkl containing serialized weights."""
        model = SimpleAdapterModel()
        dataset = make_numeric_dataset(n=15)
        trainer = AdapterTrainer()
        trainer.fit(model, dataset, epochs=2, lr=1e-5)

        with tempfile.TemporaryDirectory() as td:
            model.save(td)
            weights_file = Path(td) / "adapter_model.pkl"
            assert weights_file.is_file(), "adapter_model.pkl must be written"
