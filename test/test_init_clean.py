"""Tests for aimlite init --clean and --type integration."""

import json
from pathlib import Path
from cli.commands.init import run_init


def test_init_clean_default_scratch(tmp_path: Path):
    target = tmp_path / "test_clean_proj"

    code = run_init(
        project_name="test_clean_proj",
        target_dir=str(tmp_path),
        create_venv=False,
        interactive=False,
        install_deps=False,
        clean=True,
    )
    assert code == 0
    assert target.is_dir()

    # 1. Check aimlite.json
    manifest_file = target / "aimlite.json"
    assert manifest_file.exists()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert manifest["name"] == "test_clean_proj"
    assert manifest["template"] == "scratch"
    assert manifest["clean"] is True
    assert manifest["dependencies"] == []

    # 2. Check Python package directory
    pkg = target / "test_clean_proj"
    assert (pkg / "__init__.py").exists()
    assert (pkg / "config.py").exists()
    assert (pkg / "data.py").exists()
    assert (pkg / "model.py").exists()
    assert (pkg / "trainer.py").exists()
    assert (pkg / "evaluator.py").exists()
    assert (pkg / "inference.py").exists()

    # 3. Verify files contain NO sample/dummy implementations, only skeletons and comments
    model_content = (pkg / "model.py").read_text(encoding="utf-8")
    assert "raise NotImplementedError" in model_content
    assert "class AppModel(Model):" in model_content

    trainer_content = (pkg / "trainer.py").read_text(encoding="utf-8")
    assert "raise NotImplementedError" in trainer_content
    assert "class AppTrainer(BaseTrainer):" in trainer_content

    evaluator_content = (pkg / "evaluator.py").read_text(encoding="utf-8")
    assert "raise NotImplementedError" in evaluator_content
    assert "class AppEvaluator(BaseEvaluator):" in evaluator_content

    data_content = (pkg / "data.py").read_text(encoding="utf-8")
    assert "class AppDataset(Dataset):" in data_content
    # Sample dataset CSV should NOT exist in data/
    assert not (target / "data" / "dataset.csv").exists()
    assert (target / "data" / ".gitkeep").exists()

    # 4. Root files
    assert (target / "experiments" / "benchmark.py").exists()
    assert (target / "client.py").exists()
    assert (target / "README.md").exists()
    assert (target / ".gitignore").exists()
    assert (target / ".env.example").exists()


def test_init_clean_with_type_scratch(tmp_path: Path):
    target = tmp_path / "test_scratch_clean"

    code = run_init(
        project_name="test_scratch_clean",
        target_dir=str(tmp_path),
        template_type="scratch",
        create_venv=False,
        interactive=False,
        install_deps=False,
        clean=True,
    )
    assert code == 0
    assert target.is_dir()

    manifest = json.loads((target / "aimlite.json").read_text(encoding="utf-8"))
    assert manifest["template"] == "scratch"
    assert manifest["clean"] is True
    assert not (target / "data" / "dataset.csv").exists()
    assert (target / "data" / ".gitkeep").exists()


def test_init_clean_with_type_rag(tmp_path: Path):
    target = tmp_path / "test_rag_clean"

    code = run_init(
        project_name="test_rag_clean",
        target_dir=str(tmp_path),
        template_type="rag",
        create_venv=False,
        interactive=False,
        install_deps=False,
        clean=True,
    )
    assert code == 0
    assert target.is_dir()

    manifest = json.loads((target / "aimlite.json").read_text(encoding="utf-8"))
    assert manifest["template"] == "rag"
    assert manifest["clean"] is True

    # RAG starter code exists
    pkg = target / "test_rag_clean"
    assert (pkg / "chat_provider.py").exists()
    assert (pkg / "store.py").exists()
    assert (pkg / "model.py").exists()

    # But sample documents in data/ are NOT generated
    assert not (target / "data" / "knowledge_base.md").exists()
    assert not (target / "data" / "faq.md").exists()
    assert (target / "data" / ".gitkeep").exists()


def test_init_clean_with_type_adapters(tmp_path: Path):
    target = tmp_path / "test_lora_clean"

    code = run_init(
        project_name="test_lora_clean",
        target_dir=str(tmp_path),
        template_type="fine-tuning",
        create_venv=False,
        interactive=False,
        install_deps=False,
        clean=True,
    )
    assert code == 0
    assert target.is_dir()

    manifest = json.loads((target / "aimlite.json").read_text(encoding="utf-8"))
    assert manifest["template"] == "fine-tuning"
    assert manifest["clean"] is True

    # Adapter starter code exists
    pkg = target / "test_lora_clean"
    assert (pkg / "adapter.py").exists()
    assert (pkg / "model.py").exists()

    # But sample instructions in data/ are NOT generated
    assert not (target / "data" / "instructions.jsonl").exists()
    assert (target / "data" / ".gitkeep").exists()


def test_init_standard_mode_still_has_samples(tmp_path: Path):
    target = tmp_path / "test_standard_scratch"

    code = run_init(
        project_name="test_standard_scratch",
        target_dir=str(tmp_path),
        template_type="scratch",
        create_venv=False,
        interactive=False,
        install_deps=False,
        clean=False,
    )
    assert code == 0
    assert target.is_dir()

    manifest = json.loads((target / "aimlite.json").read_text(encoding="utf-8"))
    assert manifest["template"] == "scratch"
    assert manifest["clean"] is False
    # Standard scratch generates sample dataset.csv
    assert (target / "data" / "dataset.csv").exists()


def test_init_standard_rag_has_samples(tmp_path: Path):
    target = tmp_path / "test_standard_rag"

    code = run_init(
        project_name="test_standard_rag",
        target_dir=str(tmp_path),
        template_type="rag",
        create_venv=False,
        interactive=False,
        install_deps=False,
        clean=False,
    )
    assert code == 0
    assert target.is_dir()

    manifest = json.loads((target / "aimlite.json").read_text(encoding="utf-8"))
    assert manifest["template"] == "rag"
    assert manifest["clean"] is False

    # Standard RAG generates sample knowledge_base.md and faq.md
    faq_file = target / "data" / "faq.md"
    kb_file = target / "data" / "knowledge_base.md"
    assert faq_file.exists()
    assert kb_file.exists()
    assert "Frequently Asked Questions" in faq_file.read_text(encoding="utf-8")
    assert "Knowledge Base" in kb_file.read_text(encoding="utf-8")


def test_init_standard_fine_tuning_has_readme(tmp_path: Path):
    target = tmp_path / "test_standard_lora"

    code = run_init(
        project_name="test_standard_lora",
        target_dir=str(tmp_path),
        template_type="fine-tuning",
        create_venv=False,
        interactive=False,
        install_deps=False,
        clean=False,
    )
    assert code == 0
    assert target.is_dir()

    # Standard Fine-Tuning generates README.md and sample instructions
    assert (target / "README.md").exists()
    assert (target / "data" / "instructions.jsonl").exists()
