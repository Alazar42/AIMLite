"""Tests for aimlite init --clean and --type integration."""

import json
from pathlib import Path
from aimlite.cli.commands.init import run_init


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

    # 3. Verify code files are clean empty files with zero dummy/sample code
    assert (pkg / "model.py").read_text(encoding="utf-8") == ""
    assert (pkg / "trainer.py").read_text(encoding="utf-8") == ""
    assert (pkg / "evaluator.py").read_text(encoding="utf-8") == ""
    assert (pkg / "data.py").read_text(encoding="utf-8") == ""
    assert (pkg / "inference.py").read_text(encoding="utf-8") == ""
    # Sample dataset CSV should NOT exist in data/
    assert not (target / "data" / "dataset.csv").exists()
    assert (target / "data" / ".gitkeep").exists()

    # 4. Root files
    assert (target / "README.md").exists()
    assert (target / ".gitignore").exists()
    assert (target / ".env.example").exists()
    assert (target / ".agents" / "skills" / "aimlite" / "SKILL.md").exists()
    assert not (target / "skills").exists()
    assert (target / "AGENTS.md").exists()
    assert (target / "llms.txt").exists()


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

    pkg = target / "test_scratch_clean"
    assert (pkg / "model.py").read_text(encoding="utf-8") == ""
    assert (pkg / "data.py").read_text(encoding="utf-8") == ""


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

    # RAG starter code files are empty in clean mode
    pkg = target / "test_rag_clean"
    assert (pkg / "chat_provider.py").exists()
    assert (pkg / "chat_provider.py").read_text(encoding="utf-8") == ""
    assert (pkg / "store.py").exists()
    assert (pkg / "store.py").read_text(encoding="utf-8") == ""
    assert (pkg / "model.py").exists()
    assert (pkg / "model.py").read_text(encoding="utf-8") == ""

    # But sample documents in data/ are NOT generated
    assert not (target / "data" / "knowledge_base.md").exists()
    assert not (target / "data" / "faq.md").exists()
    assert (target / "data" / ".gitkeep").exists()


def test_init_clean_with_type_adapter(tmp_path: Path):
    target = tmp_path / "test_lora_clean"

    code = run_init(
        project_name="test_lora_clean",
        target_dir=str(tmp_path),
        template_type="adapter",
        create_venv=False,
        interactive=False,
        install_deps=False,
        clean=True,
    )
    assert code == 0
    assert target.is_dir()

    manifest = json.loads((target / "aimlite.json").read_text(encoding="utf-8"))
    assert manifest["template"] == "adapter"
    assert manifest["clean"] is True

    # Adapter starter code files are empty in clean mode
    pkg = target / "test_lora_clean"
    assert (pkg / "adapter.py").exists()
    assert (pkg / "adapter.py").read_text(encoding="utf-8") == ""
    assert (pkg / "model.py").exists()
    assert (pkg / "model.py").read_text(encoding="utf-8") == ""

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


def test_init_standard_adapter_has_readme(tmp_path: Path):
    target = tmp_path / "test_standard_lora"

    code = run_init(
        project_name="test_standard_lora",
        target_dir=str(tmp_path),
        template_type="adapter",
        create_venv=False,
        interactive=False,
        install_deps=False,
        clean=False,
    )
    assert code == 0
    assert target.is_dir()

    manifest = json.loads((target / "aimlite.json").read_text(encoding="utf-8"))
    assert manifest["template"] == "adapter"
    assert manifest["clean"] is False

    # Standard Adapter generates README.md and sample instructions
    assert (target / "README.md").exists()
    assert (target / "data" / "instructions.jsonl").exists()


def test_init_scaffolds_agent_skills_and_llms_txt(tmp_path: Path):
    target = tmp_path / "test_agent_rag"

    code = run_init(
        project_name="test_agent_rag",
        target_dir=str(tmp_path),
        template_type="rag",
        create_venv=False,
        interactive=False,
        install_deps=False,
        clean=True,
    )
    assert code == 0
    assert target.is_dir()

    # 1. Agent Skill Guide in .agents/
    agent_skill = target / ".agents" / "skills" / "aimlite" / "SKILL.md"
    assert agent_skill.exists()
    assert not (target / "skills").exists()
    skill_text = agent_skill.read_text(encoding="utf-8")
    assert "name: aimlite" in skill_text
    assert "Expert guide for AIMLite" in skill_text
    assert "aimlite train" in skill_text

    # 2. AGENTS.md at workspace root
    agents_md = target / "AGENTS.md"
    assert agents_md.exists()
    agents_text = agents_md.read_text(encoding="utf-8")
    assert "test_agent_rag" in agents_text
    assert "rag" in agents_text
    assert "Zero-Path CLI Commands" in agents_text

    # 3. llms.txt at workspace root
    llms_txt = target / "llms.txt"
    assert llms_txt.exists()
    llms_text = llms_txt.read_text(encoding="utf-8")
    assert "test_agent_rag" in llms_text
    assert "aimlite serve" in llms_text


def test_init_unknown_paradigm_raises_exception(tmp_path: Path):
    import pytest
    from aimlite.cli.commands.init import UnknownParadigmError

    with pytest.raises(UnknownParadigmError, match="Unknown paradigm type 'invalid_type'"):
        run_init(
            project_name="bad_proj",
            target_dir=str(tmp_path),
            template_type="invalid_type",
            interactive=False,
        )


def test_init_clean_accepts_adapters_type_alias(tmp_path: Path):
    target = tmp_path / "test_clean_adapters_alias"
    code = run_init(
        project_name="test_clean_adapters_alias",
        target_dir=str(tmp_path),
        template_type="adapters",
        create_venv=False,
        interactive=False,
        clean=True,
    )
    assert code == 0
    manifest = json.loads((target / "aimlite.json").read_text(encoding="utf-8"))
    assert manifest["template"] == "adapter"
    assert manifest["clean"] is True
    pkg = target / "test_clean_adapters_alias"
    assert (pkg / "adapter.py").exists()
    assert (pkg / "adapter.py").read_text(encoding="utf-8") == ""
    assert (pkg / "model.py").read_text(encoding="utf-8") == ""


def test_init_with_leading_dot_and_special_chars(tmp_path: Path):
    from aimlite.cli.discovery import resolve_project_context, sanitize_package_name

    assert sanitize_package_name(".my_ai") == "my_ai"
    assert sanitize_package_name("my-project") == "my_project"
    assert sanitize_package_name("123app") == "pkg_123app"

    code = run_init(
        project_name=".custom_rag",
        target_dir=str(tmp_path),
        template_type="rag",
        create_venv=False,
        interactive=False,
        clean=False,
    )
    assert code == 0
    target = tmp_path / ".custom_rag"
    assert target.is_dir()
    manifest = json.loads((target / "aimlite.json").read_text(encoding="utf-8"))
    assert manifest["name"] == ".custom_rag"
    assert manifest["entrypoint"] == "custom_rag"

    pkg_dir = target / "custom_rag"
    assert pkg_dir.is_dir()
    assert (pkg_dir / "model.py").exists()
    assert "from custom_rag.chat_provider import" in (pkg_dir / "model.py").read_text(encoding="utf-8")

    ctx = resolve_project_context(start_dir=target)
    assert ctx.model_cls is not None
    assert ctx.dataset_cls is not None
    assert ctx.model_cls.__name__ == "SupportDocRAG"


