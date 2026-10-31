import pytest

from src.config import PROMPTS_DIR
from src.prompts import PromptConfig, load_prompt


def test_load_v1_prompt_validates_and_renders() -> None:
    """Verify v1.yaml loads cleanly and injects both schema and ticket_text."""
    prompt_cfg = load_prompt(name="triage", version="v1")

    assert isinstance(prompt_cfg, PromptConfig)
    assert prompt_cfg.version == "v1"
    assert prompt_cfg.name == "triage"
    assert prompt_cfg.model == "openai/gpt-oss-20b"
    assert prompt_cfg.max_tokens >= 1024

    sample_ticket = "My payment failed twice on checkout."
    messages = prompt_cfg.render_messages(ticket_text=sample_ticket)

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert (
        "TriageOutput" in messages[0]["content"] or "category" in messages[0]["content"]
    )
    assert messages[1]["role"] == "user"
    assert sample_ticket in messages[1]["content"]


def test_all_yaml_files_in_registry_are_valid() -> None:
    """
    Automatically discover and validate every prompts/triage/*.yaml artifact.
    Guarantees any future v2.yaml or v3.yaml added to the repo conforms to PromptConfig.
    """
    triage_dir = PROMPTS_DIR / "triage"
    yaml_files = list(triage_dir.glob("*.yaml"))
    assert len(yaml_files) >= 1, "Expected at least one YAML prompt in prompts/triage/"

    for yaml_path in yaml_files:
        version_id = yaml_path.stem
        cfg = load_prompt(name="triage", version=version_id)
        assert cfg.version == version_id
        rendered = cfg.render_messages(ticket_text="Test ticket")
        assert len(rendered) == 2


def test_load_nonexistent_prompt_raises_file_not_found() -> None:
    """Requesting a missing prompt version must raise a clear FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="v999"):
        load_prompt(name="triage", version="v999")
