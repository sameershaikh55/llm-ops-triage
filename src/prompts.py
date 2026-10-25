import json
from functools import lru_cache

import yaml
from pydantic import BaseModel, Field

from src.config import DEFAULT_PROMPT_NAME, DEFAULT_PROMPT_VERSION, PROMPTS_DIR
from src.schema import TriageOutput


class PromptConfig(BaseModel):
    """Strict schema validating any YAML prompt file in the prompt registry."""

    version: str = Field(description="Explicit version identifier, e.g. 'v1' or 'v3'.")
    name: str = Field(description="Task or workflow name, e.g. 'triage'.")
    description: str = Field(default="")
    model: str = Field(description="Default model ID associated with this prompt.")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, gt=0)
    system_template: str = Field(
        description="System prompt template with {json_schema}."
    )
    user_template: str = Field(description="User prompt template with {ticket_text}.")

    def render_messages(self, ticket_text: str) -> list[dict[str, str]]:
        """Render the system and user messages with the live Pydantic JSON schema."""
        schema_str = json.dumps(TriageOutput.model_json_schema(), indent=2)
        system_content = self.system_template.format(json_schema=schema_str).strip()
        user_content = self.user_template.format(ticket_text=ticket_text).strip()

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]


@lru_cache(maxsize=16)
def load_prompt(
    name: str = DEFAULT_PROMPT_NAME, version: str = DEFAULT_PROMPT_VERSION
) -> PromptConfig:
    """
    Load and validate a versioned YAML prompt from prompts/<name>/<version>.yaml.
    Cached in memory so repeated API requests incur zero disk I/O overhead.
    """
    prompt_path = PROMPTS_DIR / name / f"{version}.yaml"
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Prompt artifact not found at: {prompt_path}. "
            f"Check PROMPT_VERSION='{version}'."
        )

    with prompt_path.open("r", encoding="utf-8") as f:
        raw_data = yaml.safe_load(f)

    if not isinstance(raw_data, dict):
        raise ValueError(  # noqa: TRY004
            f"Invalid YAML structure in {prompt_path}: expected a mapping."
        )

    config = PromptConfig.model_validate(raw_data)
    if config.version != version:
        raise ValueError(
            f"Version mismatch in {prompt_path}: filename is '{version}.yaml' "
            f"but internal YAML version field is '{config.version}'."
        )

    return config
