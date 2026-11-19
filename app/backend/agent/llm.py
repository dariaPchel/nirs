
from pathlib import Path

import yaml
from ollama import chat

from agent.schemas import AnalysisResult


PROMPTS_DIR = Path(__file__).parent / "prompts"

CHARACTER = (PROMPTS_DIR / "CHARACTER.md").read_text(
    encoding="utf-8"
)


def load_prompt(filename: str) -> str:
    with (PROMPTS_DIR / filename).open(
        encoding="utf-8"
    ) as file:
        data = yaml.safe_load(file)

    return data["system_prompt"]


ANALYSIS_PROMPT = load_prompt("analysis_prompt.yaml")
RECOMMENDATION_PROMPT = load_prompt("recommendation_prompt.yaml")

MODEL = "qwen2.5:3b"


def analyze_situation(user_message: str) -> AnalysisResult:
    response = chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": CHARACTER + "\n\n" + ANALYSIS_PROMPT
            },
            {
                "role": "user",
                "content": user_message
            }
        ],
        format=AnalysisResult.model_json_schema(),
        options={
            "temperature": 0
        }
    )

    return AnalysisResult.model_validate_json(
        response.message.content
    )


def generate_recommendations(
    question: str,
    description: str,
    to_do: str
) -> str:

    response = chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": CHARACTER + "\n\n" + RECOMMENDATION_PROMPT
            },
            {
                "role": "user",
                "content": f"""
Ситуация пользователя:
{question}

Результат расследования:
{description}

Рекомендации из базы знаний:
{to_do}
"""
            }
        ],
        options={
            "temperature": 0.2
        }
    )

    return response.message.content
