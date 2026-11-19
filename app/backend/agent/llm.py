
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from groq import Groq

from agent.schemas import RAGResponse


# Загружаем переменные окружения
load_dotenv()


# Пути к промптам
PROMPTS_DIR = Path(__file__).parent / "prompts"


# Характер FRAUDI
CHARACTER = (PROMPTS_DIR / "CHARACTER.md").read_text(
    encoding="utf-8"
)


def load_prompt(filename: str) -> str:
    with (PROMPTS_DIR / filename).open(
        encoding="utf-8"
    ) as file:
        data = yaml.safe_load(file)

    return data["system_prompt"]


# Системный промпт
RAG_PROMPT = load_prompt("analysis_prompt.yaml")


# Настройки Groq
MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b"
)


def get_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "Не найден GROQ_API_KEY. "
            "Добавьте API-ключ в файл .env."
        )

    return Groq(api_key=api_key)


def generate_answer(user_message: str) -> RAGResponse:
    client = get_client()

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": CHARACTER + "\n\n" + RAG_PROMPT
            },
            {
                "role": "user",
                "content": user_message
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "fraudi_response",
                "strict": True,
                "schema": RAGResponse.model_json_schema()
            }
        },
        temperature=0
    )

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError(
            "FRAUDI вернул пустой ответ."
        )

    return RAGResponse.model_validate_json(content)
