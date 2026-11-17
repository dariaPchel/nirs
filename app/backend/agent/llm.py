
from ollama import chat

# Название используемой языковой модели
MODEL = "qwen2.5:3b"

def generate_answer(messages: list[dict]) -> str:
    """
    Отправляет сообщения в локальную Ollama
    и возвращает ответ языковой модели.
    """

    response = chat(
        model=MODEL,
        messages=messages,
        options={
            "temperature": 0.2
        }
    )

    return response.message.content
