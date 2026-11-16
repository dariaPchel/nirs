
from bm25 import retrieve


def build_context(hits: list) -> str:
    """
    Формирует контекст из найденных документов.
    """

    context = ""

    for hit in hits:
        context += (
            f"Документ: {hit['title']}\n"
            f"Содержание: {hit['content']}\n\n"
        )

    return context


def build_messages(question: str, context: str) -> list:
    """
    Формирует сообщения для будущего запроса к LLM.
    """

    messages = [
        {
            "role": "system",
            "content": (
                "Ты помощник по выявлению признаков онлайн-мошенничества. "
                "Отвечай на вопрос пользователя, опираясь на предоставленный "
                "контекст. Не придумывай факты, которых нет в контексте. "
                "Если информации недостаточно, сообщи об этом."
            )
        },
        {
            "role": "user",
            "content": (
                f"Вопрос пользователя: {question}\n\n"
                f"Контекст из базы знаний:\n{context}"
            )
        }
    ]

    return messages


def run_rag(question: str, docs: dict, bm25_index: dict, k: int = 5) -> dict:
    """
    Выполняет поиск документов и подготавливает данные для LLM.
    """

    # Ищем наиболее релевантные документы
    hits = retrieve(question, docs, bm25_index, k)

    # Собираем контекст
    context = build_context(hits)

    # Формируем сообщения для LLM
    messages = build_messages(question, context)

    return {
        "question": question,
        "hits": hits,
        "context": context,
        "messages": messages
    }
