
from rag.bm25 import retrieve
from agent.llm import generate_answer


def build_context(hits: list) -> str:
    """
    Формирует контекст из найденных документов.
    """

    documents = []

    for hit in hits:
        document = f"""
Документ {hit['id']}

Схема мошенничества:
{hit['schema']}

Признаки мошенничества:
{hit['bad_signs']}

Что делать:
{hit['to_do']}
"""
        documents.append(document)

    return "\n\n".join(documents)


def run_rag(
    question: str,
    docs: dict,
    bm25_index: dict,
    k: int = 3
) -> dict:

    # 1. Ищем подходящие документы
    hits = retrieve(
        query=question,
        docs=docs,
        bm25_index=bm25_index,
        k=k
    )

    # 2. Если ничего не найдено, LLM не вызываем
    if not hits:
        return {
            "question": question,
            "answer": (
                "В базе знаний не нашлось документов, "
                "содержащих подходящую информацию "
                "для анализа вашей ситуации."
            ),
            "sources": [],
            "retrieved": []
        }

    # 3. Формируем контекст
    context = build_context(hits)

    user_message = f"""
Описание ситуации пользователя:
{question}

Документы базы знаний:
{context}

Проанализируй ситуацию и сформируй ответ
строго по предоставленным документам.
"""

    # 4. Получаем ответ LLM
    result = generate_answer(user_message)

    # 5. Проверяем, что LLM использовала только найденные документы
    valid_ids = {hit["id"] for hit in hits}

    source_ids = [
        doc_id
        for doc_id in result.source_ids
        if doc_id in valid_ids
    ]

    # 6. Оставляем только документы, на которые опирается ответ
    sources = [
        hit
        for hit in hits
        if hit["id"] in source_ids
    ]

    return {
        "question": question,
        "answer": result.answer,
        "sources": sources,
        "retrieved": hits
    }
