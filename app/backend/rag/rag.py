
from rag.bm25 import retrieve
from agent.llm import analyze_situation, generate_recommendations


def build_context(hits: list) -> str:
    documents = []

    for hit in hits:
        document = f"""
Документ №{hit['id']}

Мошенническая схема:
{hit['schema']}

Признаки мошенничества:
{hit['bad_signs']}
"""
        documents.append(document)

    return "\n\n".join(documents)


def run_rag(
    question: str,
    docs: dict,
    bm25_index: dict,
    k: int = 1
) -> dict:

    # 1. Ищем подходящие документы
    hits = retrieve(question, docs, bm25_index, k)

    if not hits:
        return {
            "question": question,
            "document_id": None,
            "description": "В базе знаний не найдены подходящие документы.",
            "sources": [],
            "status": "finished"
        }

    # 2. Формируем компактный контекст
    context = build_context(hits)

    user_message = f"""
Ситуация пользователя:
{question}

Документы базы знаний:
{context}

Проанализируй ситуацию и верни структурированный результат.
"""

    # 3. Первый вызов LLM — расследование
    result = analyze_situation(user_message)

    # 4. Проверяем выбранный ID
    valid_ids = {hit["id"] for hit in hits}

    if result.document_id not in valid_ids:
        document_id = None
    else:
        document_id = result.document_id

    # 5. Возвращаем состояние расследования
    return {
        "question": question,
        "document_id": document_id,
        "description": result.description,
        "sources": hits,
        "status": (
            "awaiting_confirmation"
            if document_id is not None
            else "finished"
        )
    }


def continue_rag(state: dict, docs: dict) -> dict:

    document_id = state.get("document_id")

    # Проверяем, что продолжение действительно возможно
    if state.get("status") != "awaiting_confirmation":
        raise ValueError("Расследование не ожидает подтверждения.")

    if document_id not in docs:
        raise ValueError("Документ не найден в базе знаний.")

    # 1. Получаем рекомендации выбранного документа
    to_do = docs[document_id]["to_do"]

    # 2. Второй вызов LLM — рекомендации
    answer = generate_recommendations(
        question=state["question"],
        description=state["description"],
        to_do=to_do
    )

    return {
        **state,
        "answer": answer,
        "status": "finished"
    }
