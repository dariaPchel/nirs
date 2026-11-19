
import json
from datetime import datetime
from pathlib import Path

from main import docs
from rag.bm25 import build_index
from rag.rag import run_rag


# Пути к файлам
BENCHMARK_DIR = Path(__file__).resolve().parent
BENCHMARK_FILE = BENCHMARK_DIR / "benchmark.json"
RESULTS_DIR = BENCHMARK_DIR / "results"

def load_benchmark():
    with open(BENCHMARK_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    root = data["questions"][0]  # the single top-level object

    normal_questions = root["normal_questions"]
    trap_questions = root["trap_questions"]
    settings = root["settings"]

    return normal_questions + trap_questions, settings


def calculate_recall(retrieved: list, relevant: list) -> float:
    """Рассчитывает Recall@k."""

    if not relevant:
        return 0.0

    found = set(retrieved) & set(relevant)
    return len(found) / len(relevant)


def calculate_reciprocal_rank(
    retrieved: list,
    relevant: list
) -> float:
    """Рассчитывает Reciprocal Rank."""

    relevant = set(relevant)

    for position, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1 / position

    return 0.0


def run_benchmark():
    """Запускает все тестовые вопросы и рассчитывает метрики."""

    questions, settings = load_benchmark()
    TOP_K = settings["retrieval_k"] # 3

    # Строим индекс один раз для всех вопросов
    bm25_index = build_index(docs)

    print("=" * 60)
    print("BENCHMARK FRAUDI")
    print("=" * 60)
    print(f"Документов в базе: {bm25_index['N']}")
    print(f"Тестовых вопросов: {len(questions)}")
    print(f"Размер выдачи Top-k: {TOP_K}")

    results = []

    recalls = []
    reciprocal_ranks = []

    for question_data in questions:
        question_id = question_data["id"]
        question_type = question_data["type"]
        question = question_data["question"]

        relevant_ids = question_data["relevant_doc_ids"]

        print("\n" + "=" * 60)
        print(f"{question_id} | {question_type}")
        print("=" * 60)
        print(f"Вопрос: {question}")

        # Запускаем RAG
        state = run_rag(
            question=question,
            docs=docs,
            bm25_index=bm25_index,
            k=TOP_K
        )

        # Получаем результаты поиска
        retrieved_ids = [
            item["id"] for item in state["retrieved"]
        ]

        print("\nРезультаты BM25:")

        for item in state["retrieved"]:
            print(
                f"- {item['id']}: {item['schema']} "
                f"(score: {item['score']:.4f})"
            )

        if not retrieved_ids:
            print("Документы не найдены.")

        # Метрики поиска считаем только для обычных вопросов
        if question_type == "normal":
            recall = calculate_recall(
                retrieved_ids,
                relevant_ids
            )

            reciprocal_rank = calculate_reciprocal_rank(
                retrieved_ids,
                relevant_ids
            )

            recalls.append(recall)
            reciprocal_ranks.append(reciprocal_rank)

            print(f"\nRecall@{TOP_K}: {recall:.4f}")
            print(f"Reciprocal Rank: {reciprocal_rank:.4f}")

        else:
            recall = None
            reciprocal_rank = None

        # Показываем ответ LLM
        print("\nОтвет FRAUDI:")
        print(state["answer"])

        print("\nИсточники LLM:")
        for source in state["sources"]:
            print(f"- {source['id']}: {source['schema']}")

        if not state["sources"]:
            print("Источники не использованы.")

        # Сохраняем результат
        result = {
            "question_id": question_id,
            "type": question_type,
            "question": question,
            "relevant_doc_ids": relevant_ids,
            "retrieved_doc_ids": retrieved_ids,
            "bm25_scores": {
                item["id"]: item["score"]
                for item in state["retrieved"]
            },
            "recall_at_k": recall,
            "reciprocal_rank": reciprocal_rank,
            "generated_answer": state["answer"],
            "generated_source_ids": [
                source["id"] for source in state["sources"]
            ],
            "refusal_correct": None,
            "answer_correct": None
        }

        results.append(result)

    # Итоговые метрики
    average_recall = (
        sum(recalls) / len(recalls) if recalls else 0.0
    )

    mrr = (
        sum(reciprocal_ranks) / len(reciprocal_ranks)
        if reciprocal_ranks else 0.0
    )

    print("\n" + "=" * 60)
    print("ИТОГОВЫЕ МЕТРИКИ")
    print("=" * 60)

    print(f"Average Recall@{TOP_K}: {average_recall:.4f}")
    print(f"MRR@{TOP_K}: {mrr:.4f}")

    # Сохраняем результаты в отдельный JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = RESULTS_DIR / f"results_{timestamp}.json"

    output = {
        "document_count": bm25_index["N"],
        "question_count": len(questions),
        "top_k": TOP_K,
        "average_recall_at_k": average_recall,
        "mrr_at_k": mrr,
        "results": results
    }

    with open(result_file, "w", encoding="utf-8") as file:
        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=4
        )

    print(f"\nРезультаты сохранены: {result_file}")


if __name__ == "__main__":
    run_benchmark()
