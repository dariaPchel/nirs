
from collections import defaultdict, Counter
from math import log

from rag.preprocessing import tokens


# Параметры алгоритма BM25 из презентации
k1 = 1.2
b = 0.75


def build_index(docs: dict) -> dict:
    """
    Строит инвертированный индекс BM25.
    """

    index = defaultdict(dict)
    lengths = {}

    for doc, data in docs.items():

        # Объединяем содержательные поля документа
        text = (
            data["schema"] + " " +
            data["bad_signs"] + " " +
            data["to_do"]
        )

        # Токенизируем текст
        words = tokens(text)

        # Сохраняем длину документа
        lengths[doc] = len(words)

        # Считаем частоту каждого слова
        counts = Counter(words)

        # Заполняем инвертированный индекс
        for word, tf in counts.items():
            index[word][doc] = tf

    # Общее количество документов
    N = len(docs)

    # Средняя длина документа
    avgdl = sum(lengths.values()) / N if N > 0 else 0

    return {
        "index": index,
        "lengths": lengths,
        "avgdl": avgdl,
        "N": N
    }


def retrieve(
    query: str,
    docs: dict,
    bm25_index: dict,
    k: int = 3
) -> list:
    """
    Ищет k наиболее релевантных документов по запросу.
    """

    index = bm25_index["index"]
    lengths = bm25_index["lengths"]
    avgdl = bm25_index["avgdl"]
    N = bm25_index["N"]

    scores = defaultdict(float)

    # Обрабатываем уникальные слова запроса
    for word in dict.fromkeys(tokens(query)):

        # Получаем документы, содержащие слово
        postings = index.get(word, {})

        # Количество документов, содержащих слово
        df = len(postings)

        if df == 0:
            continue

        # Рассчитываем IDF
        idf = log(1 + (N - df + 0.5) / (df + 0.5))

        # Рассчитываем вклад слова в оценку документов
        for doc, tf in postings.items():

            if avgdl == 0:
                continue

            norm = k1 * (
                1 - b + b * lengths[doc] / avgdl
            )

            scores[doc] += (
                idf * tf * (k1 + 1) / (tf + norm)
            )

    # Сортируем документы по убыванию оценки
    ranked = sorted(
        scores,
        key=lambda doc: (-scores[doc], doc)
    )

    # Формируем результаты поиска
    results = []

    for doc in ranked[:k]:
        results.append({
            "id": doc,
            "schema": docs[doc]["schema"],
            "bad_signs": docs[doc]["bad_signs"],
            "to_do": docs[doc]["to_do"],
            "source": docs[doc].get("source", ""),
            "score": scores[doc]
        })

    return results
