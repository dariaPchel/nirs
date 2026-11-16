
from collections import defaultdict, Counter
from math import log

from preprocessing import tokens


# Параметры алгоритма BM25
k1 = 1.5
b = 0.75


def build_index(docs: dict) -> dict:
    """
    Строит инвертированный индекс BM25.

    docs — словарь документов:
    {
        "D01": {"title": "...", "content": "..."},
        ...
    }
    """

    index = defaultdict(dict)
    lengths = {}

    # Проходим по всем документам
    for doc, data in docs.items():

        # Токенизируем содержимое документа
        words = tokens(data["content"])

        # Сохраняем длину документа
        lengths[doc] = len(words)

        # Считаем частоту каждого слова в документе
        counts = Counter(words)

        # Записываем частоты в инвертированный индекс
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


def retrieve(query: str, docs: dict, bm25_index: dict, k: int = 5) -> list:
    """
    Ищет наиболее релевантные документы по запросу.

    Возвращает k документов с наибольшими оценками BM25.
    """

    index = bm25_index["index"]
    lengths = bm25_index["lengths"]
    avgdl = bm25_index["avgdl"]
    N = bm25_index["N"]

    scores = defaultdict(float)

    # Убираем повторяющиеся слова из запроса
    for word in dict.fromkeys(tokens(query)):

        # Получаем документы, содержащие слово
        postings = index.get(word, {})

        # Количество документов, содержащих это слово
        df = len(postings)

        # IDF — оценка редкости слова
        idf = log(1 + (N - df + 0.5) / (df + 0.5))

        # Рассчитываем вклад слова в оценку каждого документа
        for doc, tf in postings.items():

            norm = k1 * (
                1 - b + b * lengths[doc] / avgdl
            )

            scores[doc] += (
                idf * tf * (k1 + 1) / (tf + norm)
            )

    # Сортируем документы по убыванию оценки BM25
    ranked = sorted(
        scores,
        key=lambda doc: (-scores[doc], doc)
    )

    # Формируем результат поиска
    results = []

    for doc in ranked[:k]:
        results.append({
            "id": doc,
            "title": docs[doc]["title"],
            "content": docs[doc]["content"],
            "score": scores[doc],
            "source": docs[doc].get("source", "")
        })

    return results
