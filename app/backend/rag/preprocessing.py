
import re
import pymorphy3


# Создаём морфологический анализатор
morph = pymorphy3.MorphAnalyzer()


def tokens(text: str) -> list[str]:
    """
    Разбивает текст на токены и приводит русские слова
    к начальной форме.
    """

    # Приводим текст к нижнему регистру
    text = text.lower()

    # Выделяем русские и английские слова, а также числа
    words = re.findall(r"[а-яё]+|[a-z0-9]+", text)

    result = []

    for word in words:
        # Лемматизируем только русские слова
        if re.fullmatch(r"[а-яё]+", word):
            word = morph.parse(word)[0].normal_form

        result.append(word)

    return result
