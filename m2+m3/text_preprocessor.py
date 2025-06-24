import re
import string
import logging
import sys
from logging.handlers import RotatingFileHandler
import nltk
from nltk.corpus import stopwords
from pymystem3 import Mystem
from langdetect import detect, LangDetectException
from nltk.stem import WordNetLemmatizer
from functools import lru_cache

# --- Настройка логирования ---
try:
    from data_loader import setup_logger
except ImportError:
    def setup_logger(name, log_file='text_preprocessor.log', level=logging.INFO):
        logger = logging.getLogger(name)
        logger.setLevel(level)
        if not logger.handlers:
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            file_handler = RotatingFileHandler(log_file, maxBytes=1024 * 1024, backupCount=5, encoding='utf-8')
            file_handler.setFormatter(formatter)
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        return logger

logger = setup_logger('text_preprocessor_module', 'text_preprocessor.log')

# --- Проверка и загрузка необходимых ресурсов NLTK ---
@lru_cache(maxsize=1)
def check_and_download_nltk_resource(resource_name, download_name=None):
    if download_name is None:
        download_name = resource_name

    try:
        nltk.data.find(f'corpora/{resource_name}')
        logger.info(f"NLTK ресурс '{resource_name}' уже загружен.")
    except LookupError:
        logger.info(f"NLTK ресурс '{resource_name}' не найден. Попытка загрузки...")
        try:
            nltk.download(download_name, quiet=True)
            logger.info(f"NLTK ресурс '{resource_name}' успешно загружен.")
        except Exception as e:
            logger.error(f"Ошибка при загрузке NLTK ресурса '{resource_name}': {e}", exc_info=True)
            sys.exit(1)

check_and_download_nltk_resource('stopwords')
check_and_download_nltk_resource('wordnet')

# Инициализация лемматизаторов
mystem = None
try:
    mystem = Mystem()
    logger.info("Mystem лемматизатор инициализирован.")
except Exception as e:
    logger.error(f"Ошибка при инициализации Mystem: {e}. Русская лемматизация будет пропущена.", exc_info=True)

wordnet_lemmatizer = WordNetLemmatizer()
logger.info("WordNetLemmatizer (для английского) инициализирован.")

# Загрузка стоп-слов
try:
    russian_stopwords = set(stopwords.words('russian'))
    english_stopwords = set(stopwords.words('english'))
    logger.info(f"Загружено {len(russian_stopwords)} русских и {len(english_stopwords)} английских стоп-слов.")
except Exception as e:
    logger.error(f"Ошибка при загрузке списков стоп-слов: {e}. Работа без стоп-слов.", exc_info=True)
    russian_stopwords = set()
    english_stopwords = set()

# Дополнительные стоп-слова
additional_stopwords = {'это', 'который', 'человек', 'новый', 'проходить', 'год', 'весь', 'наш'}
russian_stopwords.update(additional_stopwords)
logger.info(f"Добавлено {len(additional_stopwords)} специфичных стоп-слов.")

# Кэширование часто используемых функций
@lru_cache(maxsize=1000)
def extract_hashtags(text):
    """Извлекает хэштеги из текста."""
    if not isinstance(text, str):
        return []
    hashtags = re.findall(r'#(\w+)', text)
    return hashtags

@lru_cache(maxsize=1000)
def clean_text(text):
    """Очищает текст от ненужных символов и ссылок."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S*@\S*\s?', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)  # Удаляем сами хэштеги
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@lru_cache(maxsize=1000)
def lemmatize_russian_text(text):
    """Лемматизирует русский текст с помощью Mystem."""
    if mystem is None:
        return text

    try:
        lemmas_raw = mystem.lemmatize(text)
        filtered_lemmas = []
        for word in lemmas_raw:
            word_stripped = word.strip()
            if word_stripped and word_stripped != '\n':
                # Фильтруем короткие бессмысленные слова
                if len(word_stripped) <= 2 and word_stripped not in ['ки', 'сша', 'ул', 'рф', 'ес', 'ит']:
                    continue
                filtered_lemmas.append(word_stripped)
        return " ".join(filtered_lemmas)
    except Exception as e:
        logger.error(f"Ошибка при русской лемматизации текста: {text[:50]}... Ошибка: {e}", exc_info=True)
        return text

@lru_cache(maxsize=1000)
def lemmatize_english_word(word):
    """Лемматизирует английское слово с помощью WordNetLemmatizer."""
    # Сохраняем акронимы длиной 2–5 букв без изменения
    if re.fullmatch(r'[A-Z]{2,5}|[a-z]{2,5}', word):
        return word
    return wordnet_lemmatizer.lemmatize(word)

@lru_cache(maxsize=1000)
def remove_stopwords_by_lang(text, lang_code):
    """Удаляет стоп-слова из текста в зависимости от языка."""
    if not isinstance(text, str):
        return ""
    words = text.split()
    if lang_code == 'ru':
        filtered_words = [word for word in words if word not in russian_stopwords]
    elif lang_code == 'en':
        filtered_words = [word for word in words if word not in english_stopwords]
    else:
        filtered_words = words
    return " ".join(filtered_words)

def preprocess_text_for_clustering(text):
    """Главная функция предобработки текста для кластеризации/тематического моделирования."""
    if not isinstance(text, str) or not text.strip():
        logger.debug(f"Входной текст пуст или некорректен: '{text}'")
        return "", []

    logger.debug(f"Исходный текст: '{text[:100]}'")

    raw_hashtags = extract_hashtags(text)
    processed_hashtags = []

    for hashtag_word in raw_hashtags:
        cleaned_hashtag = hashtag_word.lower()
        lemmatized_ht = cleaned_hashtag  # По умолчанию оставляем как есть

        # Определение языка хештега
        lang = 'unknown'
        try:
            lang = detect(cleaned_hashtag)
        except LangDetectException:
            lang = 'unknown'
        except Exception as e:
            logger.warning(f"Ошибка при определении языка хештега '{hashtag_word}': {e}. Обработка как 'unknown'.")
            lang = 'unknown'

        if lang == 'ru':
            lemmatized_ht = lemmatize_russian_text(cleaned_hashtag)
            lemmatized_ht = remove_stopwords_by_lang(lemmatized_ht, 'ru')
        elif lang == 'en':
            lemmatized_ht = lemmatize_english_word(cleaned_hashtag)
            lemmatized_ht = remove_stopwords_by_lang(lemmatized_ht, 'en')

        if lemmatized_ht and lemmatized_ht.strip():
            processed_hashtags.append(lemmatized_ht.strip())

    # Удаляем дублирующие хэштеги
    processed_hashtags = list(set(processed_hashtags))

    logger.debug(f"Извлеченные и обработанные хештеги: {processed_hashtags}")

    cleaned_main_text = clean_text(text)
    lemmatized_main_text = lemmatize_russian_text(cleaned_main_text)
    final_main_text = remove_stopwords_by_lang(lemmatized_main_text, 'ru')

    combined_text_parts = []
    if final_main_text.strip():
        combined_text_parts.append(final_main_text.strip())
    if processed_hashtags:
        combined_text_parts.append(' '.join(processed_hashtags))

    final_processed_text = ' '.join(combined_text_parts)
    logger.debug(f"Финальный обработанный текст (с хештегами): '{final_processed_text[:100]}'")

    return final_processed_text.strip(), raw_hashtags

# --- Пример использования модуля в режиме тестирования ---
if __name__ == '__main__':
    import pandas as pd

    logger.info("=== Запуск модуля text_preprocessor.py  ===")

    try:
        from data_loader import load_data
    except ImportError:
        logger.critical("Не удалось импортировать data_loader.py. Убедитесь, что он находится в том же каталоге.")
        sys.exit(1)

    try:
        df = load_data('messages.csv')
        logger.info(f"Загружено {len(df)} записей из messages.csv.")

        if df.empty:
            logger.critical("Загруженный DataFrame пуст. Нечего обрабатывать.")
            sys.exit(1)

    except (FileNotFoundError, ValueError) as e:
        logger.critical(f"Критическая ошибка при загрузке данных: {e}. Программа будет завершена.")
        sys.exit(1)

    logger.info("Начало предобработки текстов, включая хештеги...")

    try:
        from tqdm.auto import tqdm
        tqdm.pandas()
        processed_results = df['text'].progress_apply(preprocess_text_for_clustering)
        logger.info("Предобработка текстов завершена.")
    except Exception as e:
        logger.error(f"Ошибка при применении предобработчика: {e}", exc_info=True)
        sys.exit(1)

    logger.info("Процесс завершён успешно!")