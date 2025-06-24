import re
import string
import logging
import sys
from logging.handlers import RotatingFileHandler
import nltk
from nltk.corpus import stopwords
from pymystem3 import Mystem # Для русской лемматизации

# Проверяем и скачиваем необходимые ресурсы NLTK, если их нет
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords', quiet=True)
    logging.info("NLTK stopwords загружены.")

# Инициализируем логгер
try:
    from data_loader import setup_logger 
except ImportError:
    def setup_logger(name, log_file='text_preprocessor.log', level=logging.INFO):
        logger = logging.getLogger(name)
        logger.setLevel(level)
        if not logger.handlers:
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            file_handler = RotatingFileHandler(log_file, maxBytes=1024*1024, backupCount=5, encoding='utf-8')
            file_handler.setFormatter(formatter)
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        return logger

logger = setup_logger('text_preprocessor_module', 'text_preprocessor.log')

# Инициализация лемматизатора Mystem
mystem = None
try:
    mystem = Mystem()
    logger.info("Mystem лемматизатор инициализирован.")
except Exception as e:
    logger.error(f"Ошибка при инициализации Mystem: {e}. Лемматизация будет пропущена.", exc_info=True)

# Загружаем русские стоп-слова
russian_stopwords = set(stopwords.words('russian'))
logger.info(f"Загружено {len(russian_stopwords)} русских стоп-слов.")

def extract_hashtags(text):
    """
    Извлекает хештеги из текста.
    Возвращает список хештегов (без символа #).
    """
    if not isinstance(text, str):
        return []
    # Найти все слова, начинающиеся с #
    hashtags = re.findall(r'#(\w+)', text)
    return hashtags

def clean_text(text):
    """
    Выполняет общую очистку текста, *но не удаляет хештеги сразу*.
    Удаляет URL, email, упоминания, цифры, лишние пробелы и пунктуацию.
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()

    # Удаление URL-адресов
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Удаление email-адресов
    text = re.sub(r'\S*@\S*\s?', '', text)

    # Удаление упоминаний (@)
    text = re.sub(r'@\w+', '', text)

    # Удаление цифр
    text = re.sub(r'\d+', '', text)

    # Удаление пунктуации (все символы из string.punctuation), но *не* #
    # Создаем таблицу для перевода, исключая '#'
    punctuations_to_remove = string.punctuation.replace('#', '')
    text = text.translate(str.maketrans('', '', punctuations_to_remove))

    # Удаление лишних пробелов
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def lemmatize_text(text):
    """
    Лемматизирует текст на русском языке с помощью Mystem.
    """
    if mystem is None:
        logger.warning("Mystem лемматизатор не инициализирован. Лемматизация пропущена.")
        return text 
        
    try:
        lemmas = mystem.lemmatize(text)
        return " ".join([word.strip() for word in lemmas if word.strip() and word.strip() != '\n'])
    except Exception as e:
        logger.error(f"Ошибка при лемматизации текста: {text[:50]}... Ошибка: {e}", exc_info=True)
        return text

def remove_stopwords(text, stopwords_set):
    """
    Удаляет стоп-слова из текста.
    """
    if not isinstance(text, str):
        return ""
        
    words = text.split()
    filtered_words = [word for word in words if word not in stopwords_set]
    return " ".join(filtered_words)


def preprocess_text_for_clustering(text):
    """
    Главная функция предобработки текста для кластеризации/тематического моделирования.
    Включает извлечение хештегов и их добавление к тексту.
    """
    if not isinstance(text, str) or not text.strip():
        return "", [] # Возвращаем пустую строку и пустой список хештегов

    logger.debug(f"Начало предобработки: {text[:50]}...")
    
    # 1. Извлечение хештегов до основной очистки
    raw_hashtags = extract_hashtags(text)
    # Лемматизируем извлеченные хештеги (без символа #)
    lemmatized_hashtags_list = [lemmatize_text(ht).strip() for ht in raw_hashtags if ht.strip()]
    lemmatized_hashtags_filtered = [ht for ht in lemmatized_hashtags_list if ht and ht not in russian_stopwords]
    logger.debug(f"Извлеченные хештеги: {raw_hashtags} -> {lemmatized_hashtags_filtered}")

    # 2. Очистка текста (теперь хештеги # будут удалены функцией clean_text,
    # т.к. мы их уже извлекли и обработали отдельно)
    cleaned_text = clean_text(text)
    logger.debug(f"После очистки: {cleaned_text[:50]}...")

    # 3. Лемматизация основного текста
    lemmatized_text = lemmatize_text(cleaned_text)
    logger.debug(f"После лемматизации: {lemmatized_text[:50]}...")

    # 4. Объединение лемматизированного текста и лемматизированных хештегов
    # Хештеги добавляются в конец, чтобы они не влияли на порядок слов в основном тексте,
    # но при этом были учтены в эмбеддингах.
    combined_text = f"{lemmatized_text} {' '.join(lemmatized_hashtags_filtered)}"
    
    # 5. Удаление стоп-слов из объединенного текста
    final_text = remove_stopwords(combined_text, russian_stopwords)
    logger.debug(f"После удаления стоп-слов и добавления хештегов: {final_text[:50]}...")
    
    return final_text.strip(), raw_hashtags # Возвращаем обработанный текст и оригинальные хештеги


if __name__ == '__main__':
    import pandas as pd # Импортируем здесь для тестового блока

    logger.info("=== Запуск модуля text_preprocessor.py в режиме тестирования ===")

    try:
        from data_loader import load_data
    except ImportError:
        logger.critical("Не удалось импортировать data_loader.py. Убедитесь, что он находится в том же каталоге.")
        sys.exit(1)

    # 1. Загрузка данных
    try:
        df = load_data('messages.csv')
        logger.info(f"Загружено {len(df)} записей из messages.csv.")
        
        if df.empty:
            logger.critical("Загруженный DataFrame пуст. Нечего обрабатывать.")
            sys.exit(1)

    except (FileNotFoundError, ValueError) as e:
        logger.critical(f"Критическая ошибка при загрузке данных: {e}. Программа будет завершена.")
        sys.exit(1)

    # Применяем предобработку к колонке 'text'
    logger.info("Начало предобработки текстов, включая хештеги...")
    # preprocess_text_for_clustering теперь возвращает tuple: (processed_text, hashtags)
    processed_results = df['text'].apply(preprocess_text_for_clustering)
    df['processed_text'] = processed_results.apply(lambda x: x[0])
    df['extracted_hashtags'] = processed_results.apply(lambda x: x[1])

    logger.info("Предобработка текстов завершена.")

    logger.info("\nОригинальные, предобработанные тексты и извлеченные хештеги:")
    for i in range(min(10, len(df))):
        print(f"Оригинал ({df['message_id'].iloc[i]}): {df['text'].iloc[i]}")
        print(f"Обработано      : {df['processed_text'].iloc[i]}")
        print(f"Хештеги         : {df['extracted_hashtags'].iloc[i]}\n")
        
    empty_processed_count = df['processed_text'].apply(lambda x: x.strip() == '').sum()
    if empty_processed_count > 0:
        logger.warning(f"Обнаружено {empty_processed_count} пустых строк после предобработки. "
                       "Эти строки будут отфильтрованы на этапах векторизации/тематического моделирования.")

    logger.info("=== Тестирование модуля text_preprocessor.py завершено ===")