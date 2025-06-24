import logging
import sys
from logging.handlers import RotatingFileHandler
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Импортируем модуль предобработки текста
try:
    from text_preprocessor import preprocess_text_for_clustering, setup_logger
except ImportError:
    # Запасной вариант, если text_preprocessor не найден или setup_logger не доступен
    def setup_logger(name, log_file, level=logging.INFO):
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
    
    logger_fallback = setup_logger('text_vectorizer_module_fallback', 'text_vectorizer.log')
    logger_fallback.error("Не удалось импортировать 'text_preprocessor.py'. Убедитесь, что он находится в том же каталоге.")
    sys.exit(1) # Выходим, если не можем импортировать ключевой модуль

# Инициализируем логгер для этого модуля
logger = setup_logger('text_vectorizer_module', 'text_vectorizer.log')

class TextVectorizer:
    """
    Класс для векторизации текстовых данных с использованием TF-IDF.
    """
    def __init__(self, max_features=5000, min_df=5, max_df=0.8):
        """
        Инициализирует векторизатор TF-IDF.

        Args:
            max_features (int): Максимальное количество признаков (слов) для TF-IDF.
                                 Ограничивает размерность матрицы.
            min_df (int or float): Когда строить словарь, игнорировать термины,
                                   которые имеют частоту документа (document frequency)
                                   ниже заданного порога.
                                   Если int, то это абсолютное число документов.
                                   Если float, то это доля документов.
            max_df (int or float): Когда строить словарь, игнорировать термины,
                                   которые имеют частоту документа (document frequency)
                                   выше заданного порога.
                                   Если int, то это абсолютное число документов.
                                   Если float, то это доля документов.
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            norm='l2',          # Нормализация L2 (по умолчанию)
            use_idf=True,       # Использовать IDF (по умолчанию)
            smooth_idf=True,    # Добавить 1 к частотам документов для сглаживания (по умолчанию)
            sublinear_tf=False, # Применить сублинейное масштабирование TF (1 + log(tf))
            # analyzer='word' (по умолчанию),
            # tokenizer=None (по умолчанию, т.к. текст уже токенизирован и объединен)
        )
        logger.info(f"TF-IDF векторизатор инициализирован с параметрами: "
                    f"max_features={max_features}, min_df={min_df}, max_df={max_df}")

    def fit_transform(self, texts):
        """
        Обучает векторизатор на текстах и преобразует их в TF-IDF матрицу.

        Args:
            texts (list of str): Список предобработанных текстовых строк.

        Returns:
            scipy.sparse.csr_matrix: TF-IDF матрица.
        """
        if not texts:
            logger.warning("Получен пустой список текстов для fit_transform. Возвращаем пустую матрицу.")
            return pd.DataFrame().sparse.to_coo() # Возвращаем пустую разреженную матрицу

        logger.info(f"Начало обучения и преобразования TF-IDF для {len(texts)} текстов.")
        try:
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            logger.info(f"TF-IDF векторизация завершена. Получена матрица размером: {tfidf_matrix.shape}")
            # logger.debug(f"Пример первого вектора TF-IDF: {tfidf_matrix[0].toarray()}")
            return tfidf_matrix
        except Exception as e:
            logger.critical(f"Ошибка при обучении/преобразовании TF-IDF: {e}", exc_info=True)
            raise

    def transform(self, texts):
        """
        Преобразует новые тексты в TF-IDF матрицу, используя уже обученный векторизатор.

        Args:
            texts (list of str): Список предобработанных текстовых строк.

        Returns:
            scipy.sparse.csr_matrix: TF-IDF матрица.
        """
        if not hasattr(self.vectorizer, 'vocabulary_'):
            logger.error("Векторизатор TF-IDF не обучен. Вызовите fit_transform сначала.")
            raise RuntimeError("Векторизатор TF-IDF не обучен.")

        if not texts:
            logger.warning("Получен пустой список текстов для transform. Возвращаем пустую матрицу.")
            return pd.DataFrame().sparse.to_coo()

        logger.info(f"Начало преобразования TF-IDF для {len(texts)} новых текстов.")
        try:
            tfidf_matrix = self.vectorizer.transform(texts)
            logger.info(f"TF-IDF преобразование завершено. Получена матрица размером: {tfidf_matrix.shape}")
            return tfidf_matrix
        except Exception as e:
            logger.critical(f"Ошибка при преобразовании новых текстов TF-IDF: {e}", exc_info=True)
            raise

    def get_feature_names(self):
        """
        Возвращает названия признаков (слов) из обученного векторизатора.
        """
        if hasattr(self.vectorizer, 'get_feature_names_out'):
            return self.vectorizer.get_feature_names_out()
        elif hasattr(self.vectorizer, 'get_feature_names'): # Для старых версий scikit-learn
            return self.vectorizer.get_feature_names()
        else:
            logger.warning("Метод get_feature_names_out или get_feature_names не найден в векторизаторе.")
            return []

# --- Пример использования модуля ---
if __name__ == '__main__':
    logger.info("=== Запуск модуля text_vectorizer.py в режиме тестирования ===")


    # Инициализация векторизатора
    vectorizer_obj = TextVectorizer(max_features=1000, min_df=1, max_df=0.9)

    # Обучение и преобразование текстов
    tfidf_matrix = vectorizer_obj.fit_transform(sample_processed_texts)

    logger.info("\nРазмер полученной TF-IDF матрицы:")
    print(tfidf_matrix.shape) # (количество документов, количество уникальных слов)

    logger.info("\nПризнаки (слова), извлеченные TF-IDF векторизатором:")
    feature_names = vectorizer_obj.get_feature_names()
    print(feature_names[:10]) # Выводим первые 10 признаков

    logger.info("\nTF-IDF вектор для первого текста (пример):")
    # Чтобы увидеть не нулевые значения, можно преобразовать в массив и найти индексы
    # tfidf_vector_0 = tfidf_matrix[0].toarray().flatten()
    # print([ (feature_names[i], tfidf_vector_0[i]) for i in tfidf_vector_0.nonzero()[0] ])

    # Или просто посмотреть на разреженную форму (чаще всего так и работают)
    print(tfidf_matrix[0]) # разреженное представление

    logger.info("=== Тестирование модуля text_vectorizer.py завершено ===")