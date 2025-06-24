import logging
import sys
from logging.handlers import RotatingFileHandler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

# Импортируем настроенный логгер
try:
    from text_vectorizer import setup_logger # Или из text_preprocessor, или из utils
except ImportError:
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

# Инициализируем логгер для этого модуля
logger = setup_logger('vectorization_evaluator_module', 'vectorization_evaluator.log')

class VectorizationEvaluator:
    """
    Класс для оценки качества векторизации текстовых данных.
    """
    def __init__(self):
        logger.info("Модуль оценки векторизации инициализирован.")

    def plot_2d_embeddings(self, embeddings, labels=None, title="2D Embeddings Visualization", method="TSNE"):
        """
        Визуализирует многомерные встраивания (embeddings) на 2D-графике.

        Args:
            embeddings (np.ndarray or scipy.sparse.csr_matrix): Векторизованные данные.
            labels (np.ndarray, optional): Метки кластеров для раскраски точек. По умолчанию None.
            title (str): Заголовок графика.
            method (str): Метод уменьшения размерности: "TSNE" или "UMAP".
        """
        if embeddings.shape[0] < 2:
            logger.warning(f"Недостаточно данных ({embeddings.shape[0]} образцов) для 2D-визуализации.")
            return

        logger.info(f"Начало 2D-визуализации с помощью {method}...")

        # Если данные разреженные, t-SNE/UMAP могут работать с ними напрямую,
        # но для небольших наборов данных toarray() может быть проще.
        # Для больших данных лучше избегать toarray().
        data_to_reduce = embeddings
        if hasattr(embeddings, 'toarray'): # Если это разреженная матрица
            # Проверяем размер, чтобы избежать MemoryError
            if embeddings.shape[0] * embeddings.shape[1] > 10**7: # Примерный порог, настройте по необходимости
                logger.warning("Матрица слишком велика для преобразования в плотный массив. TSNE/UMAP будут работать напрямую с разреженной матрицей, что может быть медленнее.")
                pass # Работаем с разреженной матрицей
            else:
                data_to_reduce = embeddings.toarray()
                logger.debug("Разреженная матрица преобразована в плотный массив для TSNE/UMAP.")


        try:
            if method.upper() == "TSNE":
                # Perplexity должна быть меньше, чем количество образцов
                perplexity_val = min(30, max(1, embeddings.shape[0] - 1))
                reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity_val, n_jobs=-1) # n_jobs=-1 использует все ядра
            # elif method.upper() == "UMAP":
            #     reducer = UMAP(n_components=2, random_state=42) # Раскомментируйте, если используете UMAP
            else:
                logger.error(f"Неизвестный метод уменьшения размерности: {method}. Используется TSNE.")
                perplexity_val = min(30, max(1, embeddings.shape[0] - 1))
                reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity_val, n_jobs=-1)
            
            # Применяем уменьшение размерности
            reduced_embeddings = reducer.fit_transform(data_to_reduce)

            plt.figure(figsize=(10, 8))
            if labels is not None:
                # Фильтруем NaN или -1 метки, если они есть
                valid_indices = (labels != -1) & (~pd.isna(labels))
                if np.any(valid_indices):
                    scatter = plt.scatter(reduced_embeddings[valid_indices, 0], reduced_embeddings[valid_indices, 1],
                                          c=labels[valid_indices], cmap='viridis', alpha=0.7)
                    plt.colorbar(scatter, label='Номер кластера')
                else:
                    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.7)
                    logger.warning("Все метки кластеров невалидны (-1 или NaN), точки не раскрашены.")
            else:
                plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.7)

            plt.title(title)
            plt.xlabel(f'{method}-Component 1')
            plt.ylabel(f'{method}-Component 2')
            plt.grid(True)
            plt.show()
            logger.info(f"Визуализация 2D-встраиваний с помощью {method} успешно построена.")

        except Exception as e:
            logger.critical(f"Ошибка при построении 2D-визуализации с помощью {method}: {e}", exc_info=True)

    def evaluate_cosine_similarity(self, vectorizer, texts_to_compare, labels=None):
        """
        Вычисляет и выводит косинусное сходство между парами текстов.
        
        Args:
            vectorizer: Обученный TF-IDF векторизатор (экземпляр TextVectorizer).
            texts_to_compare (list of tuple): Список кортежей, где каждый кортеж содержит:
                                             (индекс_текста_1, индекс_текста_2, "Ожидаемое_отношение")
                                             Пример: [(0, 1, "Похожие"), (0, 5, "Непохожие")]
            labels (list of str, optional): Список оригинальных текстов для вывода.
        """
        if not hasattr(vectorizer.vectorizer, 'vocabulary_'):
            logger.error("Векторизатор не обучен. Невозможно вычислить косинусное сходство.")
            return

        logger.info("Начало вычисления косинусного сходства для заданных пар текстов.")
        
        for idx1, idx2, expected_relation in texts_to_compare:
            try:
                # Преобразуем выбранные тексты в векторы
                # Убедимся, что тексты существуют в исходном списке labels
                if labels and (idx1 >= len(labels) or idx2 >= len(labels)):
                    logger.warning(f"Индексы {idx1} или {idx2} выходят за пределы списка текстов ({len(labels)}). Пропускаем пару.")
                    continue

                text1 = labels[idx1] if labels else f"текст с индексом {idx1}"
                text2 = labels[idx2] if labels else f"текст с индексом {idx2}"

                vec1 = vectorizer.transform([labels[idx1]])
                vec2 = vectorizer.transform([labels[idx2]])

                # Вычисляем косинусное сходство
                similarity = cosine_similarity(vec1, vec2)[0][0]
                
                logger.info(f"Сходство между '{text1[:50]}...' и '{text2[:50]}...': {similarity:.4f} (Ожидается: {expected_relation})")
            except Exception as e:
                logger.error(f"Ошибка при вычислении сходства для пары ({idx1}, {idx2}): {e}", exc_info=True)

    def analyze_top_feature_names(self, vectorizer, num_top_features=20):
        """
        Анализирует и выводит топ-N признаков (слов) из обученного векторизатора.
        Это помогает понять, какие слова векторизатор считает наиболее важными.

        Args:
            vectorizer: Обученный TF-IDF векторизатор (экземпляр TextVectorizer).
            num_top_features (int): Количество топ-признаков для вывода.
        """
        if not hasattr(vectorizer.vectorizer, 'vocabulary_'):
            logger.error("Векторизатор не обучен. Невозможно проанализировать признаки.")
            return

        feature_names = vectorizer.get_feature_names()
        # TF-IDF векторизатор не имеет прямого 'feature_importances_',
        # но мы можем посмотреть на idf_ - инверсную частоту документа
        # Чем выше idf, тем реже слово встречается и тем больше его вес
        
        # Сортируем признаки по их IDF (Inverse Document Frequency) в порядке убывания
        # IDF для каждого признака доступен через vectorizer.vectorizer.idf_
        # Но idf_ имеет тот же порядок, что и vocabulary_
        
        # Получаем словарь vocabulary_ {слово: индекс}
        vocabulary = vectorizer.vectorizer.vocabulary_
        # Создаем список пар (слово, idf)
        word_idf_pairs = []
        for word, idx in vocabulary.items():
            if idx < len(vectorizer.vectorizer.idf_): # Проверка на всякий случай
                word_idf_pairs.append((word, vectorizer.vectorizer.idf_[idx]))
        
        # Сортируем по IDF в убывающем порядке
        word_idf_pairs.sort(key=lambda x: x[1], reverse=True)

        logger.info(f"Топ-{num_top_features} признаков (слов) по их IDF (чем выше, тем реже слово и важнее):")
        for i, (word, idf_value) in enumerate(word_idf_pairs[:num_top_features]):
            logger.info(f"  {i+1}. {word} (IDF: {idf_value:.4f})")

        # Также можно посмотреть на общие статистики
        logger.info(f"\nОбщее количество извлеченных признаков (слов): {len(feature_names)}")
        if feature_names:
            logger.info(f"Примеры первых 10 признаков: {feature_names[:10]}")
            logger.info(f"Примеры последних 10 признаков: {feature_names[-10:]}")


# --- Пример использования модуля ---
if __name__ == '__main__':
    logger.info("=== Запуск модуля vectorization_evaluator.py в режиме тестирования ===")

    try:
        from data_loader import load_data
        from text_preprocessor import preprocess_text_for_clustering
        from text_vectorizer import TextVectorizer
        from text_clusterer import TextClusterer # Для получения кластеров для визуализации
    except ImportError as e:
        logger.critical(f"Не удалось импортировать один из необходимых модулей: {e}. "
                        "Убедитесь, что все модули находятся в том же каталоге.")
        sys.exit(1)

    # 1. Загрузка данных
    try:
        df = load_data('messages.csv')
        logger.info(f"Загружено {len(df)} записей из messages.csv.")
    except FileNotFoundError:
        logger.warning("Файл 'messages.csv' не найден. Создаем тестовые данные для демонстрации.")
        
        df = pd.DataFrame(sample_data)
        logger.info(f"Создано {len(df)} тестовых записей.")

    # 2. Предобработка текста
    logger.info("Начало предобработки текстов...")
    df['processed_text'] = df['text'].apply(preprocess_text_for_clustering)
    texts_to_vectorize = df['processed_text'].tolist()
    
    original_len = len(texts_to_vectorize)
    # df_filtered_for_vec = df[df['processed_text'].apply(lambda x: x.strip() != '')].copy()
    # texts_to_vectorize = df_filtered_for_vec['processed_text'].tolist()
    # current_indices = df_filtered_for_vec.index.tolist() # Сохраняем индексы текстов, которые были векторизованы
    
    valid_texts_data = []
    valid_original_indices = []
    for idx, text in enumerate(df['processed_text'].tolist()):
        if isinstance(text, str) and text.strip():
            valid_texts_data.append(text)
            valid_original_indices.append(idx)
    
    texts_to_vectorize = valid_texts_data
    
    if len(texts_to_vectorize) < original_len:
        logger.warning(f"Удалено {original_len - len(texts_to_vectorize)} пустых текстов после предобработки.")

    if not texts_to_vectorize:
        logger.critical("Все тексты оказались пустыми после предобработки. Оценка векторизации невозможна.")
        sys.exit(1)

    logger.info("Предобработка текстов завершена.")

    # 3. Векторизация текста
    logger.info("Начало векторизации текстов...")
    vectorizer_obj = TextVectorizer(max_features=1500, min_df=2, max_df=0.7)
    tfidf_matrix = vectorizer_obj.fit_transform(texts_to_vectorize)
    logger.info(f"Векторизация текстов завершена. Размер матрицы: {tfidf_matrix.shape}")

    if tfidf_matrix.shape[0] == 0:
        logger.critical("TF-IDF матрица пуста после векторизации. Оценка невозможна.")
        sys.exit(1)

    # 4. Кластеризация (для получения меток кластеров для визуализации)
    logger.info("Начало кластеризации для получения меток...")
    # Здесь мы используем оптимальное K, которое вы могли бы найти ранее.
    # Для демонстрации возьмем, например, 5 или 11 кластеров.
    num_clusters_for_eval = min(5, tfidf_matrix.shape[0] if tfidf_matrix.shape[0] > 0 else 1)
    if tfidf_matrix.shape[0] >= 2 and num_clusters_for_eval >= 2:
        clusterer = TextClusterer(n_clusters=num_clusters_for_eval, random_state=42)
        clusterer.fit(tfidf_matrix)
        cluster_labels_for_eval = clusterer.predict(tfidf_matrix)
        logger.info(f"Кластеризация выполнена с {num_clusters_for_eval} кластерами.")
    else:
        logger.warning("Недостаточно данных для кластеризации. Метки кластеров для визуализации будут отсутствовать.")
        cluster_labels_for_eval = None # Или np.zeros(tfidf_matrix.shape[0])


    # 5. Оценка векторизации
    evaluator = VectorizationEvaluator()

    # 5.1. Визуализация 2D-встраиваний
    logger.info("\nЗапуск визуализации 2D-встраиваний (t-SNE)...")
    # Передаем только те тексты, которые были векторизованы
    evaluator.plot_2d_embeddings(tfidf_matrix, labels=cluster_labels_for_eval, 
                                 title="Визуализация TF-IDF векторов (t-SNE)", method="TSNE")
    
    
    # 5.2. Оценка косинусного сходства
    logger.info("\nЗапуск оценки косинусного сходства для выбранных пар текстов...")
    # Примеры пар текстов для сравнения
    # Используем индексы из df, так как evaluator.evaluate_cosine_similarity ожидает оригинальные тексты
    
    evaluator.evaluate_cosine_similarity(vectorizer_obj, texts_for_similarity_check, labels=df['text'].tolist())

    # 5.3. Анализ топ-признаков векторизатора
    logger.info("\nАнализ топ-признаков (слов) векторизатора...")
    evaluator.analyze_top_feature_names(vectorizer_obj, num_top_features=30)

    logger.info("=== Тестирование модуля vectorization_evaluator.py завершено ===")