import logging
import sys
from logging.handlers import RotatingFileHandler
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.exceptions import ConvergenceWarning
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Отключаем предупреждения ConvergenceWarning от KMeans, чтобы они не засоряли логи
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Импортируем настроенный логгер 
try:
    # Пытаемся импортировать setup_logger из text_vectorizer или text_preprocessor
    # В реальном проекте, setup_logger лучше вынести в отдельный utils.py
    from text_vectorizer import setup_logger 
except ImportError:
    # Запасной вариант, если setup_logger не найден
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
logger = setup_logger('text_clusterer_module', 'text_clusterer.log')

class TextClusterer:
    """
    Класс для кластеризации текстовых данных с использованием K-Means,
    с возможностью подбора оптимального количества кластеров.
    """
    def __init__(self, n_clusters=5, random_state=42):
        """
        Инициализирует кластеризатор K-Means.

        Args:
            n_clusters (int): Количество кластеров, на которые будут разделены данные.
                              Может быть переопределено методом find_optimal_clusters.
            random_state (int): Зерно для генератора случайных чисел для воспроизводимости.
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans_model = None
        self.inertia_values = {}  # Для метода "Локтя"
        self.silhouette_scores = {} # Для Silhouette Score
        logger.info(f"K-Means кластеризатор инициализирован с параметрами: "
                    f"n_clusters={n_clusters}, random_state={random_state}")

    def fit(self, vectorized_data):
        """
        Обучает K-Means модель на векторизованных данных.

        Args:
            vectorized_data (scipy.sparse.csr_matrix or np.ndarray): Векторизованные текстовые данные.
        """
        if vectorized_data.shape[0] == 0:
            logger.warning("Получены пустые векторизованные данные для обучения. Обучение пропущено.")
            return

        # Если количество образцов меньше желаемого количества кластеров, корректируем n_clusters
        if vectorized_data.shape[0] < self.n_clusters:
            logger.warning(f"Количество образцов ({vectorized_data.shape[0]}) меньше, чем желаемое количество кластеров ({self.n_clusters}). "
                           f"Устанавливаем n_clusters равным количеству образцов (или 1, если 0).")
            self.n_clusters = max(1, vectorized_data.shape[0])
            if self.n_clusters == 1: # K-Means с 1 кластером не всегда полезен, но предотвращает ошибки
                 logger.info("Установлено n_clusters=1, так как недостаточно образцов для более чем одного кластера.")

        if self.n_clusters < 1:
            logger.warning("Невозможно обучить модель с n_clusters < 1. Обучение пропущено.")
            return

        logger.info(f"Начало обучения K-Means модели на {vectorized_data.shape[0]} образцах с {self.n_clusters} кластерами.")
        try:
            self.kmeans_model = KMeans(
                n_clusters=self.n_clusters,
                init='k-means++',
                max_iter=300,
                n_init=10,
                random_state=self.random_state,
            )
            self.kmeans_model.fit(vectorized_data)
            logger.info("K-Means модель успешно обучена.")
        except Exception as e:
            logger.critical(f"Ошибка при обучении K-Means модели: {e}", exc_info=True)
            raise

    def predict(self, vectorized_data):
        """
        Предсказывает кластеры для векторизованных данных.

        Args:
            vectorized_data (scipy.sparse.csr_matrix or np.ndarray): Векторизованные текстовые данные.

        Returns:
            np.ndarray: Массив меток кластеров для каждого образца.
        """
        if self.kmeans_model is None:
            logger.error("K-Means модель не обучена. Вызовите метод fit() сначала.")
            raise RuntimeError("Модель кластеризации не обучена.")
        
        if vectorized_data.shape[0] == 0:
            logger.warning("Получены пустые векторизованные данные для предсказания. Возвращаем пустой массив.")
            return np.array([])

        logger.info(f"Начало предсказания кластеров для {vectorized_data.shape[0]} образцов.")
        try:
            cluster_labels = self.kmeans_model.predict(vectorized_data)
            logger.info("Предсказание кластеров завершено.")
            return cluster_labels
        except Exception as e:
            logger.critical(f"Ошибка при предсказании кластеров: {e}", exc_info=True)
            raise

    def evaluate_clusters(self, vectorized_data, cluster_labels):
        """
        Оценивает качество кластеризации с использованием метрики Silhouette Score.

        Args:
            vectorized_data (scipy.sparse.csr_matrix or np.ndarray): Векторизованные текстовые данные.
            cluster_labels (np.ndarray): Метки кластеров, предсказанные моделью.

        Returns:
            float: Значение Silhouette Score. Чем ближе к 1, тем лучше.
        """
        # Уникальные метки кластеров
        unique_labels = np.unique(cluster_labels)
        
        # Для расчета Silhouette Score требуется как минимум 2 кластера и количество образцов > 1
        if len(unique_labels) < 2 or vectorized_data.shape[0] < 2:
            logger.warning("Недостаточно образцов или кластеров для расчета Silhouette Score.")
            return -1.0 # Или np.nan

        logger.info("Начало расчета Silhouette Score.")
        try:
            score = silhouette_score(vectorized_data, cluster_labels)
            logger.info(f"Silhouette Score: {score:.3f}")
            return score
        except Exception as e:
            logger.critical(f"Ошибка при расчете Silhouette Score: {e}", exc_info=True)
            return -1.0 # Возвращаем -1 в случае ошибки

    def find_optimal_clusters(self, vectorized_data, max_clusters=10, plot_results=True):
        """
        Находит оптимальное количество кластеров, используя методы "локтя" и Silhouette Score.

        Args:
            vectorized_data (scipy.sparse.csr_matrix or np.ndarray): Векторизованные текстовые данные.
            max_clusters (int): Максимальное количество кластеров для тестирования.
            plot_results (bool): Если True, построит графики для визуализации результатов.

        Returns:
            int: Рекомендуемое количество кластеров (на основе Silhouette Score).
        """
        if vectorized_data.shape[0] < 2:
            logger.warning("Недостаточно данных для поиска оптимального количества кластеров (менее 2 образцов).")
            return 1 # Или другое значение по умолчанию

        # Определяем верхнюю границу для тестирования кластеров
        # Количество кластеров не может быть больше, чем количество образцов - 1
        # Также должно быть не менее 2 кластеров для Silhouette Score
        upper_bound_clusters = min(max_clusters, vectorized_data.shape[0] -1)
        if upper_bound_clusters < 2:
            logger.warning(f"Недостаточно уникальных образцов ({vectorized_data.shape[0]}) для тестирования более чем 1 кластера. "
                           f"Возвращаем {max(1, vectorized_data.shape[0])} как оптимальное.")
            return max(1, vectorized_data.shape[0])


        logger.info(f"Начало поиска оптимального количества кластеров (от 2 до {upper_bound_clusters}).")
        
        inertia = []
        silhouette_scores = []
        possible_n_clusters = range(2, upper_bound_clusters + 1) # Начинаем с 2, т.к. K-Means требует >=1 кластер

        for i in possible_n_clusters:
            logger.debug(f"Тестирование K-Means с n_clusters={i}...")
            try:
                # Временно создаем новый KMeans для каждого n_clusters, чтобы не влиять на основной self.kmeans_model
                kmeans = KMeans(
                    n_clusters=i,
                    init='k-means++',
                    max_iter=300,
                    n_init=10,
                    random_state=self.random_state
                )
                kmeans.fit(vectorized_data)
                inertia.append(kmeans.inertia_)
                self.inertia_values[i] = kmeans.inertia_

                # Вычисляем Silhouette Score
                cluster_labels = kmeans.labels_
                if len(np.unique(cluster_labels)) > 1: # Проверка на наличие хотя бы 2 уникальных кластеров
                    score = silhouette_score(vectorized_data, cluster_labels)
                    silhouette_scores.append(score)
                    self.silhouette_scores[i] = score
                else:
                    silhouette_scores.append(-1.0) # Если только один кластер, score не определен
                    self.silhouette_scores[i] = -1.0
            except Exception as e:
                logger.error(f"Ошибка при тестировании n_clusters={i}: {e}", exc_info=True)
                inertia.append(np.nan)
                silhouette_scores.append(np.nan)

        if not inertia or all(np.isnan(inertia)):
            logger.error("Не удалось вычислить инерцию для любого количества кластеров.")
            return self.n_clusters # Возвращаем дефолтное значение

        if plot_results:
            self._plot_elbow_and_silhouette(possible_n_clusters, inertia, silhouette_scores)

        # Выбираем оптимальное количество кластеров на основе Silhouette Score
        # Ищем кластер с максимальным скором, игнорируя -1.0 (для случаев с 1 кластером или ошибками)
        if not silhouette_scores or all(s <= 0 for s in silhouette_scores):
            logger.warning("Silhouette Score не смог определить оптимальное количество кластеров. Возможно, кластеры плохо разделены или данных недостаточно. "
                           "Рекомендуется проанализировать график метода 'локтя'. Возвращаем дефолтное количество кластеров.")
            return self.n_clusters

        # Находим индекс максимального Silhouette Score
        # Прибавляем 2, т.к. range начинается с 2
        optimal_n_clusters_silhouette = possible_n_clusters[np.argmax(silhouette_scores)]
        logger.info(f"Рекомендуемое количество кластеров по Silhouette Score: {optimal_n_clusters_silhouette}")
        
        # Можно также попытаться найти "локоть" (более сложная эвристика)
        # Это требует более сложного анализа или сторонних библиотек,
        # поэтому пока полагаемся на Silhouette Score.

        self.n_clusters = optimal_n_clusters_silhouette # Обновляем n_clusters для текущего объекта
        return optimal_n_clusters_silhouette

    def _plot_elbow_and_silhouette(self, n_clusters_range, inertia, silhouette_scores):
        """
        Строит графики метода "локтя" и Silhouette Score.
        """
        plt.figure(figsize=(14, 6))

        # График метода "Локтя"
        plt.subplot(1, 2, 1)
        plt.plot(n_clusters_range, inertia, marker='o')
        plt.title('Метод "Локтя" (Elbow Method)')
        plt.xlabel('Количество кластеров')
        plt.ylabel('Инерция (Sum of Squared Distances)')
        plt.xticks(n_clusters_range)
        plt.grid(True)

        # График Silhouette Score
        plt.subplot(1, 2, 2)
        # Фильтруем nan, чтобы не ломался график, но сохраняем соответствие осей
        valid_indices = [i for i, score in enumerate(silhouette_scores) if not np.isnan(score)]
        valid_n_clusters = [n_clusters_range[i] for i in valid_indices]
        valid_scores = [silhouette_scores[i] for i in valid_indices]

        plt.plot(valid_n_clusters, valid_scores, marker='o', color='red')
        plt.title('Silhouette Score')
        plt.xlabel('Количество кластеров')
        plt.ylabel('Silhouette Score')
        plt.xticks(n_clusters_range)
        plt.grid(True)

        plt.tight_layout()
        plt.show()
        logger.info("Графики метода 'Локтя' и Silhouette Score построены.")

#  Пример использования модуля 
if __name__ == '__main__':
    logger.info("=== Запуск модуля text_clusterer.py в режиме тестирования с подбором N кластеров ===")

    # Для демонстрации, импортируем предыдущие модули
    try:
        from data_loader import load_data
        from text_preprocessor import preprocess_text_for_clustering
        from text_vectorizer import TextVectorizer
    except ImportError as e:
        logger.critical(f"Не удалось импортировать один из необходимых модулей: {e}. "
                        "Убедитесь, что data_loader.py, text_preprocessor.py и text_vectorizer.py находятся в том же каталоге.")
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
    # Удаляем пустые строки, которые могут возникнуть после предобработки
    original_len = len(texts_to_vectorize)
    texts_to_vectorize = [text for text in texts_to_vectorize if text.strip()]
    if len(texts_to_vectorize) < original_len:
        logger.warning(f"Удалено {original_len - len(texts_to_vectorize)} пустых текстов после предобработки.")

    if not texts_to_vectorize:
        logger.critical("Все тексты оказались пустыми после предобработки. Кластеризация невозможна.")
        sys.exit(1)


    # 3. Векторизация текста
    logger.info("Начало векторизации текстов...")
    vectorizer_obj = TextVectorizer(max_features=1500, min_df=2, max_df=0.7) # Немного меняем параметры для разнообразия
    tfidf_matrix = vectorizer_obj.fit_transform(texts_to_vectorize)
    logger.info(f"Векторизация текстов завершена. Размер матрицы: {tfidf_matrix.shape}")

    # Важно: если матрица пуста или имеет 0 строк, кластеризация невозможна
    if tfidf_matrix.shape[0] == 0:
        logger.critical("TF-IDF матрица пуста после векторизации. Кластеризация невозможна.")
        sys.exit(1)


    # 4. Подбор оптимального количества кластеров
    logger.info("Начало подбора оптимального количества кластеров...")
    # Ограничиваем максимальное количество кластеров, чтобы не было ошибок при малом количестве данных
    max_possible_clusters = min(15, tfidf_matrix.shape[0] -1) # Не больше 15 и не больше чем документов - 1
    if max_possible_clusters < 2:
        logger.warning(f"Недостаточно данных для поиска оптимального количества кластеров (max_possible_clusters={max_possible_clusters}). "
                       "Кластеризация будет выполнена с n_clusters=1 (если данных достаточно).")
        optimal_k = 1 if tfidf_matrix.shape[0] > 0 else 0
    else:
        # Инициализируем кластеризатор с временным значением n_clusters
        clusterer = TextClusterer(n_clusters=5, random_state=42) # n_clusters здесь - начальное/запасное
        optimal_k = clusterer.find_optimal_clusters(tfidf_matrix, max_clusters=max_possible_clusters, plot_results=True)
        logger.info(f"Оптимальное количество кластеров, рекомендованное системой: {optimal_k}")

    # 5. Кластеризация с оптимальным количеством кластеров
    if optimal_k > 0:
        logger.info(f"Начало финальной кластеризации текстов с {optimal_k} кластерами...")
        final_clusterer = TextClusterer(n_clusters=optimal_k, random_state=42)
        final_clusterer.fit(tfidf_matrix)
        cluster_labels = final_clusterer.predict(tfidf_matrix)

        # Добавляем метки кластеров к DataFrame
        # Важно: метки кластеров относятся только к непустым текстам, которые были векторизованы
        df_processed = df[df['processed_text'].apply(lambda x: x.strip() != '')].copy()
        df_processed['cluster'] = cluster_labels

        # Если оригинальный DataFrame содержит пустые строки, которые были отфильтрованы,
        # нужно корректно присвоить им NaN или -1.
        df['cluster'] = np.nan
        df.loc[df_processed.index, 'cluster'] = df_processed['cluster']
        df['cluster'] = df['cluster'].fillna(-1).astype(int) # Заполняем NaN, например, -1

        logger.info("Финальная кластеризация завершена.")

        # Оценка качества кластеризации (если возможно)
        if optimal_k > 1 and tfidf_matrix.shape[0] > 1: # Проверка на достаточное количество кластеров и данных
            silhouette_avg = final_clusterer.evaluate_clusters(tfidf_matrix, cluster_labels)
            logger.info(f"Финальный средний Silhouette Score для кластеров: {silhouette_avg:.3f}")
        else:
            logger.warning("Невозможно рассчитать Silhouette Score для финальной кластеризации (менее 2 кластеров или 2 образцов).")

        logger.info("\nDataFrame с назначенными кластерами (первые 15 записей):")
        print(df[['message_id', 'text', 'processed_text', 'cluster']].head(15))

        # Выводим содержимое каждого кластера
        logger.info("\nПримеры текстов из каждого кластера:")
        for i in range(optimal_k):
            cluster_texts_df = df[df['cluster'] == i]
            if not cluster_texts_df.empty:
                logger.info(f"\n Кластер {i} ({len(cluster_texts_df)} документов) ")
                for j, row in enumerate(cluster_texts_df.head(5).itertuples()): # Выводим до 5 примеров
                    logger.info(f"  {j+1}. {row.text[:100]}...")
                if len(cluster_texts_df) > 5:
                    logger.info("  ...(и еще)...")
            else:
                logger.info(f"\n Кластер {i} (пуст) ")


    else:
        logger.warning("Оптимальное количество кластеров не найдено или равно 0/1. Кластеризация не выполнялась.")
        df['cluster'] = -1 # Присваиваем -1, если кластеризация невозможна

    logger.info("=== Тестирование модуля text_clusterer.py завершено ===")