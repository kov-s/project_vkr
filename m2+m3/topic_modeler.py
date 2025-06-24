import sys
import logging
from logging.handlers import RotatingFileHandler
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import plotly.graph_objects as go # Для сохранения Plotly графиков

#  Настройка логирования 

try:
    from data_loader import setup_logger
except ImportError:
    def setup_logger(name, log_file='topic_modeler.log', level=logging.INFO):
        logger = logging.getLogger(name)
        logger.setLevel(level)
        if not logger.handlers: # Избегаем добавления нескольких обработчиков при повторном импорте
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            
            # Файловый обработчик с ротацией
            file_handler = RotatingFileHandler(log_file, maxBytes=1024*1024*5, backupCount=5, encoding='utf-8') # 5MB
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            # Консольный обработчик
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        return logger

logger = setup_logger('topic_modeler_module', 'topic_modeler.log')

class TopicModeler:
    """
    Класс для выполнения тематического моделирования с использованием BERTopic.
    """
    def __init__(self, embedding_model_name='paraphrase-multilingual-MiniLM-L12-v2', # Рекомендуемая мультиязычная модель
                 nr_topics="auto", # Автоматическое определение количества тем
                 min_topic_size=10, # Минимальное количество документов для формирования темы
                 n_gram_range=(1, 2) # Использование униграм и биграмм для извлечения ключевых слов
                ):
        """
        Инициализирует TopicModeler.
        :param embedding_model_name: Название модели SentenceTransformer для эмбеддингов.
                                     Примеры для русского: 'paraphrase-multilingual-MiniLM-L12-v2',
                                     'cointegrated/rubert-tiny2' (более тяжелая, но точная).
        :param nr_topics: Количество тем. 'auto' для автоматического определения HDBSCAN,
                          или целое число для фиксированного количества.
        :param min_topic_size: Минимальное количество документов, необходимых для формирования кластера.
        :param n_gram_range: Диапазон n-грамм для извлечения ключевых слов тем.
        """
        logger.info(f"Инициализация TopicModeler с моделью эмбеддингов: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        logger.info("Модель SentenceTransformer успешно загружена.")

        self.topic_model = BERTopic(
            embedding_model=self.embedding_model,
            nr_topics=nr_topics,
            min_topic_size=min_topic_size,
            n_gram_range=n_gram_range,
            verbose=True # Включаем подробный вывод от BERTopic
        )
        logger.info("BERTopic модель инициализирована.")
        self.topics = None
        self.probabilities = None

    def fit_transform(self, documents):
        """
        Обучает модель BERTopic на документах и присваивает им темы.
        :param documents: Список предобработанных текстовых документов.
        :return: Кортеж из (список присвоенных тем, список вероятностей).
        """
        logger.info(f"Начало обучения BERTopic на {len(documents)} документах...")
        self.topics, self.probabilities = self.topic_model.fit_transform(documents)
        logger.info("Обучение BERTopic завершено.")
        return self.topics, self.probabilities

    def get_topic_info(self):
        """
        Возвращает DataFrame с информацией о темах.
        """
        if self.topic_model:
            return self.topic_model.get_topic_info()
        logger.warning("Модель BERTopic не инициализирована или не обучена.")
        return pd.DataFrame()

    def get_intertopic_distance_map(self):
        """
        Генерирует и возвращает интерактивную карту расстояний между темами.
        """
        if self.topic_model and self.topics is not None:
            logger.info("Генерация карты межабстрактных расстояний...")
            try:
                fig = self.topic_model.visualize_topics()
                logger.info("Карта межабстрактных расстояний успешно сгенерирована.")
                return fig
            except Exception as e:
                logger.error(f"Ошибка при генерации карты межабстрактных расстояний: {e}", exc_info=True)
                return None
        logger.warning("Модель BERTopic не обучена для визуализации расстояний между темами.")
        return None

    def get_topic_word_barchart(self, topic=None, top_n_topics=None):
        """
        Генерирует и возвращает интерактивный барчарт слов для тем.
        :param topic: Конкретный номер темы для визуализации (например, 0, 1, 2...).
                      Если None, будет сгенерирован для top_n_topics.
        :param top_n_topics: Количество лучших тем для визуализации, если topic=None.
        """
        if self.topic_model and self.topics is not None:
            logger.info(f"Генерация барчарта слов для тем (topic={topic}, top_n_topics={top_n_topics})...")
            try:
                if topic is not None:
                    fig = self.topic_model.visualize_barchart(topics=topic)
                elif top_n_topics is not None:
                    # Получаем ID лучших тем, исключая Topic -1
                    topic_ids = self.topic_model.get_topic_info().loc[self.topic_model.get_topic_info().Topic != -1, 'Topic'].head(top_n_topics).tolist()
                    fig = self.topic_model.visualize_barchart(topics=topic_ids)
                else:
                    logger.warning("Необходимо указать 'topic' или 'top_n_topics' для визуализации барчарта.")
                    return None
                logger.info("Барчарт слов для тем успешно сгенерирован.")
                return fig
            except Exception as e:
                logger.error(f"Ошибка при генерации барчарта слов для тем: {e}", exc_info=True)
                return None
        logger.warning("Модель BERTopic не обучена для визуализации барчартов слов.")
        return None

    # Дополнительные визуализации, если нужны (например, visualize_hierarchy, visualize_heatmap)
    # def visualize_hierarchy(self):
    #     if self.topic_model and self.topics is not None:
    #         fig = self.topic_model.visualize_hierarchy()
    #         return fig
    #     return None


# запуск модуля 
if __name__ == '__main__':
    logger.info("=== Запуск модуля topic_modeler.py в режиме тестирования ===")

    # Импорт зависимостей
    try:
        from data_loader import load_data
        from text_preprocessor import preprocess_text_for_clustering
    except ImportError as e:
        logger.critical(f"Ошибка импорта зависимостей: {e}. Убедитесь, что data_loader.py "
                        "и text_preprocessor.py находятся в том же каталоге и доступны.")
        sys.exit(1)

    # 1. Загрузка данных
    try:
        logger.info("Загрузка данных из 'messages.csv'...")
        df = load_data('messages.csv')
        
        if df.empty:
            logger.critical("Загруженный DataFrame пуст. Нечего обрабатывать.")
            sys.exit(1)
        logger.info(f"Загружено {len(df)} записей из messages.csv.")

    except Exception as e:
        logger.critical(f"Критическая ошибка при загрузке данных: {e}. Программа будет завершена.")
        sys.exit(1)

    # 2. Предобработка текстов
    logger.info("Начало предобработки текстов с помощью text_preprocessor.py...")
  
    try:
        from tqdm.auto import tqdm
        tqdm.pandas() # Включаем pandas integration для tqdm
        processed_results = df['text'].progress_apply(preprocess_text_for_clustering)
    except ImportError:
        logger.warning("tqdm не найдена, предобработка без индикатора прогресса.")
        processed_results = df['text'].apply(preprocess_text_for_clustering)

    df['processed_text'] = processed_results.apply(lambda x: x[0])
    df['extracted_hashtags'] = processed_results.apply(lambda x: x[1])
    
    # 3. Фильтрация пустых строк после предобработки
    initial_rows = len(df)
    df_filtered = df[df['processed_text'].str.strip() != ''].copy()
    rows_removed = initial_rows - len(df_filtered)
    
    if rows_removed > 0:
        logger.warning(f"Удалено {rows_removed} сообщений, ставших пустыми после предобработки.")
    
    if df_filtered.empty:
        logger.critical("После предобработки не осталось валидных сообщений для тематического моделирования. Программа будет завершена.")
        sys.exit(1)

    logger.info(f"Предобработка текстов завершена. Осталось {len(df_filtered)} валидных сообщений для анализа.")
    
    documents_for_bertopic = df_filtered['processed_text'].tolist()
    ####   временно 
    logger.info(f"Первые 5 предобработанных документов: {documents_for_bertopic[:5]}")
    logger.info(f"Количество документов для BERTopic: {len(documents_for_bertopic)}")
    # Проверяем, есть ли пустые строки после предобработки, которые могли пройти фильтрацию
    empty_docs_after_preprocessing = [doc for doc in documents_for_bertopic if not doc.strip()]
    logger.info(f"Количество пустых строк в processed_text: {len(empty_docs_after_preprocessing)}")
    if len(empty_docs_after_preprocessing) > 0:
        logger.warning(f"Присутствуют пустые строки в processed_text. Пример: {empty_docs_after_preprocessing[:3]}")
    # Сохраняем ID и оригинальные тексты для связывания результатов
    document_ids = df_filtered['message_id'].tolist() 
    original_texts = df_filtered['text'].tolist()


    # 4. Инициализация и обучение TopicModeler
    # Рекомендуется 'paraphrase-multilingual-MiniLM-L12-v2' для начала,
    # или 'cointegrated/rubert-tiny2' для более высокого качества на русском (но медленнее)
    current_embedding_model = 'paraphrase-multilingual-MiniLM-L12-v2' 
    topic_modeler = TopicModeler(
        embedding_model_name=current_embedding_model,
        nr_topics="auto",
        min_topic_size=5, 
        n_gram_range=(1, 2)
    )
    
    topics, probabilities = topic_modeler.fit_transform(documents_for_bertopic)
    
    num_unique_topics = len(topic_modeler.get_topic_info()) - 1 # Исключаем Topic -1
    logger.info(f"Модель BERTopic обучена. Найдено {num_unique_topics} уникальных тем.")
    
    if num_unique_topics <= 0:
        logger.warning("Не найдено ни одной уникальной темы (кроме выбросов). Возможно, данных недостаточно или min_topic_size слишком большой.")

    # 5. Анализ и сохранение результатов
    logger.info("Генерация информации о темах...")
    topic_info_df = topic_modeler.get_topic_info()
    logger.info("\nИнформация о темах:\n%s", topic_info_df.to_string())

    # Сохранение информации о темах в CSV
    try:
        topic_info_df.to_csv("topic_info.csv", index=False, encoding='utf-8')
        logger.info("Информация о темах сохранена в topic_info.csv")
    except Exception as e:
        logger.error(f"Ошибка при сохранении topic_info.csv: {e}", exc_info=True)


    # Присвоение предсказанных тем обратно в DataFrame
    df_filtered['topic'] = topics
    df_filtered['topic_probability'] = probabilities

    # Сохранение DataFrame с сообщениями, их предобработанными версиями, хештегами и темами
    try:
        df_filtered[['message_id', 'text', 'processed_text', 'extracted_hashtags', 'topic', 'topic_probability']].to_csv("messages_with_topics.csv", index=False, encoding='utf-8')
        logger.info("Сообщения с присвоенными темами сохранены в messages_with_topics.csv")
    except Exception as e:
        logger.error(f"Ошибка при сохранении messages_with_topics.csv: {e}", exc_info=True)

    # 6. Генерация и сохранение визуализаций
    logger.info("Начало генерации визуализаций BERTopic...")
    try:
        fig_intertopic = topic_modeler.get_intertopic_distance_map()
        if fig_intertopic:
            fig_intertopic.write_html("intertopic_distance_map.html")
            logger.info("Карта межабстрактных расстояний сохранена в intertopic_distance_map.html")

        # Визуализация барчартов для нескольких тем (например, первых 5)
        # Убедимся, что есть темы для визуализации
        if num_unique_topics > 0:
            top_n_for_barchart = min(5, num_unique_topics) # Не более 5, или сколько есть
            fig_barchart = topic_modeler.get_topic_word_barchart(top_n_topics=top_n_for_barchart)
            if fig_barchart:
                fig_barchart.write_html("topic_word_barchart.html")
                logger.info(f"Барчарты слов для {top_n_for_barchart} тем сохранены в topic_word_barchart.html")
        else:
            logger.warning("Нет уникальных тем для генерации барчартов.")
        
    except Exception as e:
        logger.error(f"Ошибка при генерации визуализаций: {e}", exc_info=True)

    logger.info("=== Запуск модуля topic_modeler.py завершен ===")