from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
import numpy as np

def calculate_coherence(topic_model, documents, preprocessed_documents):
    """
    Вычисляет метрику когерентности тем (c_v).
    :param topic_model: Обученная модель BERTopic.
    :param documents: Список ОРИГИНАЛЬНЫХ текстовых документов.
    :param preprocessed_documents: Список ПРЕДОБРАБОТАННЫХ текстовых документов,
                                  которые были поданы в BERTopic.
    :return: Значение когерентности c_v.
    """
    if topic_model.topics is None:
        logger.warning("Модель BERTopic не обучена, когерентность не может быть вычислена.")
        return None

    # Получаем токены (слова) из предобработанных документов
    # Важно: здесь нужны именно токены, а не леммы
    # Если processed_documents - это строки, нужно их разбить на слова.
    tokenized_documents = [doc.split() for doc in preprocessed_documents]

    # Создаем словарь Gensim
    dictionary = Dictionary(tokenized_documents)
    corpus = [dictionary.doc2bow(text) for text in tokenized_documents]

    # Получаем представления тем от BERTopic
    topics_words = topic_model.get_topics() # Получаем dict {topic_id: [(word, prob), ...]}

    # Фильтруем Topic -1
    topics_words_filtered = {k: v for k, v in topics_words.items() if k != -1}

    # Извлекаем только слова для каждой темы
    texts_for_coherence = []
    for topic_id in sorted(topics_words_filtered.keys()):
        # Слова для данной темы (в порядке убывания важности)
        words = [word for word, _ in topics_words_filtered[topic_id]]
        texts_for_coherence.append(words)

    if not texts_for_coherence:
        logger.warning("Нет достаточных тем для вычисления когерентности (возможно, только выбросы).")
        return None

    # Вычисляем когерентность
    coherence_model = CoherenceModel(
        topics=texts_for_coherence, 
        texts=tokenized_documents, # Предобработанные, токенизированные документы
        dictionary=dictionary, 
        coherence='c_v'
    )
    coherence_score = coherence_model.get_coherence()
    logger.info(f"Когерентность тем (c_v): {coherence_score}")
    return coherence_score

# В `if __name__ == '__main__':` после обучения модели:
# # ... (после строки `topics, probabilities = topic_modeler.fit_transform(documents_for_bertopic)`)
# coherence_score = calculate_coherence(topic_modeler.topic_model, df_filtered['text'].tolist(), documents_for_bertopic)
# if coherence_score is not None:
#     logger.info(f"Итоговый показатель когерентности c_v: {coherence_score:.4f}")