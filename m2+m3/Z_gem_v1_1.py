import asyncio
from telethon.sync import TelegramClient
from telethon.tl.functions.channels import GetFullChannelRequest
from telethon.tl.types import Channel
import re
from nltk.corpus import stopwords
from gensim import corpora
from gensim.models import LdaModel
from collections import Counter
import logging
import os # Для работы с файловой системой

# Импорт компонентов Natasha
from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    Doc
)

# Настройка логирования для Gensim (опционально, но полезно)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# --- 1. Настройка Telegram API ---
# Замените на ваши API ID и Hash. Получите их на my.telegram.org
API_ID = 10495315
API_HASH = 'c07d122a96e631088c8f5f1e6fcbf80a'
session_name = 'anon'

# Имя сессии для TelegramClient.
# Это имя будет использоваться для файла сессии (например, 'telegram_topic_modeling_session.session').
# Файл сессии будет создан в том же каталоге, где находится скрипт.


async def get_channel_messages(client, channel_entity, limit=1000):
    """
    Асинхронно извлекает сообщения из указанного Telegram-канала.
    """
    messages_text = []
    # Используем client.iter_messages для получения сообщений.
    # Добавим try-except для обработки ошибок при доступе к сообщениям канала.
    try:
        async for message in client.iter_messages(channel_entity, limit=limit):
            if message.text:
                messages_text.append(message.text)
    except Exception as e:
        print(f"Ошибка при извлечении сообщений из канала: {e}")
        print("Возможно, у вас нет доступа к этому каналу, или он частный.")
    return messages_text

# --- 2. Инициализация компонентов Natasha и предобработка текста ---

# Инициализация компонентов Natasha.
# Эти объекты создаются один раз, так как загрузка моделей требует времени.
print("Инициализация компонентов Natasha...")
segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
print("Компоненты Natasha инициализированы.")

# Загружаем русские стоп-слова из NLTK
russian_stopwords = set(stopwords.words('russian'))

def preprocess_text(text):
    """
    Выполняет предобработку текста с использованием Natasha:
    очистка, токенизация, лемматизация, удаление стоп-слов.
    """
    # 1. Очистка текста от специфических элементов Telegram и пунктуации
    text = re.sub(r'http\S+|www\S+|bit\.ly\S+', '', text, flags=re.MULTILINE) # Удаляем ссылки
    text = re.sub(r'@\w+|#\w+', '', text) # Удаляем упоминания (@username) и хэштеги (#hashtag)
    text = re.sub(r'[\d_]+', '', text) # Удаляем цифры и подчеркивания (оставляем только буквы и пробелы)
    text = text.lower() # Приводим к нижнему регистру

    # 2. Обработка текста с Natasha
    doc = Doc(text)
    doc.segment(segmenter)      # Сегментация (разбиение на токены)
    doc.tag_morph(morph_tagger) # Определение морфологических признаков
    for token in doc.tokens:
        token.lemmatize(morph_vocab) # Лемматизация

    # 3. Фильтрация и сбор лемм
    lemmas = []
    for token in doc.tokens:
        lemma = token.lemma
        # Проверяем, что лемма состоит только из букв и не является стоп-словом
        if lemma.isalpha() and lemma not in russian_stopwords:
            lemmas.append(lemma)
    return lemmas

# --- 3. Тематическое моделирование с Gensim (LDA) ---

async def main():
    # Использование файла сессии для TelegramClient
    # Telethon автоматически создаст или загрузит файл SESSION_NAME.session
    async with TelegramClient(SESSION_NAME, API_ID, API_HASH) as client:
        # Авторизация:
        # Если файл сессии существует и действителен, авторизация пройдет автоматически.
        # В противном случае, будет запрошен номер телефона и код.
        if not await client.is_user_authorized():
            print(f"Файл сессии '{SESSION_NAME}.session' не найден или недействителен. Требуется авторизация.")
            await client.start(phone=PHONE_NUMBER)
            print(f"Авторизация прошла успешно. Сессия сохранена в '{SESSION_NAME}.session'.")
        else:
            print(f"Сессия загружена из файла '{SESSION_NAME}.session'.")

        # Запрос названия канала у пользователя
        channel_name_or_id = input("Введите username или ID Telegram-канала (например, @durov или -1001234567890): ")

        try:
            # Получаем сущность канала
            entity = await client.get_entity(channel_name_or_id)
            if not isinstance(entity, Channel):
                print(f"'{channel_name_or_id}' не является Telegram-каналом. Пожалуйста, введите корректный username или ID канала.")
                return

            print(f"Получение сообщений из канала: {entity.title} ({entity.username})...")
            messages = await get_channel_messages(client, entity, limit=1000) # Ограничиваем количество сообщений для примера

            if not messages:
                print("Не удалось найти сообщения в канале или канал пуст.")
                return

            print(f"Извлечено {len(messages)} сообщений. Начинаем предобработку...")

            # Предобработка всех сообщений
            processed_docs = [preprocess_text(msg) for msg in messages if preprocess_text(msg)]

            if not processed_docs:
                print("После предобработки не осталось валидных документов для анализа.")
                return

            # Создание словаря (Dictionary) из предобработанных документов
            dictionary = corpora.Dictionary(processed_docs)
            dictionary.filter_extremes(no_below=5, no_above=0.9)

            # Создание корпуса (Corpus) - списка частот слов для каждого документа
            corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

            print(f"Словарь содержит {len(dictionary)} уникальных токенов. Корпус содержит {len(corpus)} документов.")

            # Определение количества тем (может быть настроено)
            num_topics = 5
            print(f"Запускаем LDA-модель с {num_topics} темами...")

            # Обучение LDA-модели
            lda_model = LdaModel(corpus=corpus,
                                 id2word=dictionary,
                                 num_topics=num_topics,
                                 random_state=42,
                                 chunksize=2000,
                                 passes=10,
                                 alpha='auto',
                                 eta='auto',
                                 per_word_topics=True)

            print("\n--- Выделенные темы ---")
            for idx, topic in lda_model.print_topics(num_words=10):
                print(f"Тема #{idx}: {topic}")

            print("\n--- Распределение постов по темам (примеры) ---")
            for i, doc_bow in enumerate(corpus[:5]):
                topic_distribution = lda_model.get_document_topics(doc_bow, minimum_probability=0.01)
                if topic_distribution:
                    print(f"\nПост {i+1} (часть текста: '{messages[i][:100].replace('\n', ' ')}...'):")
                    for topic_idx, prob in sorted(topic_distribution, key=lambda x: x[1], reverse=True):
                        print(f"  Тема #{topic_idx}: Вероятность {prob:.3f}")
                else:
                    print(f"\nПост {i+1}: Не удалось определить явные темы.")

            print("\n--- Общее распределение тем по каналу ---")
            topic_counts = Counter()
            for doc_bow in corpus:
                topic_distribution = lda_model.get_document_topics(doc_bow, minimum_probability=0.01)
                if topic_distribution:
                    most_likely_topic = max(topic_distribution, key=lambda item: item[1])[0]
                    topic_counts[most_likely_topic] += 1

            total_docs_with_topics = sum(topic_counts.values())
            if total_docs_with_topics > 0:
                for topic_idx, count in topic_counts.most_common():
                    percentage = (count / total_docs_with_topics) * 100
                    print(f"Тема #{topic_idx}: {count} документов ({percentage:.2f}%)")
            else:
                print("Нет документов, распределенных по темам.")

        except Exception as e:
            print(f"\nПроизошла ошибка: {e}")
            print("Убедитесь, что вы ввели корректный username/ID канала и API_ID/API_HASH.")
            print("Для частных каналов боту нужен доступ, либо вы должны быть его участником.")

if __name__ == '__main__':
    asyncio.run(main())