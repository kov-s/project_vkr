import pandas as pd

# Загрузка исходных файлов
df_raw = pd.read_csv('raw.csv')  # message_id, text, media_path, tags, date
df_processed = pd.read_csv('processed.csv')  # message_id, text, processed_text, extracted_hashtags, topic, topic_probability
df_topic_info = pd.read_csv('topic_info.csv')  # topic, count, name, representation, representative_Docs

# Объединение raw и processed по message_id
merged = pd.merge(df_raw, df_processed[['message_id', 'topic', 'topic_probability']], on='message_id', how='inner')

# Добавление информации о теме по topic
final = pd.merge(merged, df_topic_info[['topic', 'name']], on='topic', how='left')

# Формирование итоговой таблицы с нужными столбцами
final = final[['message_id', 'text', 'media_path', 'tags', 'date', 'topic', 'name', 'topic_probability']]

# Сохранение результата
final.to_csv('messages_sum.csv', index=False)
