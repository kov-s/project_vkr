import pandas as pd

# Загрузка CSV-файлов
raw_df = pd.read_csv('raw.csv')
processed_df = pd.read_csv('processed.csv')

# Удалим дубликаты по message_id
raw_df = raw_df.drop_duplicates(subset='message_id')
processed_df = processed_df.drop_duplicates(subset='message_id')

# Объединение с приоритетом raw.csv (left join)
merged_df = pd.merge(
    raw_df,
    processed_df[['message_id', 'topic', 'topic_probability']],
    on='message_id',
    how='left'
)

# Обработка NaN
merged_df['topic'] = merged_df['topic'].fillna(-1)  # -1 вместо пустого topic
merged_df['topic_probability'] = merged_df['topic_probability'].fillna(0.0)  # 0.0 по умолчанию

# Сохранение результата
merged_df.to_csv('messages_sum.csv', index=False)

print("Файл 'messages_sum.csv' создан. Пустые topic заменены на -1.")
