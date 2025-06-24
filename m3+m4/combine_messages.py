import pandas as pd
import numpy as np 

# Загрузка CSV-файлов
# Убедитесь, что 'raw.csv' содержит 'message_id', 'text', 'date', 'tags', 'media_path'
# Убедитесь, что 'processed.csv' содержит 'message_id', 'topic', 'topic_name', 'topic_probability'
raw_df = pd.read_csv('raw.csv')
processed_df = pd.read_csv('processed.csv')

print(f"Загружено raw.csv: {len(raw_df)} записей")
print(f"Загружено processed.csv: {len(processed_df)} записей")

# Удаляем дубликаты по message_id
raw_df = raw_df.drop_duplicates(subset='message_id')
processed_df = processed_df.drop_duplicates(subset='message_id')

print(f"После удаления дубликатов raw.csv: {len(raw_df)} записей")
print(f"После удаления дубликатов processed.csv: {len(processed_df)} записей")

# Объединение с приоритетом raw_df (left join)
# Добавляем 'topic_name' в список объединяемых столбцов
merged_df = pd.merge(
    raw_df,
    processed_df[['message_id', 'topic_id', 'topic_name', 'topic_probability']], # Изменено: добавлено 'topic_name' и 'topic_id'
    on='message_id',
    how='left'
)

# Обработка NaN для новых столбцов
# -1 для числового ID темы, 0.0 для вероятности, 'Без темы' для названия темы
merged_df['topic_id'] = merged_df['topic_id'].fillna(-1).astype(int)  # Изменено: используем topic_id
merged_df['topic_name'] = merged_df['topic_name'].fillna('Без темы') # НОВОЕ: Обработка NaN для topic_name
merged_df['topic_probability'] = merged_df['topic_probability'].fillna(0.0)

# Переименуем столбец 'topic_id' обратно в 'topic'
merged_df.rename(columns={'topic_id': 'topic'}, inplace=True)


# Сохранение результата
merged_df.to_csv('messages.csv', index=False, encoding='utf-8')

print("\nФайл 'messages.csv' успешно создан.")
print("Пустые topic (ID) заменены на -1.")
print("Пустые topic_name заменены на 'Без темы'.")
print("Пустые topic_probability заменены на 0.0.")
print(f"Итоговое количество записей в messages.csv: {len(merged_df)}")
print("\nПример первых 5 строк объединенного DataFrame:")
print(merged_df.head())