import pandas as pd
from transformers import pipeline
import warnings
import sys
import os

# Отключаем предупреждения UserWarning от bertopic (если установлено)
warnings.filterwarnings("ignore", category=UserWarning, module='bertopic')

# --- 1. Загрузка данных ---
file_path = 'messages_sum.csv'
try:
    df = pd.read_csv(file_path)
    # Преобразование тегов из строки в список
    df['tags'] = df['tags'].apply(lambda x: [tag.strip() for tag in x.split(',')] if isinstance(x, str) else [])
    print(f"Загружено {len(df)} сообщений.")
except FileNotFoundError:
    print(f"Ошибка: Файл '{file_path}' не найден.")
    sys.exit(1)

# Проверяем, есть ли колонки 'topic' и 'topic_probability'
if 'topic' not in df.columns or 'topic_probability' not in df.columns:
    print("Внимание: В данных отсутствуют столбцы 'topic' или 'topic_probability'. Фильтрация по ним не будет работать.")

# --- 2. Инициализация модели суммаризации ---
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", framework="pt")
print("Модель для суммаризации загружена.")

# --- 3. Функция суммаризации с сохранением в файлы ---
def summarize_text_with_criteria(
    df: pd.DataFrame,
    topic_filter: str = None,
    min_topic_probability: float = 0.7,
    tags_filter: list = None,
    summary_length: int = 100,
    min_summary_length: int = 30,
    output_dir: str = None
):
    filtered_df = df.copy()

    if topic_filter and 'topic' in filtered_df.columns and 'topic_probability' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['topic'] == topic_filter) &
            (filtered_df['topic_probability'] >= min_topic_probability)
        ]
        print(f"Отфильтровано по теме '{topic_filter}': {len(filtered_df)} сообщений")

    if tags_filter:
        filtered_df = filtered_df[
            filtered_df['tags'].apply(lambda tags: any(tag in tags for tag in tags_filter) if isinstance(tags, list) else False)
        ]
        print(f"Отфильтровано по тегам {tags_filter}: {len(filtered_df)} сообщений")

    if filtered_df.empty:
        print("Нет сообщений для суммаризации по заданным критериям.")
        return {}

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    summaries = {}
    for _, row in filtered_df.iterrows():
        message_id = row.get('message_id', None)
        text = row['text']

        if not isinstance(text, str) or len(text.split()) < 10:
            print(f"Сообщение ID {message_id} слишком короткое для суммаризации, пропускаем.")
            continue

        try:
            summary = summarizer(
                text,
                max_length=summary_length,
                min_length=min_summary_length,
                do_sample=False
            )[0]['summary_text']
            summaries[message_id] = summary
            print(f"Суммаризовано сообщение ID {message_id}")

            if output_dir and message_id is not None:
                filename = os.path.join(output_dir, f"summary_{message_id}.txt")
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(summary)

        except Exception as e:
            print(f"Ошибка при суммаризации сообщения ID {message_id}: {e}")

    return summaries

# 
if __name__ == "__main__":
   
    print("--- Суммаризация всех сообщений ---")
    all_summaries = summarize_text_with_criteria(
        df,
        summary_length=60,
        min_summary_length=20,
        output_dir="summaries"
    )
    print(f"Всего сгенерировано и сохранено резюме: {len(all_summaries)}")

   
