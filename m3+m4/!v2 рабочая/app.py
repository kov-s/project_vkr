from flask import Flask, render_template
import pandas as pd
import re

app = Flask(__name__)

def parse_tags(tag_str):
    if not isinstance(tag_str, str) or tag_str.strip() == "":
        return []
    tags = tag_str.strip().split()
    clean_tags = []
    for tag in tags:
        tag = tag.lstrip("#")
        # Отсекаем теги, которые начинаются с цифры
        if re.match(r'^[0-9]', tag):
            continue
        # Тег должен содержать хотя бы одну букву
        if re.search(r'[a-zA-Zа-яА-Я]', tag):
            clean_tags.append(tag)
    return clean_tags


@app.route("/")
def index():
    df = pd.read_csv("messages_sum.csv")

    # Обрабатываем поле tags в список
    df["tags"] = df["tags"].fillna("").apply(parse_tags)

    # Собираем уникальные теги из всех сообщений
    all_tags = set()
    for tags_list in df["tags"]:
        all_tags.update(tags_list)
    all_tags = sorted(all_tags)

    # Уникальные темы (topic)
    all_topics = sorted(df["topic"].dropna().unique(), key=lambda x: str(x))

    messages = []
    for _, row in df.iterrows():
        messages.append({
            "id": row["message_id"],
            "date": row["date"],
            "message": row["text"],
            "media_url": row["media_path"] if pd.notna(row["media_path"]) else "",
            "predicted_tags": row["tags"],
            "topic": row["topic"],
            "topic_probability": row["topic_probability"]
        })

    return render_template("index.html", messages=messages, all_tags=all_tags, all_topics=all_topics)


if __name__ == "__main__":
    app.run(debug=True)
