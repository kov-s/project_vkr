from flask import Flask, render_template
import pandas as pd
import ast

app = Flask(__name__)

@app.route("/")
def index():
    # Загружаем CSV
    df = pd.read_csv("messages_sum.csv")

    # Преобразуем 'tags' в список
    def parse_tags(tag_str):
        try:
            return ast.literal_eval(tag_str)
        except:
            return []

    df["tags"] = df["tags"].fillna("[]").apply(parse_tags)
    
    # Формируем список словарей для шаблона
    messages = []
    for _, row in df.iterrows():
        messages.append({
            "date": row["date"],
            "message": row["text"],
            "media_url": row["media_path"] if pd.notna(row["media_path"]) else "",
            "predicted_tags": row["tags"]
        })

    return render_template("index.html", messages=messages)

if __name__ == "__main__":
    app.run(debug=True)
