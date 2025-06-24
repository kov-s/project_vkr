from flask import Flask, render_template, request
import pandas as pd
import os

app = Flask(__name__)

RAW_FILE = 'raw.csv'
PROCESSED_FILE = 'processed.csv'
CSV_FILE = 'messages.csv'

def create_or_update_messages_csv():
    if not os.path.exists(RAW_FILE) or not os.path.exists(PROCESSED_FILE):
        print("raw.csv или processed.csv не найдены.")
        return

    raw = pd.read_csv(RAW_FILE)
    processed = pd.read_csv(PROCESSED_FILE)

    df = pd.merge(raw, processed[['message_id', 'topic', 'topic_probability']], on='message_id', how='left')
    df.rename(columns={'topic': 'tags'}, inplace=True)

    if 'date' not in df.columns:
        df['date'] = pd.Timestamp.now()

    df.to_csv(CSV_FILE, index=False)

def load_messages():
    create_or_update_messages_csv()

    if not os.path.exists(CSV_FILE):
        return pd.DataFrame()

    df = pd.read_csv(CSV_FILE)

    # Дата
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    else:
        df['date'] = pd.NaT

    # Теги
    if 'tags' not in df.columns:
        df['tags'] = ''

    # Обработка media_path
    if 'media_path' in df.columns:
        df['media_path'] = df['media_path'].fillna('').apply(
            lambda p: p.replace('static/', '') if p.startswith('static/') else p
        )
    else:
        df['media_path'] = ''

    return df


@app.route('/')
@app.route('/')
def index():
    df = load_messages()

    search_text = request.args.get('search', '').strip()
    selected = request.args.get('tag', '')
    sort_order = request.args.get('sort_order', 'desc')

    if search_text:
        df = df[df['text'].str.contains(search_text, case=False, na=False)]

    if selected:
        df = df[df['tags'].fillna('').apply(lambda x: selected in x.split())]

    df = df.sort_values(by='date', ascending=(sort_order == 'asc'))

    # Пагинация
    page_size = 9
    try:
        page = int(request.args.get('page', 1))
        if page < 1:
            page = 1
    except ValueError:
        page = 1

    total_pages = max(1, (len(df) + page_size - 1) // page_size)
    if page > total_pages:
        page = total_pages

    paged = df.iloc[(page - 1) * page_size: page * page_size].copy()
    paged['date'] = paged['date'].dt.strftime('%Y-%m-%d %H:%M:%S').fillna('')

    tags = sorted(set(tag for t in df['tags'].dropna() for tag in t.split()))
    messages = df.to_dict(orient='records')  # отправляем все


    return render_template(
        'index.html',
        messages=messages,
        tags=tags,
        selected=selected,
        search=search_text,
        sort_order=sort_order,
        page=page,
        total_pages=total_pages
    )



@app.route('/topics')
def topics():
    df = load_messages()
    if df.empty:
        topics_list = []
    else:
        topics_set = set()
        for tags_str in df['tags'].dropna():
            topics_set.update(tags_str.split())
        topics_list = sorted(topics_set)

    selected_topic = request.args.get('topic', '')

    filtered_msgs = df
    if selected_topic:
        filtered_msgs = df[df['tags'].fillna('').apply(lambda x: selected_topic in x.split())]

    filtered_msgs = filtered_msgs.sort_values(by='date', ascending=False)
    filtered_msgs['date'] = filtered_msgs['date'].dt.strftime('%Y-%m-%d %H:%M:%S').fillna('')

    messages = filtered_msgs.to_dict(orient='records')

    return render_template(
        'topics.html',
        topics=topics_list,
        selected_topic=selected_topic,
        messages=messages
    )


if __name__ == '__main__':
    app.run(debug=True)
