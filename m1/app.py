from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from datetime import datetime
import os

app = Flask(__name__)

RAW_FILE = 'raw.csv'
PROCESSED_FILE = 'processed.csv'
CSV_FILE = 'messages.csv'


# Объединение данных из raw.csv и processed.csv
def create_or_update_messages_csv():
    if not os.path.exists(RAW_FILE) or not os.path.exists(PROCESSED_FILE):
        print("raw.csv или processed.csv не найдены.")
        return

    raw = pd.read_csv(RAW_FILE)
    processed = pd.read_csv(PROCESSED_FILE)

    # Объединяем по message_id
    df = pd.merge(raw, processed[['message_id', 'topic', 'topic_probability']], on='message_id', how='left')

    #'topic' в 'tags'
    df.rename(columns={'topic': 'tags'}, inplace=True)

    # столбц 'date'
    if 'date' not in df.columns:
        df['date'] = pd.Timestamp.now()

    df.to_csv(CSV_FILE, index=False)


# Загрузка messages.csv 

def load_messages():
    create_or_update_messages_csv()

    df = pd.read_csv(CSV_FILE)

    if 'date' not in df.columns:
        df['date'] = pd.NaT
    else:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    if 'tags' not in df.columns:
        df['tags'] = ''

    return df


@app.route('/')
def index():
    df = load_messages()

    selected = request.args.get('tag')
    date_from = request.args.get('from')
    date_to = request.args.get('to')
    sort_order = request.args.get('sort_order', 'desc')
    page = int(request.args.get('page', 1))

    if selected:
        df = df[df['tags'].fillna('').str.contains(rf'\b{selected}\b', na=False)]

    if date_from:
        df = df[df['date'] >= pd.to_datetime(date_from, errors='coerce')]
    if date_to:
        df = df[df['date'] <= pd.to_datetime(date_to, errors='coerce')]

    df = df.sort_values(by='date', ascending=(sort_order == 'asc'))

    page_size = 9
    total_pages = (len(df) + page_size - 1) // page_size
    paged = df.iloc[(page - 1) * page_size: page * page_size]

    tags = sorted(set(tag for t in df['tags'].dropna() for tag in t.split()))

    return render_template(
        'index.html',
        messages=paged.to_dict(orient='records'),
        tags=tags,
        selected=selected,
        date_from=date_from or '',
        date_to=date_to or '',
        sort_order=sort_order,
        page=page,
        total_pages=total_pages
    )


@app.route('/add_tag/<int:message_id>', methods=['POST'])
def add_tag(message_id):
    new_tag = request.form.get('new_tag', '').strip().lstrip('#')
    if not new_tag:
        return redirect(url_for('index'))

    df = load_messages()
    idx = df[df['message_id'] == message_id].index
    if not idx.empty:
        current = df.at[idx[0], 'tags']
        tags = set(current.split() if pd.notna(current) else [])
        tags.add(new_tag)
        df.at[idx[0], 'tags'] = ' '.join(tags)
        df.to_csv(CSV_FILE, index=False)

    return redirect(request.referrer)


@app.route('/remove_tag/<int:message_id>/<tag>')
def remove_tag(message_id, tag):
    df = load_messages()
    idx = df[df['message_id'] == message_id].index
    if not idx.empty:
        tags = set(df.at[idx[0], 'tags'].split())
        if len(tags) > 1 and tag in tags:
            tags.remove(tag)
            df.at[idx[0], 'tags'] = ' '.join(tags)
            df.to_csv(CSV_FILE, index=False)
    return redirect(request.referrer)


if __name__ == '__main__':
    app.run(debug=True)
