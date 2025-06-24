from flask import Flask, render_template, request, jsonify
from database.session import SessionLocal
from database.models import Message
import logging
import re
import traceback
from typing import List, Optional
from fred_summarizer import FredSummarizer  # импорт класса

import json
import os

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['JSON_AS_ASCII'] = False

# Создаем один экземпляр глобально
summarizer = None

DATA_FILE = 'data.json'


def load_data():
    if not os.path.exists(DATA_FILE):
        return []
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_data(data):
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


@app.route('/')
def index():
    db = None
    try:
        db = SessionLocal()
        # Извлекаем последние сообщения, отсортированные по дате.
        messages = db.query(Message).order_by(Message.date.desc()).limit(700).all()

        if not messages:
            logging.warning("В базе данных не найдено сообщений!")
            # Если сообщений нет, возвращаем пустые списки
            return render_template('index.html', messages=[], all_topics=[])

        messages_data = []
        all_topics_for_filter = set() # Используем set для сбора уникальных названий тем

        for msg in messages:
            try:
                # Извлекаем название темы из объекта сообщения.
                # Если topic_name отсутствует или пустое, используем 'Без темы'.
                topic_display_name = msg.topic_name if msg.topic_name and msg.topic_name.strip() != '' else 'Без темы'

                msg_data = {
                    'id': msg.message_id,
                    'message': msg.text,
                    'date': msg.date.isoformat() if msg.date else None,
                    'predicted_tags': parse_tags(msg.tags) if msg.tags else [],
                    'topic': msg.topic, # Сохраняем числовой ID темы, если он нужен для внутренних целей
                    'topic_name': topic_display_name, # Название темы для отображения и фильтрации на фронтенде
                    'topic_probability': float(msg.topic_probability) if msg.topic_probability else 0.0,
                    'media_url': msg.media_path or ""
                }
                messages_data.append(msg_data)

                # Добавляем название темы в набор для формирования списка фильтров
                all_topics_for_filter.add(topic_display_name)

            except Exception as e:
                # Логируем ошибку, но продолжаем обработку остальных сообщений
                logging.error(f"Ошибка при обработке сообщения ID {getattr(msg, 'message_id', 'N/A')}: {str(e)}")
                continue # Переходим к следующему сообщению

        # Сортируем названия тем для выпадающего списка на фронтенде
        sorted_topics = safe_sort_topics(list(all_topics_for_filter))
        logging.debug(f"Отсортированные названия тем для фронтенда: {sorted_topics}")

        return render_template(
            'index.html',
            messages=messages_data,
            all_topics=sorted_topics # Передаем названия тем
        )

    except Exception as e:
        # Обработка критических ошибок на уровне всего маршрута
        logging.error(f"Критическая ошибка в маршруте index: {str(e)}\n{traceback.format_exc()}")
        return "Внутренняя ошибка сервера", 500
    finally:
        # Убедимся, что соединение с БД закрыто
        if db:
            db.close()

@app.route('/summary_result')
def summary_result():
    topic = request.args.get('topic', '')
    return render_template('summary_result.html', topic=topic)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('flask_error.log'),
        logging.StreamHandler()
    ]
)


def parse_tags(tag_str: Optional[str]) -> List[str]:
    if not tag_str or not isinstance(tag_str, str) or tag_str.strip() == "":
        return []

    tags = tag_str.strip().split()
    clean_tags = []

    for tag in tags:
        tag = tag.lstrip("#")
        if re.match(r'^[0-9]', tag) or not re.search(r'[a-zA-Zа-яА-Я]', tag):
            continue
        clean_tags.append(tag.lower())

    return clean_tags


def safe_sort_topics(topics):
    def key_func(x):
        try:
            return (0, float(x))
        except (ValueError, TypeError):
            try:
                return (1, str(x).lower())
            except:
                return (2, str(x))

    return sorted(topics, key=key_func)


@app.route('/update_message', methods=['POST'])
def update_message():
    data = request.get_json()
    if not data or 'id' not in data:
        return jsonify({'error': 'Invalid request'}), 400

    db = SessionLocal()
    try:
        msg = db.query(Message).get(data['id'])
        if msg:
            if 'predicted_tags' in data and isinstance(data['predicted_tags'], list):
                msg.tags = ' '.join(data['predicted_tags'])
                db.commit()
                return jsonify({'status': 'success'})
        return jsonify({'error': 'Message not found'}), 404
    except Exception as e:
        db.rollback()
        logging.error(f"Update error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500
    finally:
        db.close()

@app.route('/summarize', methods=['POST'])
def summarize():
    global summarizer

    if summarizer is None:
        summarizer = FredSummarizer()

    # Получение текста из запроса
    if request.is_json:
        data = request.get_json()
        text = data.get('text', '')
    else:
        text = request.form.get('text', '')

    if not text:
        return jsonify({'error': 'Пустой текст'}), 400

    try:
        summary = summarizer.summarize(text)
        if summary is None:
            return jsonify({'error': 'Суммирование не удалось'}), 500

        return jsonify({'summary': summary}), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Ошибка суммирования: {str(e)}'}), 500


@app.route('/manage_tags')
def manage_tags():
    db = SessionLocal()
    try:
        messages = db.query(Message).all()
        tags = {}
        for msg in messages:
            msg_tags = parse_tags(msg.tags)
            for tag in msg_tags:
                tags[tag] = tags.get(tag, 0) + 1
        sorted_tags = dict(sorted(tags.items()))
        return render_template('manage_tags.html', tags=sorted_tags)  # передаем переменную 'tags'
    except Exception as e:
        logging.error(f"Manage tags error: {e}\n{traceback.format_exc()}")
        return "Internal Server Error", 500
    finally:
        db.close()



@app.route('/api/tags/add', methods=['POST'])
def add_tag():
    data = request.json
    new_tag = data.get('tag', '').strip()
    if not new_tag:
        return jsonify({'error': 'Empty tag'}), 400

    messages = load_data()

    all_tags = set()
    for msg in messages:
        all_tags.update(msg.get('predicted_tags', []))
    if new_tag in all_tags:
        return jsonify({'error': 'Tag already exists'}), 400

    messages.append({
        "id": f"taggen_{len(messages) + 1}",
        "message": "",
        "predicted_tags": [new_tag],
        "topic": "",
        "topic_probability": 0,
        "date": ""
    })
    save_data(messages)
    return jsonify({'success': True})


@app.route('/api/tags/edit', methods=['POST'])
def edit_tag():
    data = request.json
    old_tag = data.get('old_tag')
    new_tag = data.get('new_tag', '').strip()

    if not old_tag or not new_tag:
        return jsonify({'error': 'Invalid data'}), 400

    db = SessionLocal()
    try:
        # Найдем все сообщения, у которых в строке тегов есть old_tag
        messages = db.query(Message).filter(Message.tags.like(f'%{old_tag}%')).all()
        
        if not messages:
            return jsonify({'error': 'Old tag not found'}), 404

        for msg in messages:
            # Парсим строку тегов в список
            tags_list = parse_tags(msg.tags)
            # Заменяем старый тег на новый
            updated_tags = [new_tag if t == old_tag else t for t in tags_list]
            # Обновляем строку тегов в модели
            msg.tags = ' '.join(updated_tags)

        db.commit()  # Важно: коммитим изменения в базе!

        return jsonify({'success': True})
    except Exception as e:
        db.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        db.close()


@app.route('/api/tags/delete', methods=['POST'])
def delete_tag():
    data = request.json
    tag = data.get('tag')
    if not tag:
        return jsonify({'error': 'No tag specified'}), 400

    messages = load_data()
    changed = False
    for msg in messages:
        if tag in msg.get('predicted_tags', []):
            msg['predicted_tags'] = [t for t in msg['predicted_tags'] if t != tag]
            changed = True

    messages = [m for m in messages if tag not in m.get('predicted_tags', []) or m.get('message')]

    if changed:
        save_data(messages)

    return jsonify({'success': True})


if __name__ == '__main__':
    # Для production используйте:
    # app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    # Для разработки:
    app.run(host='127.0.0.1', port=5000, debug=True)
