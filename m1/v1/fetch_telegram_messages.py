import os
import re
import csv
import asyncio
import logging
from telethon import TelegramClient
from telethon.tl.types import MessageMediaPhoto, MessageMediaDocument
from telethon.tl.functions.messages import GetHistoryRequest

# Настройки логирования 
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')

# Настройки Telegram 
api_id = xxxxxxxx  
api_hash = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'  
session_name = 'session'
channel_username = '@xxxxxx'  # @channel_name

# Путь к файлам 
media_folder = 'static/mediatest'
csv_filename = 'messagestest.csv'
offset_file = 'offsettest.txt'
os.makedirs(media_folder, exist_ok=True)

# Функции 

def extract_tags(text):
    return re.findall(r"#\w+", text.lower()) if text else []

def load_saved_message_ids():
    if not os.path.exists(csv_filename):
        return set()
    with open(csv_filename, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return {int(row["message_id"]) for row in reader if row.get("message_id", "").isdigit()}

def load_offset_id():
    if os.path.exists(offset_file):
        with open(offset_file, "r") as f:
            value = f.read().strip()
            return int(value) if value.isdigit() else 0
    return 0

def save_offset_id(offset_id):
    with open(offset_file, "w") as f:
        f.write(str(offset_id))

async def fetch_messages():
    saved_ids = load_saved_message_ids()
    logging.info(f" Уже сохранено сообщений: {len(saved_ids)}")

    offset_id = load_offset_id()
    logging.info(f" Начинаем с offset_id = {offset_id}")

    async with TelegramClient('session', api_id, api_hash) as client:
        channel = await client.get_entity(channel_username)
        limit = 100
        max_messages = 2000
        total_downloaded = 0

        file_exists = os.path.exists(csv_filename)
        with open(csv_filename, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["message_id", "text", "media_path", "tags", "date"])
            if not file_exists or os.stat(csv_filename).st_size == 0:
                writer.writeheader()

            while total_downloaded < max_messages:
                history = await client(GetHistoryRequest(
                    peer=channel,
                    offset_id=offset_id,
                    offset_date=None,
                    add_offset=0,
                    limit=limit,
                    max_id=0,
                    min_id=0,
                    hash=0
                ))

                if not history.messages:
                    break

                for message in history.messages:
                    if total_downloaded >= max_messages:
                        break

                    if message.id in saved_ids or not message.message:
                        continue

                    text = message.message
                    tags = extract_tags(text)
                    media_path = ''

                    if message.media:
                        if isinstance(message.media, MessageMediaPhoto):
                            media_path = os.path.join(media_folder, f"{message.id}.jpg")
                            await client.download_media(message.media, media_path)
                        elif isinstance(message.media, MessageMediaDocument):
                            doc = message.media.document
                            mime_type = getattr(doc, 'mime_type', '')
                            size_limit = 300 * 1024 * 1024  # 300 MB

                            if 'video' in mime_type:
                                too_long = False
                                for attr in doc.attributes:
                                    if hasattr(attr, 'duration') and attr.duration > 900:
                                        too_long = True
                                        break
                                if too_long:
                                    media_path = "Видео >15 мин"
                                elif doc.size and doc.size > size_limit:
                                    media_path = f"Видео > {size_limit // 1024 // 1024} MB"
                                else:
                                    media_path = os.path.join(media_folder, f"{message.id}.mp4")
                                    try:
                                        await asyncio.wait_for(client.download_media(doc, media_path), timeout=60)
                                    except asyncio.TimeoutError:
                                        media_path = "Ошибка загрузки (таймаут)"
                            else:
                                media_path = os.path.join(media_folder, f"{message.id}.doc")
                                try:
                                    await asyncio.wait_for(client.download_media(doc, media_path), timeout=60)
                                except asyncio.TimeoutError:
                                    media_path = "Ошибка загрузки (таймаут)"

                    row = {
                        "message_id": message.id,
                        "text": text,
                        "media_path": media_path.replace('\\', '/'),
                        "tags": ' '.join(tags),
                        "date": message.date.strftime("%Y-%m-%d %H:%M")
                    }

                    writer.writerow(row)
                    f.flush()
                    saved_ids.add(message.id)
                    total_downloaded += 1
                    offset_id = message.id
                    save_offset_id(offset_id)

                    if total_downloaded % 100 == 0:
                        logging.info(f"Обработано {total_downloaded} сообщений...")

        logging.info(f"Готово: добавлено {total_downloaded} новых сообщений в {csv_filename}")

if __name__ == "__main__":
    asyncio.run(fetch_messages())
