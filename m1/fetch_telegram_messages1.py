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

# Папка для медиа 
media_folder = 'static/media1'
os.makedirs(media_folder, exist_ok=True)

# Файл CSV 
csv_filename = 'static/raw.csv'

# Получение тегов из текста 
def extract_tags(text):
    return re.findall(r"#\w+", text.lower()) if text else []

# Загрузка ID уже сохранённых сообщений
def load_saved_message_ids():
    if not os.path.exists(csv_filename):
        return set()
    with open(csv_filename, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return {int(row["message_id"]) for row in reader}

# Асинхронная загрузка сообщений 
async def fetch_messages():
    saved_ids = load_saved_message_ids()
    logging.info(f"📂 Уже сохранено сообщений: {len(saved_ids)}")

    async with TelegramClient('session', api_id, api_hash) as client:
        channel = await client.get_entity(channel_username)
        offset_id = 0
        limit = 100
        max_messages = 2000
        total_downloaded = 0
        all_messages = []

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

                if message.id in saved_ids:
                    continue  # Пропускаем уже сохранённое сообщение

                text = message.message or ''
                tags = extract_tags(text)
                media_path = ''

                if message.media:
                    if isinstance(message.media, MessageMediaPhoto):
                        media_path = os.path.join(media_folder, f"{message.id}.jpg")
                        await client.download_media(message.media, media_path)
                    elif isinstance(message.media, MessageMediaDocument):
                        mime_type = getattr(message.media.document, 'mime_type', '')
                        ext = '.mp4' if 'video' in mime_type else '.doc'
                        media_path = os.path.join(media_folder, f"{message.id}{ext}")
                        await client.download_media(message.media, media_path)

                all_messages.append({
                    "message_id": message.id,
                    "text": text,
                    "media_path": media_path.replace('\\', '/'),
                    "tags": ' '.join(tags),
                    "date": message.date.strftime("%Y-%m-%d %H:%M")
                })

                total_downloaded += 1
                saved_ids.add(message.id)  # Добавляем в множество

            offset_id = history.messages[-1].id

        if all_messages:
            # Добавляем в существующий CSV 
            file_exists = os.path.exists(csv_filename)
            with open(csv_filename, "a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["message_id", "text", "media_path", "tags", "date"])
                if not file_exists:
                    writer.writeheader()
                writer.writerows(all_messages)

        logging.info(f"✅ Добавлено {len(all_messages)} новых сообщений в {csv_filename}")

if __name__ == "__main__":
    asyncio.run(fetch_messages())
