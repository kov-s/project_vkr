import os
import csv
import re
import asyncio
from datetime import datetime
from telethon.sync import TelegramClient
from telethon.tl.types import MessageMediaPhoto, MessageMediaDocument
from telethon.tl.functions.messages import GetHistoryRequest

# Настройки Telegram 
api_id = xxxxxxxx  
api_hash = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'  
session_name = 'session'
channel_username = '@xxxxxx'  # @channel_name

# Папка для медиа 
media_folder = 'static/media'
os.makedirs(media_folder, exist_ok=True)

# Файл CSV 
csv_filename = 'raw.csv'

# Получение тегов из текста 
def extract_tags(text):
    return re.findall(r"#\w+", text.lower()) if text else []

# Асинхронная загрузка сообщений
async def fetch_messages():
    async with TelegramClient('session', api_id, api_hash) as client:
        channel = await client.get_entity(channel_username)
        offset_id = 0
        limit = 100
        all_messages = []

        while True:
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
                if not message.message:
                    continue

                text = message.message
                tags = extract_tags(text)
                media_path = ''

                if message.media:
                    if isinstance(message.media, MessageMediaPhoto):
                        media_path = os.path.join(media_folder, f"{message.id}.jpg")
                        await client.download_media(message.media, media_path)
                    elif isinstance(message.media, MessageMediaDocument):
                        ext = '.mp4' if 'video' in message.media.document.mime_type else '.doc'
                        media_path = os.path.join(media_folder, f"{message.id}{ext}")
                        await client.download_media(message.media, media_path)

                all_messages.append({
                    "message_id": message.id,
                    "text": text,
                    "media_path": media_path.replace('\\', '/'),
                    "tags": ' '.join(tags),
                    "date": message.date.strftime("%Y-%m-%d %H:%M")
                })

                offset_id = message.id

        # Автоклассификация
        tag_counts = {}
        for m in all_messages:
            for tag in m["tags"].split():
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        common_tags = list(tag_counts.keys())

        for m in all_messages:
            if not m["tags"] and common_tags:
                # Простое приближение: если сообщение содержит ключевые слова — присваиваем тег
                for tag in common_tags:
                    if tag[1:] in m["text"].lower():
                        m["tags"] = tag
                        break

        # Сохраняем в CSV 
        with open(csv_filename, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["message_id", "text", "media_path", "tags", "date"])
            writer.writeheader()
            writer.writerows(all_messages)

        print(f"✅ Сохранено {len(all_messages)} сообщений в {csv_filename}")

if __name__ == "__main__":
    asyncio.run(fetch_messages())
