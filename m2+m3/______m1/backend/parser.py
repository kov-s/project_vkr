import asyncio
import logging
import os
import csv
from telethon import TelegramClient 
from telethon.tl.types import MessageMediaPhoto, MessageMediaDocument
from datetime import datetime
from typing import Optional

# Импорт модулей
from . import database # Для работы с базой данных
from . import models # модели внутри парсера

logger = logging.getLogger(__name__)

#  Глобальные переменные для управления парсингом 
_parsing_active = False
_parsing_task: Optional[asyncio.Task] = None
_telegram_client: Optional[TelegramClient] = None 
_channels_to_parse = [] 

#  НОВАЯ ФУНКЦИЯ ДЛЯ УСТАНОВКИ КЛИЕНТА И КАНАЛОВ 
def set_client_and_channels(client: TelegramClient, channels: list):
    global _telegram_client, _channels_to_parse
    _telegram_client = client
    _channels_to_parse = channels
    logger.info("Клиент Telegram и каналы установлены для парсера.")

#  Остальные функции парсера (start_parsing, stop_parsing, is_parsing_active, parse_channel) 

async def start_parsing():
    global _parsing_active, _parsing_task

    if _parsing_active:
        logger.warning("Парсинг уже активен. Запуск повторно игнорируется.")
        return

    if not _telegram_client:
        logger.error("TelegramClient не установлен в парсере. Невозможно начать парсинг.")
        return

    if not _channels_to_parse:
        logger.warning("Список каналов для парсинга пуст. Парсинг не будет запущен.")
        return

    _parsing_active = True
    logger.info("Парсинг запущен.")

    # Создаем директорию для медиа, если она не существует
    media_dir = "telegram_data_media"
    os.makedirs(media_dir, exist_ok=True)

    while _parsing_active:
        for channel_info in _channels_to_parse:
            channel_link = channel_info.link
            channel_db_id = channel_info.id
            last_message_id = channel_info.last_message_id if channel_info.last_message_id else 0

            logger.info(f"Парсинг канала: {channel_link} (последнее сообщение ID: {last_message_id})")

            try:
                # Получаем информацию о канале по его ссылке
                entity = await _telegram_client.get_entity(channel_link)

                # Проверяем, авторизован ли клиент (на всякий случай)
                if not await _telegram_client.is_user_authorized():
                    logger.error("Клиент Telegram не авторизован. Парсинг остановлен.")
                    await stop_parsing()
                    return

                new_last_message_id = last_message_id

                # Итерация по сообщениям, начиная с последнего ID
                
                async for message in _telegram_client.iter_messages(entity, min_id=last_message_id, reverse=True):
                    # Проверяем, не было ли команды на остановку парсинга
                    if not _parsing_active:
                        logger.info("Парсинг остановлен по команде.")
                        return

                    message_id = message.id
                    message_text = message.text if message.text else ""
                    message_date = message.date.strftime("%Y-%m-%d %H:%M:%S")

                    logger.info(f"  Сообщение ID: {message_id}, Дата: {message_date}, Текст: {message_text[:50]}...")

                    # Записываем сообщение в CSV
                    channel_slug = entity.username if entity.username else str(entity.id)
                    channel_folder = os.path.join(media_dir, channel_slug)
                    os.makedirs(channel_folder, exist_ok=True)
                    csv_file_path = os.path.join(channel_folder, "messages.csv")

                    file_exists = os.path.exists(csv_file_path)
                    with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
                        fieldnames = ['message_id', 'date', 'text', 'sender_id', 'media_path', 'media_type']
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                        if not file_exists:
                            writer.writeheader()

                        media_path = ""
                        media_type = ""

                        # Скачиваем медиа
                        if message.media:
                            if isinstance(message.media, MessageMediaPhoto):
                                media_type = "photo"
                            elif isinstance(message.media, MessageMediaDocument):
                                media_type = "document"
                                # Можно добавить проверку на тип документа (видео, аудио и т.д.)
                            else:
                                media_type = "other_media"

                            try:
                                # Сохраняем медиа в подпапке канала
                                download_path = os.path.join(channel_folder, media_type)
                                os.makedirs(download_path, exist_ok=True)
                                # Telethon скачивает файл в папку, возвращает полный путь
                                # base_name = message.file.name if message.file else f"{message_id}.{media_type}"
                                # full_download_path = os.path.join(download_path, base_name)

                                # message.download() возвращает путь относительно текущей рабочей директории или full path
                                downloaded_file = await message.download_media(file=download_path)
                                if downloaded_file:
                                    # Если downloaded_file - это путь, сохраняем его относительно project root
                                    media_path = os.path.relpath(downloaded_file, start=os.getcwd())
                                    logger.info(f"    Медиа сохранено: {media_path}")
                            except Exception as media_err:
                                logger.error(f"    Ошибка при скачивании медиа для сообщения {message_id}: {media_err}")
                                media_path = f"ERROR: {media_err}"

                        writer.writerow({
                            'message_id': message_id,
                            'date': message_date,
                            'text': message_text,
                            'sender_id': message.sender_id,
                            'media_path': media_path,
                            'media_type': media_type
                        })

                        new_last_message_id = message_id # Обновляем ID последнего сообщения

                # Обновляем last_message_id в базе данных
                if new_last_message_id > last_message_id:
                    await database.update_last_message_id(channel_db_id, new_last_message_id)
                    logger.info(f"Обновлен last_message_id для канала {channel_link} на {new_last_message_id}")

            except Exception as e:
                logger.error(f"Критическая ошибка при парсинге канала {channel_link}: {e}", exc_info=True)

            if not _parsing_active:
                break # Выходим из цикла по каналам, если парсинг остановлен

        if _parsing_active:
            logger.info("Ожидание 60 секунд перед следующим циклом парсинга...")
            await asyncio.sleep(60) # Пауза между циклами парсинга

async def stop_parsing():
    global _parsing_active, _parsing_task
    if _parsing_active:
        _parsing_active = False
        if _parsing_task:
            _parsing_task.cancel()
            try:
                await _parsing_task
            except asyncio.CancelledError:
                logger.info("Задача парсинга отменена.")
            finally:
                _parsing_task = None
        logger.info("Парсинг остановлен.")
    else:
        logger.info("Парсинг не активен, нечего останавливать.")

def is_parsing_active():
    return _parsing_active