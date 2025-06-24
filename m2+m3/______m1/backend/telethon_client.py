import os
import asyncio
import logging
from typing import Optional 
from telethon.sync import TelegramClient
from telethon.errors import SessionPasswordNeededError, FloodWaitError, AuthBytesInvalidError, PhoneCodeInvalidError, PhoneCodeExpiredError

logger = logging.getLogger(__name__)

# Глобальные переменные для хранения состояния клиента Telethon
# Важно: client должен быть None, если он не инициализирован
client: Optional[TelegramClient] = None
phone_number: Optional[str] = None
code_sent: bool = False
password_needed: bool = False # Флаг для двухфакторной аутентификации

#  Константы 
# Получаем переменные окружения. Важно, чтобы load_dotenv() был вызван в main.py до этого импорта.
API_ID = os.getenv('TELEGRAM_API_ID')
API_HASH = os.getenv('TELEGRAM_API_HASH')
SESSION_NAME = 'telegram_session' # Имя файла сессии для Telethon

#  Инициализация клиента Telethon 
async def initialize_client():
    # Объявляем client как global, потому что мы можем переприсвоить его значение (например, client = None)
    global client 

    # Если клиент уже существует и подключен, нет нужды его переинициализировать
    if client is not None and await client.is_connected():
        logger.info("TelegramClient уже инициализирован и подключен.")
        return

    # Проверка на наличие API_ID и API_HASH (они читаются из переменных окружения)
    # Если load_dotenv() не был вызван или .env файл пуст/неправильный, они будут None
    if API_ID is None or API_HASH is None:
        logger.error("Переменные окружения TELEGRAM_API_ID или TELEGRAM_API_HASH не установлены. Клиент не будет инициализирован.")
        client = None # Убеждаемся, что client = None, если API_ID/HASH отсутствуют
        return

    # Пробуем преобразовать API_ID в int. Если не удается, сбрасываем client.
    try:
        # Убедимся, что API_ID, который мы получили из os.getenv, это строка перед преобразованием
        if isinstance(API_ID, str): 
            api_id_int = int(API_ID)
        else: # Если по какой-то причине он уже не строка (например, None), просто используем его
            api_id_int = API_ID
    except (ValueError, TypeError):
        logger.error(f"TELEGRAM_API_ID '{os.getenv('TELEGRAM_API_ID')}' не является корректным числом.")
        client = None # Сбросить клиент, если ID некорректный
        return

    # Создаем клиента Telethon
    # Используем api_id_int, который уже гарантированно является числом (или None)
    client = TelegramClient(SESSION_NAME, api_id_int, API_HASH)
    logger.info(f"TelegramClient создан с сессией: {SESSION_NAME}")

    try:
        # Подключаемся к Telegram
        await client.connect()
        logger.info("TelegramClient подключен.")

        # Проверяем, авторизован ли пользователь (если есть сохраненная сессия)
        if await client.is_user_authorized():
            logger.info("Пользователь уже авторизован.")
            # Здесь НЕТ необходимости в "global" для phone_number, code_sent, password_needed,
            # так как мы просто ИЗМЕНЯЕМ их значения, а не переприсваиваем их на что-то другое.
            # Python понимает, что это ссылки на глобальные переменные модуля.
            phone_number = None
            code_sent = False
            password_needed = False
        else:
            logger.info("Пользователь не авторизован. Ожидание запроса на авторизацию.")
            # Здесь НЕТ необходимости в "global" для code_sent, password_needed.
            code_sent = False
            password_needed = False

    except Exception as e:
        logger.error(f"Ошибка при инициализации/подключении TelegramClient: {e}", exc_info=True)
        # Важно: если подключение не удалось, client должен быть None, чтобы избежать ошибок
        # при последующих попытках использовать его.
        client = None