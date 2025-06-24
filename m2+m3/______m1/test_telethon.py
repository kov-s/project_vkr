import os
import asyncio
import logging
from dotenv import load_dotenv
from telethon.sync import TelegramClient
from telethon.sessions import StringSession
from telethon.errors import SessionPasswordNeededError, FloodWaitError

# --- Настройка логирования ---
# Устанавливаем базовую конфигурацию логирования для вывода в консоль
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])

# Проверка версии Telethon в самом начале
try:
    import telethon
    logging.info(f"Тест: Модуль Telethon найден. Версия: {telethon.__version__}")
    logging.info(f"Тест: Расположение Telethon: {telethon.__file__}")
except ImportError:
    logging.error("Тест: КРИТИЧЕСКАЯ ОШИБКА: Модуль Telethon НЕ найден! Пожалуйста, установите его.")
except Exception as e:
    logging.error(f"Тест: Ошибка при проверке версии Telethon: {e}", exc_info=True)

# Загрузка переменных окружения из .env файла
load_dotenv()

# Получение API ID и API HASH из переменных окружения
api_id_str = os.getenv('API_ID')
api_hash_val = os.getenv('API_HASH')

logging.info(f"Тест: Необработанный API_ID из .env: '{api_id_str}' (Тип: {type(api_id_str)})")
logging.info(f"Тест: Необработанный API_HASH из .env: '{api_hash_val}' (Тип: {type(api_hash_val)})")

# Преобразование API_ID в целочисленный тип
try:
    api_id = int(api_id_str) if api_id_str else 0
    logging.info(f"Тест: Преобразованный API_ID: {api_id} (Тип: {type(api_id)})")
except ValueError as e:
    logging.error(f"Тест: ОШИБКА: Не удалось преобразовать API_ID '{api_id_str}' в целое число: {e}", exc_info=True)
    api_id = 0 # Устанавливаем в 0, чтобы предотвратить дальнейшие ошибки

# Проверка API_HASH
api_hash = api_hash_val if api_hash_val else ''
logging.info(f"Тест: Использованный API_ID: {api_id}")
logging.info(f"Тест: Использованный API_HASH: {'*' * len(api_hash) if api_hash else 'НЕТ'} (Длина: {len(api_hash)})")


async def run_test():
    """
    Запускает изолированный тест инициализации и подключения TelegramClient.
    Корректно адаптирован для Telethon 1.x API.
    """
    client = None
    if not api_id or not api_hash:
        logging.error("Тест: API_ID или API_HASH отсутствуют или недействительны. Невозможно инициализировать TelegramClient.")
        return

    try:
        logging.info("Тест: Попытка создания экземпляра TelegramClient...")
        
        # Создание экземпляра TelegramClient
        client = TelegramClient(StringSession(), api_id, api_hash)
        logging.info(f"Тест: Экземпляр TelegramClient создан: {client is not None}")

        if client is not None:
            logging.info("Тест: Попытка подключения TelegramClient...")
            
            # Попытка установить соединение
            try:
                await client.connect() # <-- Правильно: С 'await'
                logging.info(f"Тест: Попытка подключения завершена успешно.")
            except Exception as connect_e:
                logging.error(f"Тест: ОШИБКА: Ошибка во время client.connect(): {connect_e}", exc_info=True)
                return # Завершаем, если не удалось подключиться

            # Для Telethon 1.x: client.is_connected() возвращает bool напрямую
            # Не используем 'await' перед ним.
            is_connected_status = client.is_connected() # <-- Правильно: БЕЗ 'await'
            logging.info(f"Тест: Статус подключения клиента (из is_connected()): {is_connected_status} (Тип: {type(is_connected_status)})")

            if is_connected_status:
                logging.info("Тест: Клиент подключен.")
                
                # Проверка авторизации пользователя
                if not await client.is_user_authorized(): # <-- Правильно: С 'await'
                    logging.warning("Тест: Пользователь не авторизован. Пожалуйста, введите номер телефона для авторизации.")
                    phone_number = input("Пожалуйста, введите ваш номер телефона (например, +1234567890): ")
                    try:
                        await client.send_code_request(phone_number) # <-- Правильно: С 'await'
                        logging.info("Тест: Запрос кода отправлен. Пожалуйста, проверьте ваше приложение Telegram.")
                        code = input("Пожалуйста, введите полученный код: ")
                        await client.sign_in(phone_number, code) # <-- Правильно: С 'await'
                        logging.info("Тест: Успешный вход в систему.")
                    except SessionPasswordNeededError:
                        logging.warning("Тест: Включена двухфакторная аутентификация. Пожалуйста, введите ваш пароль.")
                        password = input("Пожалуйста, введите ваш пароль: ")
                        await client.sign_in(password=password) # <-- Правильно: С 'await'
                        logging.info("Тест: Успешный вход в систему с паролем.")
                    except FloodWaitError as fwe:
                        logging.error(f"Тест: Ошибка FloodWaitError во время авторизации: Пожалуйста, подождите {fwe.seconds} секунд.", exc_info=True)
                        client = None # Сброс клиента
                    except Exception as auth_e:
                        logging.error(f"Тест: Ошибка во время авторизации: {auth_e}", exc_info=True)
                        client = None # Сброс клиента
                else:
                    logging.info("Тест: Клиент авторизован.")

                # Если успешно подключен и авторизован, можно сохранить session_string
                # Проверяем is_user_authorized() еще раз, так как она может быть false после ошибок авторизации.
                if client and await client.is_user_authorized(): # <-- Правильно: С 'await'
                    session_string = client.session.save()
                    logging.info(f"Тест: Строка сессии сохранена (обрезано): {session_string[:30]}... (Полная длина: {len(session_string)})")
                else:
                    logging.warning("Тест: Строка сессии не сохранена, так как пользователь не авторизован или клиент не готов к сохранению.")

            else:
                logging.error("Тест: Клиент НЕ подключен после вызова connect().")

        else:
            logging.error("Тест: Конструктор TelegramClient вернул None. Не удалось создать объект клиента.")

    except FloodWaitError as fwe:
        logging.error(f"Тест: Произошла ошибка FloodWaitError: Пожалуйста, подождите {fwe.seconds} секунд. Ошибка: {fwe}", exc_info=True)
    except Exception as e:
        logging.error(f"Тест: Произошло необработанное исключение во время теста клиента Telethon: {e}", exc_info=True)
    finally:
        if client and hasattr(client, 'is_connected'):
            # Для Telethon 1.x: client.is_connected() возвращает bool напрямую
            if client.is_connected(): # <-- Правильно: БЕЗ 'await'
                logging.info("Тест FINALLY: Клиент подключен. Попытка отключения.")
                await client.disconnect() # <-- Правильно: С 'await'
                logging.info("Тест FINALLY: Клиент отключен.")
            else:
                logging.info("Тест FINALLY: Клиент не подключен. Отключение не требуется.")
        elif client:
            logging.info("Тест FINALLY: Объект клиента существует, но не имеет метода is_connected. Отключение не производилось.")
        else:
            logging.info("Тест FINALLY: Объект клиента не был должным образом инициализирован. Отключение не производилось.")


# Точка входа в скрипт
if __name__ == '__main__':
    asyncio.run(run_test())