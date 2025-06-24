import logging
import asyncio
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from dotenv import load_dotenv # <-- Обязательный импорт для чтения .env файла

# Загружаем переменные окружения из .env файла
# Это должно быть выполнено как можно раньше, чтобы API_ID/HASH были доступны
load_dotenv()

import backend.models as models # Ваши Pydantic модели
import backend.database as database
import backend.telethon_client as telethon_client
from telethon.errors import SessionPasswordNeededError, FloodWaitError, AuthBytesInvalidError, PhoneCodeInvalidError, PhoneCodeExpiredError

# --- Настройка логирования ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Глобальная переменная для блокировки состояния авторизации ---
# Это необходимо, чтобы избежать гонок состояний, когда несколько запросов
# пытаются изменить статус клиента Telethon одновременно.
AUTH_IN_PROGRESS_LOCK = asyncio.Lock()

# Глобальная переменная для контроля над парсингом
PARSING_TASK: Optional[asyncio.Task] = None
# Мьютекс для безопасного доступа к PARSING_TASK и его статусу
PARSING_LOCK = asyncio.Lock()


# --- События жизненного цикла приложения ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Приложение запускается...")
    # Инициализация базы данных
    await database.initialize_db()
    logger.info("База данных инициализирована.")
    # Инициализация клиента Telethon
    # Здесь client_initialized_successfully будет True/False в зависимости от результата
    # initialize_client устанавливает telethon_client.client
    await telethon_client.initialize_client()
    logger.info("TelegramClient успешно инициализирован при запуске (или предпринята попытка).")

    yield # Приложение готово к приему запросов

    logger.info("Приложение выключается...")
    # Остановка клиента Telethon при завершении работы
    if telethon_client.client is not None and await telethon_client.client.is_connected():
        await telethon_client.client.disconnect()
        logger.info("TelegramClient отключен.")
    # Отмена задачи парсинга при выключении
    global PARSING_TASK
    async with PARSING_LOCK: # Захватываем лок перед изменением PARSING_TASK
        if PARSING_TASK is not None and not PARSING_TASK.done():
            PARSING_TASK.cancel()
            try:
                await PARSING_TASK
            except asyncio.CancelledError:
                logger.info("Задача парсинга отменена при выключении.")
            PARSING_TASK = None # Очищаем задачу после отмены


app = FastAPI(lifespan=lifespan)

# --- Настройка статических файлов и шаблонов ---
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# --- Маршруты для веб-интерфейса ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# --- Маршруты API для авторизации ---
@app.get("/auth-status", response_model=models.AuthStatusResponse)
async def get_auth_status():
    async with AUTH_IN_PROGRESS_LOCK:
        logger.info("Запрос статуса авторизации.")
        # Проверяем, существует ли клиент и подключен ли он
        if telethon_client.client is not None and await telethon_client.client.is_connected():
            logger.info("Клиент Telethon подключен.")
            if await telethon_client.client.is_user_authorized():
                # Если пользователь авторизован, сбрасываем все флаги состояния
                telethon_client.phone_number = None
                telethon_client.code_sent = False
                telethon_client.password_needed = False
                logger.info("Пользователь авторизован.")
                return {"status": "authenticated", "message": "Клиент Telegram авторизован."}
            else:
                # Пользователь не авторизован, проверяем флаги стадий
                logger.info("Пользователь не авторизован. Проверяем стадии авторизации.")
                if telethon_client.password_needed:
                    logger.info("Требуется двухфакторная аутентификация.")
                    return {"status": "password_needed", "message": "Требуется двухфакторная аутентификация. Введите пароль."}
                elif telethon_client.code_sent:
                    logger.info("Код отправлен. Ожидание ввода кода.")
                    return {"status": "code_sent", "message": "Код отправлен. Ожидание ввода кода."}
                else:
                    logger.info("Ожидание номера телефона.")
                    return {"status": "unauthenticated", "message": "Клиент Telegram не авторизован. Введите номер телефона."}
        else:
            logger.info("Клиент Telethon не подключен или не инициализирован. Статус: unauthenticated.")
            # Если клиент не подключен, сбрасываем флаги на unauthenticated
            telethon_client.phone_number = None
            telethon_client.code_sent = False
            telethon_client.password_needed = False
            return {"status": "unauthenticated", "message": "Клиент Telegram не авторизован."}


@app.post("/send-code", response_model=models.AuthStatusResponse)
async def send_code_route(request_data: models.PhoneNumberRequest):
    async with AUTH_IN_PROGRESS_LOCK:
        # Убедимся, что клиент инициализирован и подключен
        if telethon_client.client is None:
            logger.info("Клиент Telethon не инициализирован. Попытка инициализации.")
            await telethon_client.initialize_client()
            if telethon_client.client is None: # Если после попытки инициализации client все еще None
                logger.error("Не удалось инициализировать клиент Telethon. Проверьте TELEGRAM_API_ID и TELEGRAM_API_HASH.")
                return JSONResponse(status_code=500, content={"status": "error", "message": "Не удалось инициализировать клиент Telegram. Проверьте конфигурацию сервера."})

        # Теперь client точно не None, но нужно убедиться, что он подключен
        if not await telethon_client.client.is_connected():
            logger.info("Клиент Telethon не подключен. Попытка подключиться.")
            await telethon_client.client.connect()
            if not await telethon_client.client.is_connected():
                logger.error("Не удалось подключить клиент Telethon.")
                return {"status": "error", "message": "Не удалось подключиться к Telegram. Проверьте интернет-соединение."}

        # Если уже авторизован, не отправляем код
        if await telethon_client.client.is_user_authorized():
            logger.info("Попытка отправить код, но пользователь уже авторизован.")
            return {"status": "authenticated", "message": "Вы уже авторизованы."}

        telethon_client.phone_number = request_data.phone_number
        try:
            # Отправляем запрос на получение кода
            await telethon_client.client.send_code_request(telethon_client.phone_number)
            telethon_client.code_sent = True
            telethon_client.password_needed = False # Сбрасываем флаг пароля при запросе нового кода
            logger.info(f"Код авторизации отправлен на номер: {telethon_client.phone_number}")
            return {"status": "success", "message": "Код отправлен. Проверьте Telegram."}
        except FloodWaitError as e:
            logger.error(f"FloodWaitError при отправке кода: {e}", exc_info=True)
            telethon_client.code_sent = False
            return {"status": "error", "message": f"Слишком много попыток. Пожалуйста, подождите {e.seconds} секунд."}
        except Exception as e:
            logger.error(f"Ошибка при отправке кода: {e}", exc_info=True)
            telethon_client.code_sent = False
            telethon_client.phone_number = None # Сбросить номер
            return {"status": "error", "message": f"Ошибка при отправке кода: {e}. Проверьте номер телефона или попробуйте позже."}


@app.post("/verify-code", response_model=models.AuthStatusResponse)
async def verify_code_route(request_data: models.CodeVerification):
    async with AUTH_IN_PROGRESS_LOCK:
        if telethon_client.client is None or not await telethon_client.client.is_connected():
            logger.warning("Попытка верификации кода, когда клиент не подключен или не инициализирован.")
            telethon_client.code_sent = False
            return {"status": "unauthenticated", "message": "Ошибка: клиент Telegram не подключен. Попробуйте начать заново."}

        if not telethon_client.phone_number:
            logger.warning("Попытка верификации кода без номера телефона.")
            telethon_client.code_sent = False
            return {"status": "unauthenticated", "message": "Ошибка: номер телефона не был введен. Начните авторизацию заново."}

        try:
            # Попытка войти с кодом
            await telethon_client.client.sign_in(phone=telethon_client.phone_number, code=request_data.code)
            
            # Если sign_in успешно и не выбросил исключение SessionPasswordNeededError
            telethon_client.phone_number = None
            telethon_client.code_sent = False
            telethon_client.password_needed = False
            logger.info("Код подтвержден. Пользователь успешно авторизован.")
            return {"status": "authenticated", "message": "Код успешно подтвержден. Вы авторизованы!"}

        except SessionPasswordNeededError:
            # Если требуется 2FA
            telethon_client.password_needed = True # Устанавливаем флаг, что нужен пароль
            telethon_client.code_sent = False # Сбрасываем флаг кода, так как код уже принят
            logger.warning("Требуется пароль двухфакторной аутентификации.")
            return {"status": "password_needed", "message": "Требуется двухфакторная аутентификация. Введите пароль."}
        except PhoneCodeInvalidError:
            logger.warning("Попытка верификации с неверным кодом.")
            telethon_client.code_sent = True # Оставляем code_sent True, чтобы пользователь мог ввести код снова
            telethon_client.password_needed = False
            return {"status": "error", "message": "Неверный код. Пожалуйста, проверьте и введите снова."}
        except PhoneCodeExpiredError:
            logger.warning("Попытка верификации с просроченным кодом.")
            telethon_client.code_sent = False # Код просрочен, нужно запросить новый
            telethon_client.phone_number = None
            telethon_client.password_needed = False
            return {"status": "error", "message": "Код просрочен. Пожалуйста, запросите новый код."}
        except AuthBytesInvalidError:
            logger.error("Ошибка авторизации: неверные байты авторизации. Возможно, конфликт сессии.", exc_info=True)
            telethon_client.code_sent = False
            telethon_client.phone_number = None
            telethon_client.password_needed = False
            return {"status": "error", "message": "Ошибка авторизации. Пожалуйста, попробуйте начать авторизацию заново."}
        except FloodWaitError as e:
            logger.error(f"FloodWaitError при верификации кода: {e}", exc_info=True)
            return {"status": "error", "message": f"Слишком много попыток. Пожалуйста, подождите {e.seconds} секунд."}
        except Exception as e:
            logger.error(f"Неизвестная ошибка при подтверждении кода: {e}", exc_info=True)
            telethon_client.code_sent = False # В случае неизвестной ошибки лучше сбросить
            telethon_client.phone_number = None
            telethon_client.password_needed = False
            return {"status": "error", "message": f"Произошла неизвестная ошибка: {e}. Попробуйте начать заново."}


@app.post("/verify-password", response_model=models.AuthStatusResponse)
async def verify_password_route(request_data: models.PasswordVerification):
    async with AUTH_IN_PROGRESS_LOCK:
        if telethon_client.client is None or not await telethon_client.client.is_connected():
            logger.warning("Попытка верификации пароля, когда клиент не подключен или не инициализирован.")
            telethon_client.password_needed = False
            return {"status": "unauthenticated", "message": "Ошибка: клиент Telegram не подключен. Попробуйте начать заново."}

        try:
            # Попытка войти с паролем
            await telethon_client.client.sign_in(password=request_data.password)
            
            # Если sign_in успешно, пользователь авторизован
            telethon_client.phone_number = None
            telethon_client.code_sent = False
            telethon_client.password_needed = False
            logger.info("Пароль подтвержден. Пользователь успешно авторизован.")
            return {"status": "authenticated", "message": "Пароль успешно подтвержден. Вы авторизованы!"}
        except Exception as e:
            logger.error(f"Ошибка при подтверждении пароля: {e}", exc_info=True)
            telethon_client.password_needed = True # Пароль неверный, остаемся в состоянии password_needed
            if "The password is invalid" in str(e):
                 return {"status": "error", "message": "Неверный пароль. Пожалуйста, попробуйте еще раз."}
            return {"status": "error", "message": f"Ошибка подтверждения пароля: {e}. Попробуйте еще раз."}


@app.post("/logout", response_model=models.AuthStatusResponse)
async def logout_route():
    async with AUTH_IN_PROGRESS_LOCK:
        if telethon_client.client is not None and await telethon_client.client.is_connected():
            try:
                await telethon_client.client.log_out()
                logger.info("Пользователь успешно вышел из Telegram.")
                telethon_client.phone_number = None
                telethon_client.code_sent = False
                telethon_client.password_needed = False
                return {"status": "unauthenticated", "message": "Вы успешно вышли из Telegram."}
            except Exception as e:
                logger.error(f"Ошибка при выходе из Telegram: {e}", exc_info=True)
                return {"status": "error", "message": f"Ошибка при выходе: {e}. Попробуйте позже."}
        else:
            logger.info("Попытка выхода, но клиент Telethon не подключен.")
            return {"status": "unauthenticated", "message": "Клиент Telegram не был подключен."}


# --- Маршруты API для каналов ---
@app.post("/add-channel", response_model=models.AddChannelResponse)
async def add_channel(request_data: models.ChannelLink):
    if telethon_client.client is None or not await telethon_client.client.is_user_authorized():
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Клиент Telegram не авторизован.")

    link = request_data.link
    try:
        # Проверяем, существует ли канал по ссылке/username
        entity = await telethon_client.client.get_entity(link)
        channel_id = entity.id
        channel_title = entity.title if hasattr(entity, 'title') else link

        # Добавляем канал в базу данных
        await database.add_channel(channel_id=channel_id, link=link, title=channel_title)
        logger.info(f"Канал {link} (ID: {channel_id}) успешно добавлен.")
        return {"status": "success", "message": f"Канал '{channel_title}' успешно добавлен."}
    except Exception as e:
        logger.error(f"Ошибка при добавлении канала {link}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Не удалось добавить канал: {e}. Проверьте ссылку.")


@app.get("/channels", response_model=List[models.Channel])
async def get_channels():
    if telethon_client.client is None or not await telethon_client.client.is_user_authorized():
        # Если не авторизован, возвращаем пустой список и логируем предупреждение
        logger.warning("Попытка получить каналы, когда клиент Telegram не авторизован.")
        return [] # Возвращаем пустой список, а не ошибку, чтобы фронтенд мог отобразить пустую таблицу

    channels_data = await database.get_all_channels()
    return [models.Channel(**channel) for channel in channels_data]


@app.post("/delete-channel", response_model=models.DeleteChannelResponse)
async def delete_channel_route(request_data: models.ChannelId):
    if telethon_client.client is None or not await telethon_client.client.is_user_authorized():
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Клиент Telegram не авторизован.")

    channel_id = request_data.channel_id
    try:
        await database.delete_channel(channel_id)
        logger.info(f"Канал с ID {channel_id} успешно удален.")
        return {"status": "success", "message": f"Канал успешно удален."}
    except Exception as e:
        logger.error(f"Ошибка при удалении канала {channel_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Не удалось удалить канал: {e}.")


# --- Маршруты API для парсинга ---
@app.get("/parsing-status", response_model=models.ParsingStatusResponse)
async def get_parsing_status():
    async with PARSING_LOCK:
        is_active = PARSING_TASK is not None and not PARSING_TASK.done()
        return {"is_parsing_active": is_active}


@app.post("/start-parsing", response_model=models.ParsingControlResponse)
async def start_parsing_route():
    global PARSING_TASK
    async with PARSING_LOCK:
        if PARSING_TASK is not None and not PARSING_TASK.done():
            return {"status": "error", "message": "Парсинг уже активен."}

        if telethon_client.client is None or not await telethon_client.client.is_user_authorized():
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Клиент Telegram не авторизован для начала парсинга.")

        # Получаем каналы из базы данных
        channels_to_parse = await database.get_all_channels()
        if not channels_to_parse:
            return {"status": "error", "message": "Нет добавленных каналов для парсинга."}

        logger.info("Запуск задачи парсинга.")
        # Создаем новую задачу парсинга
        PARSING_TASK = asyncio.create_task(parse_channels_periodically(channels_to_parse))
        return {"status": "success", "message": "Парсинг запущен."}


@app.post("/stop-parsing", response_model=models.ParsingControlResponse)
async def stop_parsing_route():
    global PARSING_TASK
    async with PARSING_LOCK:
        if PARSING_TASK is not None and not PARSING_TASK.done():
            PARSING_TASK.cancel()
            try:
                await PARSING_TASK # Ожидаем завершения отмены
            except asyncio.CancelledError:
                logger.info("Задача парсинга успешно отменена.")
                PARSING_TASK = None
                return {"status": "success", "message": "Парсинг остановлен."}
            except Exception as e:
                logger.error(f"Ошибка при остановке задачи парсинга: {e}", exc_info=True)
                return {"status": "error", "message": f"Ошибка при остановке парсинга: {e}"}
        else:
            return {"status": "error", "message": "Парсинг не был активен."}


# --- Функция парсинга (запускается в фоновой задаче) ---
async def parse_channels_periodically(channels: List[models.Channel]): # Указываем тип как List[models.Channel]
    # Настраиваем FloodWait для избежания бана
    flood_wait_seconds = 0
    
    while True:
        try:
            # Если есть задержка FloodWait, ждем ее
            if flood_wait_seconds > 0:
                logger.info(f"Ожидание из-за FloodWait: {flood_wait_seconds} секунд.")
                await asyncio.sleep(flood_wait_seconds)
                flood_wait_seconds = 0 # Сброс после ожидания

            if telethon_client.client is None or not await telethon_client.client.is_user_authorized():
                logger.warning("Парсинг остановлен: клиент Telegram не авторизован или не инициализирован.")
                global PARSING_TASK
                async with PARSING_LOCK:
                    if PARSING_TASK:
                        PARSING_TASK = None # Сбросить задачу
                break # Выйти из цикла парсинга

            # Обновляем список каналов из базы данных перед каждым циклом,
            # чтобы учитывать добавленные/удаленные каналы
            current_channels_data = await database.get_all_channels()
            current_channels = [models.Channel(**c) for c in current_channels_data]


            for channel_data in current_channels:
                try:
                    channel_id = channel_data.channel_id # Используем .channel_id, т.к. это models.Channel
                    last_message_id = channel_data.last_message_id # Используем .last_message_id

                    logger.info(f"Парсинг канала ID: {channel_id}, начиная с сообщения ID: {last_message_id}")

                    # Получаем entity канала, чтобы убедиться, что это существующий канал
                    entity = await telethon_client.client.get_entity(channel_id)
                    
                    # Получаем новые сообщения
                    # limit - количество сообщений за один запрос
                    # min_id - парсить сообщения, которые имеют ID больше min_id
                    messages = await telethon_client.client.iter_messages(entity, min_id=last_message_id, limit=100).collect()

                    if messages:
                        new_last_message_id = last_message_id
                        for message in messages:
                            if message.id > new_last_message_id:
                                new_last_message_id = message.id
                            
                            logger.info(f"Новое сообщение в '{entity.title}' (ID: {message.id})")
                            # Сохранение сообщения в базу данных
                            await database.save_message(
                                channel_id=channel_id,
                                message_id=message.id,
                                text=message.text,
                                date=message.date.isoformat()
                            )
                        # Обновляем last_message_id в базе данных для этого канала
                        if new_last_message_id > last_message_id:
                            await database.update_last_message_id(channel_id, new_last_message_id)
                            logger.info(f"Обновлен last_message_id для канала {entity.title} до {new_last_message_id}")
                    else:
                        logger.info(f"В канале '{entity.title}' нет новых сообщений с ID > {last_message_id}.")

                except FloodWaitError as e:
                    logger.warning(f"FloodWaitError в цикле парсинга для канала {channel_data.link} (ID: {channel_id}): {e.seconds} секунд.")
                    flood_wait_seconds = max(flood_wait_seconds, e.seconds + 1) # Обновляем задержку
                except Exception as e:
                    logger.error(f"Ошибка при парсинге канала {channel_data.link} (ID: {channel_id}): {e}", exc_info=True)
                    # Можно добавить логику для пропуска проблемного канала или повторной попытки

            # Задержка между полными циклами парсинга всех каналов
            await asyncio.sleep(60) # Парсить каждые 60 секунд (или настроить)

        except asyncio.CancelledError:
            logger.info("Задача парсинга отменена.")
            break # Выход из бесконечного цикла

        except Exception as e:
            logger.error(f"Глобальная ошибка в задаче парсинга: {e}", exc_info=True)
            # В случае критической ошибки можно решить остановить парсинг или продолжить после задержки
            await asyncio.sleep(30) # Небольшая задержка перед следующей попыткой