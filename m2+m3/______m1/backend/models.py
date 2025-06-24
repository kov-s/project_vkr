# backend/models.py
from pydantic import BaseModel
from typing import Optional, List

# --- Модели для авторизации (используются в main.py) ---

class PhoneNumberRequest(BaseModel):
    """Модель для запроса номера телефона при авторизации."""
    phone_number: str

class CodeVerification(BaseModel):
    """Модель для подтверждения кода авторизации."""
    code: str

class PasswordVerification(BaseModel):
    """Модель для подтверждения пароля двухфакторной аутентификации."""
    password: str

class AuthStatusResponse(BaseModel):
    """Модель ответа для статуса авторизации."""
    # 'authenticated', 'unauthenticated', 'code_sent', 'password_needed', 'error', 'success'
    status: str
    message: str # Сообщение для пользователя

# --- Модели для управления каналами (используются в main.py и database.py) ---

class ChannelLink(BaseModel):
    """Модель для добавления канала по ссылке/username."""
    link: str

class ChannelId(BaseModel):
    """Модель для удаления канала по ID."""
    channel_id: int

class Channel(BaseModel):
    """Модель для представления канала в базе данных и API."""
    id: Optional[int] = None # ID записи в вашей базе данных SQLite (автоматически генерируется)
    channel_id: int # Уникальный ID канала в Telegram
    link: str # Ссылка или username канала
    title: Optional[str] = None # Название канала
    last_message_id: Optional[int] = 0 # ID последнего сообщения, которое было успешно спарсено

class AddChannelResponse(BaseModel):
    """Модель ответа на запрос добавления канала."""
    status: str # 'success' или 'error'
    message: str # Сообщение о результате добавления

class DeleteChannelResponse(BaseModel):
    """Модель ответа на запрос удаления канала."""
    status: str # 'success' или 'error'
    message: str # Сообщение о результате удаления

# --- Модели для управления парсингом (используются в main.py) ---

class ParsingStatusResponse(BaseModel):
    """Модель ответа для статуса парсинга."""
    is_parsing_active: bool # True, если парсинг активен, False в противном случае

class ParsingControlResponse(BaseModel):
    """Модель ответа на запросы запуска/остановки парсинга."""
    status: str # 'success' или 'error'
    message: str # Сообщение о результате операции

# --- Модели для сохранения спарсенных сообщений (используются в database.py и main.py) ---

class MessageModel(BaseModel):
    """Модель для спарсенного сообщения."""
    channel_id: int # ID канала Telegram, к которому относится сообщение
    message_id: int # Уникальный ID сообщения в рамках канала
    text: Optional[str] = None # Текст сообщения
    date: str # Дата и время сообщения в формате ISO 8601
    # Добавьте другие поля, если вы решите их сохранять