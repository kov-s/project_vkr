import aiosqlite
import logging

DATABASE_NAME = "telegram_data.db"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def initialize_db():
    async with aiosqlite.connect(DATABASE_NAME) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS channels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                link TEXT NOT NULL UNIQUE,
                last_message_id INTEGER DEFAULT 0
            )
        """)
        await db.commit()
    logger.info("Таблица 'channels' проверена/создана.")

async def add_channel(link: str):
    async with aiosqlite.connect(DATABASE_NAME) as db:
        try:
            await db.execute("INSERT INTO channels (link) VALUES (?)", (link,))
            await db.commit()
            logger.info(f"Канал '{link}' добавлен в базу данных.")
        except aiosqlite.IntegrityError:
            logger.warning(f"Канал '{link}' уже существует в базе данных. Пропускаем.")

async def get_all_channels():
    async with aiosqlite.connect(DATABASE_NAME) as db:
        cursor = await db.execute("SELECT id, link, last_message_id FROM channels")
        channels_data = await cursor.fetchall()
        return [{"id": row[0], "link": row[1], "last_message_id": row[2]} for row in channels_data]

async def delete_channel(channel_id: int):
    async with aiosqlite.connect(DATABASE_NAME) as db:
        await db.execute("DELETE FROM channels WHERE id = ?", (channel_id,))
        await db.commit()
        logger.info(f"Канал с ID {channel_id} удален из базы данных.")


async def update_channel_last_message_id(channel_link: str, last_message_id: int):
    async with aiosqlite.connect(DATABASE_NAME) as db:
        await db.execute(
            "UPDATE channels SET last_message_id = ? WHERE link = ?",
            (last_message_id, channel_link)
        )
        await db.commit()
        logger.info(f"Обновлен last_message_id для канала '{channel_link}' до {last_message_id}.")

async def close_db_connection():
    
    logger.info("Закрытие соединения с базой данных (если применимо).")