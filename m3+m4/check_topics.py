from database import SessionLocal, engine  # Добавляем импорт engine
from database.models import Message
from sqlalchemy import distinct, func
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_database():
    """Проверка существования таблицы messages"""
    from sqlalchemy import inspect
    inspector = inspect(engine)
    if 'messages' not in inspector.get_table_names():
        raise RuntimeError("Таблица 'messages' не найдена в базе данных")
    logger.info("Таблица 'messages' существует")

def check_topics():
    db = SessionLocal()
    try:
        # Проверяем существование таблицы
        check_database()
        
        # Получаем уникальные темы
        topics = db.query(distinct(Message.topic)).filter(
            Message.topic.isnot(None)
        ).all()
        
        logger.info(f"Найдено {len(topics)} уникальных тем:")
        for i, topic in enumerate(topics[:10], 1):  # Выводим первые 10 тем
            logger.info(f"{i}. {topic[0]}")
        
        # Полная статистика по темам
        stats = db.query(
            Message.topic,
            func.count(Message.message_id).label('count')
        ).group_by(Message.topic).order_by(func.count(Message.message_id).desc()).all()
        
        logger.info("\nСтатистика по темам:")
        for topic, count in stats:
            logger.info(f"- {topic or 'Без темы'}: {count} сообщений")
            
        return [t[0] for t in topics]
        
    except Exception as e:
        logger.error(f"Ошибка при проверке: {str(e)}")
        return []
    finally:
        db.close()

if __name__ == "__main__":
    check_topics()