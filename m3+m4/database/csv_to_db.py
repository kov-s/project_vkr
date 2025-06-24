import csv
from datetime import datetime
from database import SessionLocal
from models import Message
import logging

logging.basicConfig(
    filename='csv_import.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_messages_from_csv(file_path):
    """Загрузка сообщений из CSV в базу данных"""
    db = SessionLocal()
    try:
        with open(file_path, mode='r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            for row in reader:
                # Преобразование данных
                message = Message(
                    message_id=int(row['message_id']),
                    text=row['text'],
                    media_path=row['media_path'] if row['media_path'] else None,
                    tags=row['tags'],
                    date=datetime.strptime(row['date'], '%Y-%m-%d %H:%M:%S'),  # Формат может потребовать корректировки
                    topic=row['topic'] if row['topic'] else None,
                    topic_probability=float(row['topic_probability']) if row['topic_probability'] else None
                )
                exists = db.query(Message).filter(Message.message_id == int(row['message_id'])).first()
                if exists:
                    print(f"Сообщение {row['message_id']} уже существует, пропускаем")
                    continue
                db.add(message)
            
            db.commit()
            print(f"Успешно загружено {reader.line_num - 1} сообщений")

    
    except Exception as e:
        db.rollback()
        print(f"Ошибка при загрузке: {str(e)}")
    finally:
        db.close()

if __name__ == "__main__":
    load_messages_from_csv('messages_sum.csv')

    from database import SessionLocal
    from models import Message
    db = SessionLocal()
    count = db.query(Message).count()
    print(f"Всего сообщений в БД после загрузки: {count}")
    db.close()