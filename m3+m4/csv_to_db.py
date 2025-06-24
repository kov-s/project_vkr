import csv
from datetime import datetime
from database.session import SessionLocal, engine
from database.models import Message
from sqlalchemy import inspect

def check_db():
    """Проверка существования таблицы"""
    inspector = inspect(engine)
    if 'messages' not in inspector.get_table_names():
        raise RuntimeError("Таблица messages не существует. Сначала запустите init_db.py")

def load_messages_from_csv(file_path):
    db = SessionLocal()
    try:
        with open(file_path, mode='r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            # Пропустить существующие ID
            existing_ids = {msg[0] for msg in db.query(Message.message_id).all()}
            added = 0
            
            for row in reader:
                try:
                    msg_id = int(row['message_id'])
                    if msg_id in existing_ids:
                        print(f"Сообщение ID {msg_id} уже существует, пропускаем")
                        continue
                        
                    message = Message(
                        message_id=msg_id,
                        text=row['text'],
                        media_path=row['media_path'] or None,
                        tags=row['tags'],
                        date=datetime.strptime(row['date'], '%Y-%m-%d %H:%M'),
                        topic=row['topic'] if row['topic'] and row['topic'] != '-1' else 'Без темы',
                        topic_probability=float(row['topic_probability']) if row['topic_probability'] else None
                    )
                    db.add(message)
                    added += 1
                    
                    # Периодический коммит
                    if added % 100 == 0:
                        db.commit()
                
                except Exception as e:
                    print(f"Ошибка в строке {reader.line_num}: {e}")
                    db.rollback()
                    continue
            
            db.commit()
            print(f"Успешно загружено {added} новых сообщений")
    
    except Exception as e:
        db.rollback()
        print(f"Критическая ошибка: {str(e)}")
    finally:
        db.close()
    
 
if __name__ == "__main__":
    load_messages_from_csv('messages.csv')