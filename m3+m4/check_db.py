from database import SessionLocal
from database.models import Message
from sqlalchemy import text

def check_database():
    print("Проверка подключения к БД...")
    db = SessionLocal()
    try:
        # Проверка таблиц
        tables = db.execute(text("SELECT name FROM sqlite_master WHERE type='table';")).fetchall()
        print(f"Таблицы в БД: {tables}")

        # Проверка сообщений
        messages = db.query(Message).order_by(Message.date.desc()).limit(3).all()
        
        print("\nПервые 3 сообщения:")
        for msg in messages:
            print(f"- {msg.date}: {msg.text[:50]}...")
        
        print(f"\nВсего сообщений: {db.query(Message).count()}")
        
    except Exception as e:
        print(f"\nОШИБКА: {str(e)}")
    finally:
        db.close()
        print("\nПроверка завершена")

if __name__ == "__main__":
    check_database()