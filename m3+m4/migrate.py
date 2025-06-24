from database import Base, engine
from models import Message  # Убедитесь, что у вас есть models.py с классом Message

def init_db():
    # Создаем все таблицы
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully!")

if __name__ == "__main__":
    init_db()