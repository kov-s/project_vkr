# init_db.py
from database.base import Base
from database.session import engine

def create_tables():
    Base.metadata.create_all(bind=engine)
    print("Таблицы успешно созданы")

if __name__ == "__main__":
    create_tables()