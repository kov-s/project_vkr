import os
import sys

# Добавляем родительскую директорию в sys.path
# Это позволяет Python находить 'project_sum' как пакет
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# 'project_sum' — это название корневого пакета
from project_sum.database.session import engine, Base, SQLALCHEMY_DATABASE_URL
from project_sum.database.models import Message # Импортируем Message, чтобы убедиться, что его метаданные загружены

def create_database_tables():
    # Скорректированный путь для файла базы данных
    # SQLALCHEMY_DATABASE_URL = "sqlite:///./summarizer.db" означает, что summarizer.db находится в текущей рабочей директории
    db_file_name = SQLALCHEMY_DATABASE_URL.replace("sqlite:///./", "")
    # Формируем полный путь к файлу относительно текущего скрипта
    db_file_path = os.path.join(current_dir, db_file_name)

    print(f"Попытка удалить существующий файл базы данных: {db_file_path}")
    if os.path.exists(db_file_path):
        os.remove(db_file_path)
        print(f"Файл {db_file_path} успешно удален")
    else:
        print(f"Файл базы данных не найден по пути {db_file_path}. Продолжаем создание новых таблиц.")

    print("Создание новых таблиц базы данных...")
    Base.metadata.create_all(engine) # Создаем все таблицы, определенные в метаданных
    print("Таблицы базы данных успешно созданы!")

if __name__ == "__main__":
    create_database_tables()