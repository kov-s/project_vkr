from sqlalchemy import create_engine, inspect

engine = create_engine("sqlite:///./messages.db")
inspector = inspect(engine)
print(inspector.get_table_names())