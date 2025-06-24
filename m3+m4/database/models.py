# project_sum/database/models.py
from sqlalchemy import Column, Integer, String, DateTime, Float
from .session import Base # Ensure this import is correct

class Message(Base):
    __tablename__ = 'messages'
    message_id = Column(Integer, primary_key=True, index=True)
    text = Column(String)
    date = Column(DateTime)
    topic = Column(Integer, index=True)
    topic_name = Column(String, default='Без темы') # <--- THIS LINE IS ESSENTIAL
    topic_probability = Column(Float)
    tags = Column(String, default='')
    media_path = Column(String, nullable=True)

    def __repr__(self):
        return f"<Message(id={self.message_id}, topic='{self.topic_name}', date='{self.date}')>"