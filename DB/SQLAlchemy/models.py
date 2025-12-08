from sqlalchemy import Column, Integer, JSON
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Chats(Base):
    __tablename__ = "chats"
    id = Column(Integer, primary_key=True, index=True)
    chats = Column(JSON)