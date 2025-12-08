from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import delete

from DB.SQLAlchemy.get_db import get_db
from DB.SQLAlchemy.models import Chats


class Item(BaseModel):
    item: str

router = APIRouter(prefix="/chat")

@router.get("/test_get")
def test_get():
    return {"working": True}

@router.post("/test_post")
def test_post(item: Item):
    return {"working": True, "item": item.item}

@router.post("/add_chat")
def add_chat(db: Session = Depends(get_db)):
    new_chat = Chats(chats={"chats": {}})
    db.add(new_chat)
    db.commit()
    db.refresh(new_chat)
    chat_idx = new_chat.id
    return {"chat_id": chat_idx}

@router.post("/delete_chat")
def delete_chat(chat_id: int, db: Session = Depends(get_db)):
    try:
        chat_2b_deleted = db.query(Chats).filter_by(id=chat_id).first()
    except Exception as e:
        # raise e
        pass
    if chat_2b_deleted:
        db.delete(chat_2b_deleted)
        db.commit()
        return {"found": True, "deleted": True}
    else:
        return {"found": False}