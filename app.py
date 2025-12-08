from fastapi import FastAPI

#routers
from apis.chat import router as chat_router

app = FastAPI()
app.include_router(chat_router)