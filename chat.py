# The chat is using prompts mutliple times
## Core changes are
### Setup vd here
### Changes in prompt - Change in Performance folder
### new index with context and chat index

import requests
import os
from dotenv import load_dotenv

from DB.es import ES
from chat_model import Model
from models import Prompt

load_dotenv()


class ClientChat:
    def __init__(
            self,
            model_path: str,
            index_name: str,
            vd_type: str = "elastic-search",
            mapping: dict = None,
            sentence_embed_model_name: str = None,
            device: str = "cpu"
        ):
        self.vd_type = vd_type
        self.index_name = index_name
        self.base_url = "http://localhost:8000/" if os.getenv("environment", None) else os.getenv("production_url", None)
        self.sentence_embed_model_name = sentence_embed_model_name
        self.model_path = model_path

        self.getEmbedDim()
        self.setUpVD(mapping=mapping)
        self.setUpModel()

    def getEmbedDim(self) -> None:
        # response = requests.post(url=self.base_url, data={""})
        self.embedding_dim = 384

    def setUpVD(self, mapping: dict):
        if self.vd_type == "elastic-search":
            self.vd = ES(index_name=self.index_name, embedding_dim=self.embedding_dim, mapping=mapping)
            self.chat_id = self.vd.index_name
            # Push and save into the database
            request_url = self.base_url + "/chat/add_chat"
            response = requests.post(url=request_url)
            self.chat_idx = response.json()["chat_id"]
            return self.vd
        
    def deleteChat(self) -> bool:
        """
            deletes a chat from main database
        """
        request_url = self.base_url + "/chat/delete_chat"
        response = requests.post(url=request_url, data={"chat": self.chat_id})
        return response.json()["deleted"]

    def setUpModel(self):
        self.model = Model(
            model=self.model,
            sentence_embed_model_name=self.sentence_embed_model_name,
            device=self.device
        )

    def infer(self, input: Prompt):
        return self.model.infer(input=input)