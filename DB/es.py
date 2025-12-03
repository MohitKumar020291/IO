import os
from typing import List, Dict, Union

from dotenv import load_dotenv
from elasticsearch import Elasticsearch

from DB.db import Db


# https://github.com/elastic/start-local
# Problem with this class is this ES object is assosciated with a single self.index_name
class ES(Db):
    def __init__(
            self,
            index_name: str,
            embedding_dim: int
        ):
        load_dotenv()
        self.environment = os.getenv("environment")
        self.index_name = index_name
        self.embedding_dim = embedding_dim

        self.username = os.getenv("es_username")
        self.password = os.getenv("es_password")

        self.setUp()

    def setUp(self) -> None:
        if self.environment == "development":
            self.es_url = "http://localhost:9200"
            self.es = Elasticsearch(self.es_url, basic_auth=(self.username, self.password))

            mapping = {
                "properties": {
                    "prompt": {"type": "text"},
                    "embeddings": {
                        "type": "dense_vector",
                        "dims": self.embedding_dim
                    }
                }
            }

            if not self.es.indices.exists(index=self.index_name):
                self.es.indices.create(index=self.index_name, body={"mappings": mapping})

        elif self.environment == "production":
            return
        
        return

    def insertDocuments(self, documents: Union[List[Dict], Dict]):
        if isinstance(documents, list):
            for _, document in documents:
                self.insertDocument(document)
        elif isinstance(documents, dict):
            self.insertDocument(documents)
        else:
            raise TypeError(f"documents should be of type list[dict] or dict, got {type(documents)}")
        
    def insertDocument(self, document: Dict):
        _id = super().idGenerator()
        self.es.index(index=self.index_name, id=_id, document=document)
        self.es.indices.refresh(index=self.index_name)

    def deleteIndex(self) -> True:
        self.es.indices.delete(index=self.index_name, ignore=[400, 404])
        return True