from abc import ABC, abstractmethod
from functools import wraps

class Db(ABC):
    def __init__(self):
        ...

    @abstractmethod
    def setUp(self):
        pass

    @abstractmethod
    def insertDocuments(self, documents):
        pass

    @abstractmethod
    def insertDocument(self, documents) -> bool:
        pass

    @abstractmethod
    def deleteDocument(self, index_name: str, id: str) -> bool:
        pass

    @abstractmethod
    def deleteIndex(self, index_name: str):
        pass

    # @abstractmethod
    # def query_validator(self, *args, **kwargs):
    #     pass

    # def _query_validator(self, func):
    #     @wraps(func)
    #     def validate(self, *args, **kwargs):
    #         self.query_validator(*args, **kwargs)
    #         return func(*args, **kwargs)
    #     return validate

    # @_query_validator
    @abstractmethod
    def search(self, search_query: dict, *args, **kwargs):
        pass

    def idGenerator(self) -> str:
        import uuid
        return str(uuid.uuid4())