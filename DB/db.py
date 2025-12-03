from abc import ABC, abstractmethod

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
    def insertDocument(self, documents):
        pass

    @abstractmethod
    def deleteIndex(self, index_name: str):
        pass

    def idGenerator(self) -> str:
        import uuid
        return str(uuid.uuid4())