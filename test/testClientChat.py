import unittest

from chat import ClientChat

class Test(unittest.TestCase):
    def setUp(self):
        self.client_chat = ClientChat(index_name="fofo")

    def test_setUpVd(self):
        mapping = {
            "properties": {
                "prompt": {"type": "text"},
                "context": {"type": "text"},
                "prompt_embeddings": {
                    "type": "dense_vector",
                    "dims": 384
                }
            }
        }
        try:
            vd = self.client_chat.setUpVD(mapping=mapping)
        except Exception as e:
            print(e)

        try:
            self.client_chat.vd.deleteIndex() # deleting is important here
            self.client_chat.deleteChat()
        except Exception as e:
            print(e)

    def test_infer(self):
        ...