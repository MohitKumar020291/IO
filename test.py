import unittest

import torch.nn as nn

from prompt import Model
from models import Prompt

class Test(unittest.TestCase):
    def setUp(self): 
        self.model = Model(
            nn.Linear(in_features=10, out_features=5),
            sentence_embed_model_name="all-MiniLM-L6-v2"
        )
        self.prompt = Prompt(prompt="This is a prompt")

    def testCacheEmbedding(self):
        self.model.cacheEmbeddings(self.prompt)