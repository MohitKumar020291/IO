import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

from pydantic import validate_call
from typing import List, Union
import pickle
import os

from models import TensorFiles, Prompt, Tokens, Embeddings
from helper import loadTensors


# take prompt
prompting = False
if prompting:
    prompt = input("Enter your prompt: ")

# Infer
class Model:
    def __init__(
            self, model: nn.Module, 
            tokenizer: Union[str, AutoTokenizer], 
            embedding_filename: str = None, 
            use_pretrained: bool = False,
            embedding_model_name: str = None
        ):
        """
        :param model: model for inferencing
        :type model: torch.nn.Module
        :param tokenizer: 
        :type tokenizer: Union[str, AutoTokenizer]
        :param embedding_filename: This could also be the location of the embedding_filename
        :type embedding_filename: str
        """
        if not isinstance(model, nn.Module):
            raise Exception(f"model should be of instance nn.Module")
        if not (isinstance(tokenizer, str) or isinstance(tokenizer, AutoTokenizer)):
            raise ValueError(f"tokenizer type must be either str or AutoTokenizer, got {type(tokenizer)}")
        self.model: Model = model
        self.tokenizer = tokenizer if isinstance(tokenizer, AutoTokenizer) else AutoTokenizer.from_pretrained(tokenizer)
        self.embedding_filename: TensorFiles = embedding_filename
        if os.path.splitext(embedding_filename)[-1] == "":
            raise ValueError("embedding_filename should have a valid file extension.")
        if not os.path.exists(self.embedding_filename):
            raise ValueError("embedding_filename does not exists.")
        self.use_pretrained = use_pretrained
        if self.use_pretrained:
            if not isinstance(tokenizer, str) or embedding_model_name == None:
                raise ValueError(f"to use a pre trained model for embedding generation, either provide tokenizer as a \
                                string or embedding_model_name. Got: tokenizer type = {type(tokenizer)} \
                                embedding_model_name = {embedding_model_name}")
            else:
                if isinstance(tokenizer, str):
                    self.pretrained_model = AutoModel.from_pretrained(tokenizer)
                elif embedding_model_name != None:
                    if tokenizer.name_or_path != embedding_model_name:
                        raise ValueError(f"embedding model_name provided is not equal to the tokenizer, they are incompatible.\
                                        {embedding_model_name} != {tokenizer.name_or_path}")
                    self.pretrained_model = AutoModel.from_pretrained(embedding_model_name)
        

    def cacheEmbeddings(self, input: Union[Prompt, Tokens, Embeddings]) -> None:
        """
        Caches the embeddings of a prompt or tokens or embeddings itself.
        For real why am I not storing them into the Vector Database????

        :param prompt: A simple string
        :type prompt: Prompt
        :param Tokens: A list of integers aka tokens
        :type Tokens: List[int]
        :param Embeddings: A list of embeddings
        :type Embeddings: List[torch.Tensor]
        """
        embeddings: Embeddings = self.getEmbeddings(input=input)
        prompt: str = input.Prompt if isinstance(input, Prompt) else isinstance(input, Tokens)
        serialized_embeddings = {"prompt": prompt, "embeddings": embeddings}
        with open(self.embedding_filename) as file:
            pickle.dump(obj=serialized_embeddings, file=file)
        return

    def decodeTokenize(self, tokens: Tokens) -> Prompt: 
        """
        :param tokens: these are token ids
        :type tokens: List[int]
        """
        prompt = self.tokenizer.decode(tokens.tokens)
        return Prompt(prompt=prompt)

    @staticmethod
    def cacheComputes():
        ...

    def getTokens(self, input: Prompt) -> Tokens:
        """
        :param prompt: ...
        :type prompt: Prompt
        """
        prompt = input.prompt
        tokens: torch.Tensor = self.tokenizer(prompt, return_tensors='pt')["input_ids"]
        return Tokens(tokens=tokens)

    def getEmbeddings(self, input: Union[Prompt, Tokens]) -> Embeddings:
        """
        :param input: ...
        :type input: Union[Prompt, Tokens]
        """
        print("type", type(input))
        if isinstance(input, Prompt):
            tokens = self.getTokens(input)
        embeddings: Embeddings = self.lookUpEmbeddings(tokens) if not self.use_pretrained \
            else Embeddings(embeddings=self.pretrained_model(tokens))
        return embeddings

    def lookUpEmbeddings(self, input: Tokens) -> Embeddings:
        tokens = input.tokens
        embeddings: Embeddings = loadTensors(self.embedding_filename) # load Tensors
        # load embeddings from tokens
        token_embeddings = embeddings[tokens]
        return Embeddings(embeddings=token_embeddings)
