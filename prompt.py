import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

from typing import Union, Tuple, List
import os
import warnings

from models import TensorFiles, Prompt, Tokens, Embeddings
from helper import loadTensors, generateFilename
from DB.es import ES


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
            embedding_model_name: str = None,
            vd_type: str = "elastic-search"
        ):
        """
        :param model: model for inferencing
        :type model: torch.nn.Module
        :param tokenizer: 
        :type tokenizer: Union[str, AutoTokenizer]
        :param embedding_filename: Useful for storing embeddings when we are not using pretrained models for generating new ones
        :type embedding_filename: str
        """
        if not isinstance(model, nn.Module):
            raise Exception(f"model should be of instance nn.Module")
        if not (isinstance(tokenizer, str) or isinstance(tokenizer, AutoTokenizer)):
            raise ValueError(f"tokenizer type must be either str or AutoTokenizer, got {type(tokenizer)}")
        self.model: Model = model
        self.tokenizer = tokenizer if isinstance(tokenizer, AutoTokenizer) else AutoTokenizer.from_pretrained(tokenizer)
        self.embedding_filename: TensorFiles = embedding_filename
        if self.embedding_filename is not None:
            if os.path.splitext(embedding_filename)[-1] == "":
                raise ValueError("embedding_filename should have a valid file extension.")
            if not os.path.exists(self.embedding_filename):
                raise ValueError("embedding_filename does not exists.")
        else:
            self.embedding_filename = generateFilename(base_name="embeddings")
            warning_msg = f"embedding_filename is {embedding_filename}, creating a file with filename {self.embedding_filename}"
            warnings.warn(warning_msg)
        self.use_pretrained = use_pretrained
        if self.use_pretrained:
            if not isinstance(tokenizer, str) and embedding_model_name == None:
                raise ValueError(f"""to use a pre trained model for embedding generation, either provide tokenizer as a \n
                                string or embedding_model_name. Got: tokenizer type = {type(tokenizer)} \n
                                embedding_model_name = {embedding_model_name}""")
            else:
                if isinstance(tokenizer, str):
                    self.pretrained_model = AutoModel.from_pretrained(tokenizer) #this model is cached in 
                elif embedding_model_name != None:
                    if tokenizer.name_or_path != embedding_model_name:
                        raise ValueError(f"embedding model_name provided is not equal to the tokenizer, they are incompatible.\
                                        {embedding_model_name} != {tokenizer.name_or_path}")
                    self.pretrained_model = AutoModel.from_pretrained(embedding_model_name)
        
        # Database settings
        input_embeddings = self.pretrained_model.get_input_embeddings()
        self.embedding_dim = input_embeddings.embedding_dim
        self.vd_type = vd_type
        self.index_name = "prompt_embeddings"
        self.setUpVD()

    def cacheEmbeddings(self, input: Union[Prompt, Tokens, Embeddings]) -> None:
        """
        Caches the embeddings of a prompt or tokens or embeddings itself.

        :param prompt: A simple string
        :type prompt: Prompt
        :param Tokens: A list of integers aka tokens
        :type Tokens: List[int]
        :param Embeddings: A list of embeddings
        :type Embeddings: List[torch.Tensor]
        """
        if isinstance(input, Prompt) or isinstance(input, Tokens):
            embeddings, tokens = self.getEmbeddings(input=input)
        elif isinstance(input, Embeddings):
            embeddings = embeddings
            tokens = self.embedToTokens(tokens)
        # Store in a vector database
        embeddings = embeddings.embeddings
        if embeddings.ndim == 3:
            embeddings = embeddings.squeeze(0).tolist()
        for idx, token in enumerate(tokens):
            if token != 'CLS' and token != 'END':
                mapped_embeddings = {"prompt": self.tokenToPrompt(token=token).prompt, "embeddings": embeddings[idx]}
                self.vd.insertDocument(mapped_embeddings)
        return

    def tokensToPrompt(self, tokens: Tokens) -> Prompt: 
        """
        :param tokens: these are token ids
        :type tokens: List[int]
        """
        prompt: str = self.tokenizer.decode(tokens.tokens)
        return Prompt(prompt=prompt)

    def tokenToPrompt(self, token: int) -> Prompt:
        prompt: str = self.tokenizer.decode(token)
        return Prompt(prompt=prompt)

    def embedToTokens(self, embeddings: Embeddings) -> Tokens:
        """
        Docstring for embedToTokens

        :param embeddings: ...
        :type embeddings: Embeddings
        :return: ...
        :rtype: Tokens
        """
        raise NotImplementedError("")
    
    def embedToPrompt(self, embeddings: Embeddings) -> Prompt:
        """
        Docstring for embedToPrompt

        :param embeddings: ...
        :type embeddings: Embeddings
        :return: ...
        :rtype: Prompt
        """
        tokens: Tokens = self.embedToTokens(embeddings=embeddings)
        prompt: Prompt = self.tokensToPrompt(tokens=tokens)
        return prompt

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

    def getEmbeddings(self, input: Union[Prompt, Tokens]) -> Tuple[Embeddings, List]:
        """
        :param input: ...
        :type input: Union[Prompt, Tokens]
        """
        if isinstance(input, Prompt):
            tokens: Tokens = self.getTokens(input)
        embeddings: Embeddings = self.lookUpEmbeddings(tokens) if not self.use_pretrained \
            else Embeddings(embeddings=self.pretrained_model(tokens.tokens)["last_hidden_state"])
        return embeddings, tokens.tokens.squeeze(0).tolist()

    def lookUpEmbeddings(self, input: Tokens) -> Embeddings:
        tokens = input.tokens
        embeddings: Embeddings = loadTensors(self.embedding_filename) # load Tensors
        token_embeddings = embeddings[tokens]
        return Embeddings(embeddings=token_embeddings)

    def setUpVD(self):
        if self.vd_type == "elastic-search":
            self.vd = ES(index_name=self.index_name, embedding_dim=self.embedding_dim)