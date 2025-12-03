import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

from typing import Union

from models import TensorFiles, Prompt, Tokens, Embeddings
from DB.es import ES


# take prompt
prompting = False
if prompting:
    prompt = input("Enter your prompt: ")


class Model:
    def __init__(
            self, 
            model: nn.Module,
            sentence_embed_model_name: str = None,
            vd_type: str = "elastic-search"
        ):
        """
        :param model: LLM for inferencing, do not know how to use it now.
        :type model: torch.nn.Module
        :param tokenizer: 
        :type tokenizer: Union[str, AutoTokenizer]
        :param embedding_filename: Useful for storing embeddings when we are not using pretrained models for generating new ones
        :type embedding_filename: str
        """
        if not isinstance(model, nn.Module):
            raise Exception(f"model should be of instance nn.Module")

        self.model: Model = model

        try:
            self.sentence_embed_model = SentenceTransformer(sentence_embed_model_name)
            self.tokenizer = self.sentence_embed_model.tokenizer
        except Exception as e:
            raise e

        # Database settings
        self.embedding_dim = self.sentence_embed_model.get_sentence_embedding_dimension()
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
        if isinstance(input, Prompt):
            embeddings = self.getEmbeddings(input=input)
            prompt: str = input.prompt
        elif isinstance(input, Tokens):
            embeddings = self.getEmbeddings(input=input)
            prompt: str = self.tokensToPrompt(input).prompt
        elif isinstance(input, Embeddings):
            embeddings = embeddings
            prompt: str = self.embedToPrompt(embeddings=input).prompt #
        # Store in a vector database
        embeddings = embeddings.embeddings.tolist()
        mapped_embeddings = {"prompt": prompt, "embeddings": embeddings}
        self.vd.insertDocument(mapped_embeddings)
        return

    def tokensToPrompt(self, tokens: Tokens) -> Prompt: 
        """
        :param tokens: these are token ids
        :type tokens: List[int]
        """
        prompt: str = self.tokenizer.decode(tokens.tokens)
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

    def cacheComputes(): ...

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
        if isinstance(input, Tokens):
            input: Prompt = self.tokensToPrompt(input)
        print(input.prompt)
        embeddings: Embeddings = Embeddings(embeddings=self.sentence_embed_model.encode(input.prompt))
        return embeddings

    def setUpVD(self):
        if self.vd_type == "elastic-search":
            self.vd = ES(index_name=self.index_name, embedding_dim=self.embedding_dim)