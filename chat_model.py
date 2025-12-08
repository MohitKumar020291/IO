import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

from typing import Union

from models import Prompt, Tokens, Embeddings
from ModelType.gguf import GGUF


# take prompt
prompting = False
if prompting:
    prompt = input("Enter your prompt: ")


class Model:
    def __init__(
            self, 
            model: Union[GGUF], #add other types if needed
            sentence_embed_model_name: str = None,
            device: str = "cpu"
        ):
        """
        :param model: LLM for inferencing, do not know how to use it now.
        :type model: torch.nn.Module
        :param embedding_filename: Useful for storing embeddings when we are not using pretrained models for generating new ones
        :type embedding_filename: str
        """

        self.model = model

        try:
            self.sentence_embed_model = SentenceTransformer(sentence_embed_model_name, device=device)
        except Exception as e:
            raise e

        # Database settings
        self.embedding_dim = self.sentence_embed_model.get_sentence_embedding_dimension()
        self.index_name = "prompt_embeddings"

    def cacheEmbeddings(self, input: Union[Prompt, Tokens, Embeddings]) -> str:
        """
        Caches the embeddings of a prompt or tokens or embeddings itself.

        :param prompt: A simple string
        :type prompt: Prompt
        :param Tokens: A list of integers aka tokens
        :type Tokens: List[int]
        :param Embeddings: A list of embeddings
        :type Embeddings: List[torch.Tensor]
        :return: returns id of the document just added
        :rtype: str
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
        id = self.vd.insertDocument(mapped_embeddings)
        return id

    def tokensToPrompt(self, tokens: Tokens) -> Prompt: 
        """
        :param tokens: these are token ids
        :type tokens: List[int]
        """
        prompt: str = self.model.model.detokenize(tokens.tokens)
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
        tokens: torch.Tensor = torch.Tensor(self.model.model.tokenize(prompt))
        return Tokens(tokens=tokens)

    def getEmbeddings(self, input: Union[Prompt, Tokens]) -> Embeddings:
        """
        :param input: ...
        :type input: Union[Prompt, Tokens]
        """
        if isinstance(input, Tokens):
            input: Prompt = self.tokensToPrompt(input)
        embeddings: Embeddings = Embeddings(embeddings=self.sentence_embed_model.encode(input.prompt))
        return embeddings

    def infer(self, input: Prompt | Tokens):
        return self.model.infer(input=input)