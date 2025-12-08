import os
from typing import Tuple, Union
import time

import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from chat_model import Model
from models import Prompt, Embeddings
from helper import read_yaml
from Performance.helper import sort_argsort, find_pop_idxs, do_pop_idxs


model_for_testing = nn.Linear(in_features=10, out_features=5)
sentence_embed_model_name="all-MiniLM-L6-v2"

class perfTestingModel(Model):
    def __init__(self, device="cpu"):
        self.device = device
        super().__init__(
                model=model_for_testing,              
                sentence_embed_model_name=sentence_embed_model_name, 
                device=device
            )
        self.current_dir = os.path.dirname(__file__)

    def generatetokenSizeVsRetrievalAndGenerationTime(self, save=False):
        """
        Docstring for GeneratetokenSizeVsRetrievalAndGenerationTime

        :param save: if save is true we saves the reports as an image for visualization
        :type save: str
        :return: a list containing tuple of time taken for searching embedding vs generating embedding
        :rtype: list[]
        """

        # base prompt and sub prompts with varying token sizes
        # Cache base prompt
        # retrieve a sub prompt embedding using base prompts <-- measure time1
        # generate a sub prompt's embedding <-- measure time2
        # delete base prompt
        cacheVsGen: dict = read_yaml(os.path.join(self.current_dir, "cacheVsGen.yaml"))
        base_prompt: Prompt = Prompt(prompt=cacheVsGen.get("base_prompt"))
        reports = list()
        id = super().cacheEmbeddings(input=base_prompt)

        sub_prompts_keys = list(cacheVsGen.keys())
        sub_prompts_keys.remove('base_prompt')
        for spk in sub_prompts_keys:
            sub_prompt_prompt = cacheVsGen.get(spk)
            sub_prompt: Prompt = Prompt(prompt=sub_prompt_prompt)
            sub_prompt_tokens: Tokens = super().getTokens(sub_prompt)
            retrieve_time, generate_time = self.promptVsRAndG(prompt=sub_prompt)
            reports.append((retrieve_time, generate_time, len(sub_prompt_tokens.tokens.squeeze(0))))

        if not self.vd.deleteDocument(id=id):
            raise Exception("base prompt has not been deleted successfully.")
        
        print("reports\n", reports)

        if save:
            token_sizes, retrieve_times, generate_times = [], [], []
            for t1, t2, token_size in reports:
                token_sizes.append(token_size)
                retrieve_times.append(t1)
                generate_times.append(t2)

            #####################CLEANING##########################
            retrieve_times = np.array(retrieve_times)
            generate_times = np.array(generate_times)
            token_sizes = np.array(token_sizes)
            token_sizes_sort = np.argsort(token_sizes)

            token_sizes, retrieve_times, generate_times =\
                sort_argsort(retrieve_times, generate_times, token_sizes, sort_idxs=token_sizes_sort)

            token_diff_thresh = 5
            pop_indexes = find_pop_idxs(token_sizes, threshold=token_diff_thresh)

            retrieve_times, generate_times, token_sizes =\
                do_pop_idxs(retrieve_times, generate_times, token_sizes, pop_idxs=pop_indexes)
            ###############################################

            # change to a helper function
            plt.plot(token_sizes, retrieve_times, label="Retrieval Time")
            plt.plot(token_sizes, generate_times, label="Generation Time")
            plt.xlabel("token size")
            plt.ylabel("time")
            plt.title(f"Token Size Vs Retrieval And Generation Time on {self.device}")
            plt.legend()

            public_dir = os.path.join(self.current_dir, "public")
            if not os.path.exists(public_dir):
                os.mkdir(public_dir)
            filepath = os.path.join(public_dir, "tokenSizeVsRetrievalAndGenerationTime.png")
            plt.savefig(filepath)

    def search_prompt_in_vd(
            self,
            input: Union[Prompt, Embeddings], 
            return_time: bool=True, 
            return_result: bool=False, 
            match_on: str="prompt",
            size: int = 1,
        ) -> float:

        start_time = time.time()
        if match_on == "prompt":
            search_body = {
                "query": {
                    "match": {"prompt": input.prompt}
                },
                "_source": {
                    "includes": ["prompt", "embeddings"]
                }
            }
        else:
            search_body = {
                "_source": {
                    "includes": ["prompt", "embeddings"]
                    },
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'embeddings') + 1.0",
                            "params": {
                                "query_vector": input.embeddings.tolist()
                            }
                        }
                    }
                }
            }
        result = self.vd.search(search_body=search_body)
        end_time = time.time()
        time_taken = end_time - start_time
        if return_result:
            if return_time:
                return time_taken, result
            else:
                return result
        return time_taken

    def generate_prompt_embeddings(self, prompt: Prompt) -> float:
        start_time = time.time()
        _ = super(perfTestingModel, self).getEmbeddings(input=prompt)
        end_time = time.time()
        return end_time - start_time

    def promptVsRAndG(self, prompt: Prompt) -> Tuple[float]:
        """
        This function returns time for generating embeddings vs retreiving through base prompt

        :param prompt: A prompt
        :type prompt: Prompt
        """

        # IDK why I need this warm up run, elastic search have some problem!
        for _ in range(3):
            self.search_prompt_in_vd(input=prompt)
            self.generate_prompt_embeddings(prompt=prompt)

        retrieve_time = self.search_prompt_in_vd(prompt=prompt)
        generate_time = self.generate_prompt_embeddings(prompt=prompt)
        return retrieve_time, generate_time

    def SemanticSearch(self, match_on="prompt"):
        """
        some random prompts are also added to the vector database along with base prompt
        :param match_on: if embeddings then we look embeddings similarity
        """
        prompts: dict = read_yaml(os.path.join(os.path.dirname(__file__), "TenRandomPrompt.yaml"))
        ids = []
        for prompt in list(prompts.values())[0]:
            ids.append(super().cacheEmbeddings(input=Prompt(prompt=prompt)))
        sub_prompts: dict = read_yaml(os.path.join(os.path.dirname(__file__), "cacheVsGen.yaml"))

        results_text = []
        for prompt_type, sub_prompt in sub_prompts.items():
            if prompt_type == "base_prompt":
                continue
            input_ = Prompt(prompt=sub_prompt) if match_on=="prompt" \
                else super().getEmbeddings(input=Prompt(prompt=sub_prompt))
            result = self.search_prompt_in_vd(input=input_, return_time=False, return_result=True, match_on=match_on)
            results_text.append((sub_prompt, result['hits']['hits'][0]['_source']['prompt']))
        
        self.vd.deleteDocuments(ids=ids)

        return results_text


# this file is actually a report
if __name__ == "__main__":
    ptm = perfTestingModel()
    ptm.SemanticSearch(match_on="embeddings")