import os
import threading
from typing import Tuple
import time

import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from prompt import Model
from models import Prompt, Tokens
from helper import read_yaml


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

            ###############################################
            retrieve_times = np.array(retrieve_times)
            generate_times = np.array(generate_times)
            token_sizes = np.array(token_sizes)
            token_sizes_sort = np.argsort(token_sizes)
            token_sizes = token_sizes[token_sizes_sort]
            retrieve_times = retrieve_times[token_sizes_sort]
            generate_times = generate_times[token_sizes_sort]

            token_diff_thresh = 5
            pop_indexes = []
            prev_preserved = 0
            for idx, (i, j) in enumerate(zip(token_sizes[:-1], token_sizes[1:])):
                if j - i <= token_diff_thresh and j - token_sizes[prev_preserved] <= token_diff_thresh:
                    pop_indexes.append(idx+1)
                else:
                    prev_preserved = idx+1

            print(pop_indexes)

            k = 0
            for pop_idx in pop_indexes:
                retrieve_times = np.delete(retrieve_times, pop_idx - k)
                generate_times = np.delete(generate_times, pop_idx - k)
                token_sizes = np.delete(token_sizes, pop_idx - k)
                k += 1

            print(retrieve_times)
            print(generate_times)
            print(token_sizes)
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


    def promptVsRAndG(self, prompt: Prompt) -> Tuple[float]:
        """
        This function returns time for generating embeddings vs retreiving through base prompt

        :param prompt: A prompt
        :type prompt: Prompt
        """
        # time-retrieval
        def search_prompt_in_vd(prompt: Prompt) -> float:
            start_time = time.time()
            search_body = {
                "query": {
                    "match": {
                        "prompt": prompt.prompt
                    }
                },
                "_source": {
                    "includes": ["prompt", "embeddings"]
                }
            }
            _ = self.vd.search(search_body=search_body)
            end_time = time.time()
            return end_time - start_time

        # time-generation
        def generate_prompt_embeddings(prompt: Prompt) -> float:
            start_time = time.time()
            _ = super(perfTestingModel, self).getEmbeddings(input=prompt)
            end_time = time.time()
            return end_time - start_time
        
        # t1 = threading.Thread(target=search_prompt_in_vd, args=(prompt,))
        # t2 = threading.Thread(target=generate_prompt_embeddings, args=(prompt,))

        # t1.start()
        # t2.start()

        # time1 = t1.join()
        # time2 = t2.join()

        # IDK why I need this warm up run, elastic search have some problem!
        for _ in range(3):
            search_prompt_in_vd(prompt=prompt)
            generate_prompt_embeddings(prompt=prompt)

        retrieve_time = search_prompt_in_vd(prompt=prompt)
        generate_time = generate_prompt_embeddings(prompt=prompt)
        return retrieve_time, generate_time


# this file is actually a report
if __name__ == "__main__":
    ptm = perfTestingModel()
    ptm.generatetokenSizeVsRetrievalAndGenerationTime(save=True)