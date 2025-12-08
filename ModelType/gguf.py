import os

from llama_cpp import Llama
from torch import cuda

from models import Prompt, Tokens


class GGUF:
    """
        For 
            - loading a gguf model
            - inferencing
            - others I don't know might explore
    """
    def __init__(
            self, 
            model_path: str,
            ctx: int,
            device: str = "cpu",
        ):
        """
            :param ctx: token length of the input + model's response 
            :type ctx: int
        """
        if not os.path.exists(model_path):
            raise
        
        _, ext = os.path.splitext(model_path)
        if ext != ".gguf":
            raise
        self.model_path = model_path
        self.ctx = ctx
        
        if device.lower() == "gpu":
            if not cuda.is_available():
                raise ValueError("device provided cuda, found none on current machine")
        self.device = device

        n_gpu_layers = self.select_n_gpu_layers()
        self.model = Llama(
                    model_path=model_path,
                    n_gpu_layers=n_gpu_layers,
                    n_ctx=4096,
                    verbose=False
                )

    def select_n_gpu_layers(self):
        print("Finding optimal n_gpu_layers")
        n_gpu_layers = 0
        model = None
        while True:
            try:
                model = Llama(
                        model_path=self.model_path,
                        n_gpu_layers=n_gpu_layers,
                        n_ctx=4096,
                        verbose=False
                    )
            except Exception as e:
                n_gpu_layers -= 1
                del model
                raise e
            n_gpu_layers += 1
    
        print(f"Found n_gpu_layers = {n_gpu_layers}")
        return n_gpu_layers

    def infer(self, input: Prompt | Tokens):
        response = self.model.create_completion(
            prompt=input.prompt if isinstance(input, Prompt) else input.Tokens,
            max_tokens=None
        )
        return response["choices"][0]["text"]
    

if __name__ == "__main__":
    ...