import torch
from transformers import GPT2Tokenizer, GPT2Model


class PolicyModel:
    def __init__(self, max_length=1024):
        self.max_length = max_length
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2Model.from_pretrained("gpt2")

    def get_tokenizer(self):
        return self.tokenizer

    def run_inference(self, inputs):
        # Run the data through the GPT-2 model
        outputs = self.model(**inputs)
        return outputs

    def to(self, device):
        self.model.to(device)
        return self

if __name__ == "__main__":
    # Example usage:
    data_definition = """
    {
      "metadata": {
        "result": 1.0/-1.0,
        "elapsed": 42627.0,
        ...
      },
      "sequence": {
        "state": [
          ...
        ],
        "action": [
          ...
        ],
        "option": [
          ...
        ]
      }
    }
    """
    model = PolicyModel()
    inputs = ...
    outputs = model.run_inference(inputs)
    print(outputs)