import torch
from transformers import AutoModelForSeq2SeqLM as MOD, AutoTokenizer as TOK


class PredictorModel:
    def __init__(self, max_length=1024, pretrained="google/long-t5-local-base"):
        self.max_length = max_length
        self.tokenizer = TOK.from_pretrained(pretrained)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = MOD.from_pretrained(pretrained)
        self.model.train()

    def get_tokenizer(self):
        return self.tokenizer

    def run_inference(self, inputs):
        # Run the data through the GPT-2 model
        outputs = self.model(**inputs)
        return outputs

    def run_prediction(self, inputs):
        outputs = self.model.generate(**inputs, max_length=self.max_length, num_beams=5)
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
