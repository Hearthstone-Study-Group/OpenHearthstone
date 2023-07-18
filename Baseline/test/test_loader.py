import unittest
from data.loader import DataLoader
from transformers import GPT2Tokenizer
import numpy as np


class LoaderTestCase(unittest.TestCase):
    def test_loading(self):
        folder_path = "../storage/v0.1"
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        data_loader = DataLoader(folder_path, tokenizer)
        data_loader.max_length = 1e9
        training_data_property = data_loader.check_data_loader()
        source_data = np.array(training_data_property)
        print()
        print("state length max", np.max(source_data[:, 0]), "min", np.min(source_data[:, 0]))
        print("action length max", np.max(source_data[:, 1]), "min", np.min(source_data[:, 1]))
        print("option length max", np.max(source_data[:, 2]), "min", np.min(source_data[:, 2]))


if __name__ == '__main__':
    unittest.main()
