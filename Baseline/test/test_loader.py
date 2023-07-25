import unittest
from data.loader import DataLoader
from data.transition import TransitionLoader
from transformers import AutoTokenizer as TOK
import numpy as np


class LoaderTestCase(unittest.TestCase):
    def test_loading(self):
        folder_path = "../storage/v0.1"
        tokenizer = TOK.from_pretrained("google/long-t5-tglobal-base")
        data_loader = DataLoader(folder_path, tokenizer)
        data_loader.max_length = 10240
        training_data_property = data_loader.check_data_loader()
        source_data = np.array(training_data_property)
        print()
        print("state length max", np.max(source_data[:, 0]), "min", np.min(source_data[:, 0]))
        print("action length max", np.max(source_data[:, 1]), "min", np.min(source_data[:, 1]))
        print("option length max", np.max(source_data[:, 2]), "min", np.min(source_data[:, 2]))


    def test_transition(self):
        folder_path = "../storage/v0.1"
        tokenizer = TOK.from_pretrained("google/long-t5-tglobal-base")
        data_loader = TransitionLoader(folder_path, tokenizer, difference=True)
        data_loader.max_length = 10240
        training_data_property = data_loader.check_data_loader()
        source_data = np.array(training_data_property)
        print()
        print("state length max", np.max(source_data[:, 0]), "min", np.min(source_data[:, 0]))
        print("action length max", np.max(source_data[:, 1]), "min", np.min(source_data[:, 1]))
        print("next action length max", np.max(source_data[:, 2]), "min", np.min(source_data[:, 2]))



if __name__ == '__main__':
    unittest.main()
