# -*- coding: utf-8 -*-

"""Test the encoder fine-tuning module"""

import unittest

import torch
from datasets import Dataset
from transformers import BertTokenizer

from itk_transformer_nlp.encoder_cola import tokenize_single_sent_dataset


class ColaTest(unittest.TestCase):
    """Test functions related to fine-tuning on HuCoLA"""

    @classmethod
    def setUpClass(cls) -> None:
        """Fixture setup: create a dummy dataset and
        load a tokenizer
        """
        cls._tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        cls._dataset = Dataset.from_dict({
            "sentence": [
                "This is a correct sentence.",
                "This sentence correct is not."
            ],
            "id": [1, 2],
            "label": [1, 0]
        })

    def test_tokenize_single_sent_dataset(self) -> None:
        """Test dataset tokenization"""
        data_loader = tokenize_single_sent_dataset(
            dataset=self._dataset,
            text_col_name="sentence",
            label_col_name="label",
            tokenizer=self._tokenizer,
            batch_size=2,
            max_seq_length=16
        )
        data_point = next(iter(data_loader))
        self.assertEqual(len(data_point.keys()), 3)
        self.assertIsInstance(data_point["input_ids"], torch.Tensor)


if __name__ == '__main__':
    unittest.main()
