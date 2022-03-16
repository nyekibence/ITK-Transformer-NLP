"""A module for testing the question answering module"""

import unittest

import torch
from transformers import BartTokenizer
from datasets import Dataset

from itk_transformer_nlp.transformer_qa import (
    tokenize_dataset,
    extract_answer,
)


class QATest(unittest.TestCase):
    """Test class for functions related to question answering"""

    @classmethod
    def setUpClass(cls) -> None:
        """Fixture setup: get a tokenizer"""
        cls.tokenizer = BartTokenizer.from_pretrained("a-ware/bart-squadv2")
        cls._input_ids_name = "input_ids"
        cls._attn_mask_name = "attention_mask"

    def test_tokenize_dataset(self) -> None:
        """Test dataset tokenization"""
        dataset = Dataset.from_dict({
            "question": ["How old is the US president?", "How many days are there in a year?"],
            "context": ["The US president is 79 years old.", "There are 365 days in a year."]
        })
        data_loader = tokenize_dataset(
            dataset=dataset,
            text_col_names=("question", "context"),
            tokenizer=self.tokenizer,
            batch_size=2,
            max_seq_length=64
        )
        data_point = next(iter(data_loader))
        expected_keys = [self._input_ids_name, self._attn_mask_name]
        self.assertEqual(expected_keys, list(data_point.keys()))
        self.assertIsInstance(data_point[self._input_ids_name], torch.Tensor)
        self.assertEqual(data_point[self._attn_mask_name].shape[0], 2)

    def test_extract_answers(self) -> None:
        """Test extracting an answer from input token IDs"""
        text = ("What is the capital of England?", "London is the capital of England.")
        expected_answer = "London"
        input_ids = self.tokenizer.batch_encode_plus(
            [text], return_tensors="pt")[self._input_ids_name]
        answer_token_pos = next(
            i for i, token_id in enumerate(torch.squeeze(input_ids).tolist())
            if self.tokenizer.decode(token_id) == expected_answer) - 1  # Subtract 1 to get the token before the start
        start_scores = torch.rand(*input_ids.shape)
        end_scores = torch.rand(*input_ids.shape)
        start_scores[0, answer_token_pos] += 1.
        end_scores[0, answer_token_pos] += 1.

        res = torch.squeeze(extract_answer(
            input_ids,
            start_scores=start_scores,
            end_scores=end_scores,
            pad_token_id=self.tokenizer.pad_token_id
        ))
        res = self.tokenizer.decode(res, skip_special_tokens=True)
        self.assertEqual(expected_answer, res)


if __name__ == "__main__":
    unittest.main()
