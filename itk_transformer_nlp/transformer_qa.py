"""Question answering (QA) with BART

This module does QA with a fine-tuned encoder-decoder model.
The QA task demonstrated here is similar to a reading comprehension
task: the input consists of a question and a context from which the
answer should be extracted, then the model predicts two integers:
the positions of the tokens that indicate the start of the answer
and its end, respectively.

The workflow includes the following steps:
* Load a dataset
* Tokenize the dataset
* Load the model
* Call the model on data batches
* Output the results
"""

import re
from typing import (
    Tuple, Dict, Any, Sequence, Generator, Optional)

import torch
from torch.utils.data import DataLoader
from datasets import Dataset, load_dataset
from transformers import (
    BartForQuestionAnswering,
    BartTokenizer,
    PreTrainedTokenizer,
    BatchEncoding
)


def tokenize_dataset(
        dataset: Dataset,
        text_col_names: Sequence[str],
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
        max_seq_length: Optional[int] = None,
        remove_old_cols: bool = True
) -> DataLoader:
    """Tokenize a dataset

    Args:
        dataset: The input data as a `datasets.Dataset` object
        text_col_names: The dataset columns (which can also be called keys) that contains the text data.
            None of them should be `"input_ids"`, `"attention_mask"` or `"token_type_ids"` as those are
            the columns returned by the tokenizer. Text data will be fed to the tokenizer in the order
            the elements are specified in `text_col_names`
        tokenizer: A pre-trained tokenizer
        batch_size: Batch size for tokenization
        max_seq_length: Optional. If the number of tokens in a sequence is `n` and
            `n` > `max_seq_length`, the sequence will be truncated. This means cutting off
            the last `n - max_seq_length` tokens. If not specified, truncation will not be used
        remove_old_cols: If `True`, the original dataset columns (that can be also called keys)
            will be removed, leaving only the columns created after tokenization. Defaults to `True`

    Returns:
         The tokenized dataset as a `DataLoader`
    """
    tokenizer_cols = tokenizer("Dummy text").keys()
    if set(text_col_names) & tokenizer_cols:
        raise ValueError(f"Invalid text column names: {text_col_names}")
    cols_to_remove = list(dataset.features.keys()) if remove_old_cols else None
    tokenizer.model_max_length = max_seq_length

    def tok_func(example: Dict[str, Any]) -> BatchEncoding:
        text_cols = [example[text_col_name] for text_col_name in text_col_names]
        return tokenizer(*text_cols, padding=True, truncation=True)

    dataset = dataset.map(
        tok_func, batched=True, batch_size=batch_size, remove_columns=cols_to_remove)
    dataset.set_format(type="torch", columns=list(tokenizer_cols))
    return DataLoader(dataset, batch_size=batch_size)


def extract_answer(
        input_ids: torch.Tensor,
        start_scores: torch.Tensor,
        end_scores: torch.Tensor,
        pad_token_id: int
) -> torch.Tensor:
    """Extract answer token IDs from input IDs

    Args:
        input_ids: An input token ID tensor of shape `(batch_size, sequence_length)`
        start_scores: Scores that indicate the answer start position. This is a tensor
            of shape `(batch_size, sequence_length)`
        end_scores: Scores that indicate the answer end position. This is a tensor
            of shape `(batch_size, sequence_length)`
        pad_token_id: The padding token ID

    Returns:
        A tensor of padded answer token IDs. The padded shape is the same as the
        shape of the input token ID tensor
    """
    batch_size, seq_length = input_ids.shape
    answer_start = torch.argmax(start_scores, dim=-1, keepdim=True)
    answer_end = torch.argmax(end_scores, dim=-1, keepdim=True)
    mask = torch.arange(seq_length).repeat(batch_size, 1)
    mask = torch.where(
        torch.logical_and(answer_start <= mask, mask <= answer_end),
        True, False)
    return torch.where(mask, input_ids, pad_token_id)


def decode_bart_input(input_ids: torch.Tensor, tokenizer: BartTokenizer) -> Tuple[Tuple[str, ...], ...]:
    """Detokenize input IDs created with a `BartTokenizer`

    Args:
        input_ids: An input token ID tensor of shape `(batch_size, sequence_length)`
        tokenizer: A pre-trained BART tokenizer for decoding

    Returns:
        A batch of detokenized input sequence parts as tuples of strings
    """
    sep_token = tokenizer.sep_token
    outputs = tokenizer.batch_decode(input_ids)
    return tuple(tuple(seq.split(sep_token)) for seq in outputs)


def clean_decoded_batch(
        decoded_batch: Tuple[Tuple[str, ...], ...],
        pattern: re.Pattern
) -> Tuple[Tuple[str, ...], ...]:
    """Clean decoded text by removing meta tokens

    Args:
        decoded_batch: A tuple of tuples of strings. Each nested tuple corresponds to a text sequence
            with split into subsequences (e.g. a question and a context)
        pattern: A regex to clean the sequences

    Returns:
        The cleaned data with the data structure unaltered
    """
    clean_batch = []
    for seq in decoded_batch:
        clean_batch.append(tuple(cleaned_subseq for subseq in seq
                           if (cleaned_subseq := pattern.sub("", subseq).strip())))
    return tuple(clean_batch)


def get_predictions(
        model: BartForQuestionAnswering,
        data_loader: DataLoader,
        tokenizer: BartTokenizer,
        input_ids_name: str = "input_ids"
) -> Generator[Tuple[str, str, str], None, None]:
    """Use the model for inference

    Args:
        model: A BART model fine-tuned for QA
        data_loader: A `DataLoader` that outputs dicts whose keys are strings
            and the values are PyTorch tensors of shape `(batch_size, sequence_length)`
        tokenizer: A pre-trained BART tokenizer for decoding
        input_ids_name: The key in the data loader outputs that indicates input token IDs.
            Defaults to `input_ids`

    Returns:
        A generator of `question - context - answer` triplets
    """
    pad_id = tokenizer.pad_token_id
    cleaning_pattern = re.compile(
        "|".join([tokenizer.cls_token, tokenizer.pad_token, tokenizer.sep_token]))
    for batch in data_loader:
        start_logits, end_logits = model(
            **batch, output_attentions=False, return_dict=False)[:2]
        input_ids = batch[input_ids_name]
        answers = extract_answer(input_ids, start_logits, end_logits, pad_id)
        decoded_inputs = clean_decoded_batch(decode_bart_input(input_ids, tokenizer), cleaning_pattern)
        decoded_answers = clean_decoded_batch(decode_bart_input(answers, tokenizer), cleaning_pattern)
        for decoded_input, decoded_answer in zip(decoded_inputs, decoded_answers):
            yield decoded_input + decoded_answer


def main() -> None:
    """Main function"""
    dataset = load_dataset("squad_v2", split="validation")
    tokenizer = BartTokenizer.from_pretrained("a-ware/bart-squadv2")
    data_loader = tokenize_dataset(
        dataset=dataset,
        text_col_names=("question", "context"),
        tokenizer=tokenizer,
        batch_size=4,
        max_seq_length=128
    )
    model = BartForQuestionAnswering.from_pretrained("a-ware/bart-squadv2")
    for qca in get_predictions(model, data_loader, tokenizer):
        print("\t".join(qca))


if __name__ == "__main__":
    main()
