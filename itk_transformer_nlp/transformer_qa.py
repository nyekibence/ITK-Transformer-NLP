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

Task:
    Replace instances of `None` with your own code
    unless instructed otherwise in the comments
"""

from argparse import ArgumentParser, Namespace
from typing import (
    Tuple, Dict, Any, Sequence, Generator, Optional, Union)
import re

import torch
from torch.utils.data import DataLoader
from datasets import Dataset, load_dataset
from transformers import (
    BartForQuestionAnswering,
    BartTokenizer,
    PreTrainedTokenizer,
    BatchEncoding
)


def load_jsonl_dataset(dataset_path: str, cache_dir: Optional[str] = None) -> Dataset:
    """Load a dataset from a jsonlines file

    Args:
        dataset_path: Path to the jsonlines dataset file
        cache_dir: Optional. Cache directory to read/write data.
            This can also be set by setting the `HF_DATASETS_CACHE` environment variable.
            By default, the `~/.cache/huggingface/datasets` will be used
    """
    #  It is expected that the locally stored dataset is not divided to splits.
    #  In this case, simply specify `"train"` as the value of the `split` argument of `load_dataset`
    return load_dataset("json", data_files=dataset_path, split="train", cache_dir=cache_dir)


def check_positive_int(input_int: Union[str, int]) -> int:
    """Check if `input_int` is a positive integer.
    If it is, return it as an `int`. Raise `TypeError` otherwise
    """
    input_int = int(input_int)
    if input_int <= 0:
        raise ValueError(f"A positive integer is expected, got {input_int}")
    return input_int


def get_qa_args() -> Namespace:
    """Get command line arguments"""
    parser = ArgumentParser(description="Arguments for QA inference")
    parser.add_argument("dataset", type=load_jsonl_dataset,
                        help="Path to the jsonlines dataset file")
    parser.add_argument("--text-col-names", dest="text_col_names", nargs="+",
                        default=["question", "context"],
                        help="Names of the dataset columns that contain the text data. "
                             "Defaults to `['question', 'context']`")
    parser.add_argument("--max-seq-length", dest="max_seq_length", type=check_positive_int, default=256,
                        help="Maximal sequence length in tokens. If a sequence is longer, "
                             "it will be truncated. Defaults to 256")
    parser.add_argument("--batch-size", dest="batch_size", type=check_positive_int, default=2,
                        help="Batch size used to process data. Defaults to 2")
    return parser.parse_args()


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
        text_col_names: The dataset columns (which can also be called keys) that contain the text data.
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
        #  Call the tokenizer on the text data, turn on padding and truncation.
        #  Feel free to browse the documentation:
        #  https://huggingface.co/docs/transformers/v4.17.0/en/main_classes/tokenizer
        return None

    #  Tokenize the whole dataset using the `map` method. Apply batched tokenization and set the batch_size
    #  Also use the `cols_to_remove` variable to remove unused dataset columns
    #  Please refer to the documentation at
    #  https://huggingface.co/docs/datasets/v2.0.0/en/package_reference/main_classes#datasets.Dataset.map
    dataset = None
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
    #  Adding 1 below is necessary, as it seems that the model `bart-squadv2`
    #  predicts the token before the last answer token
    answer_end = torch.argmax(end_scores, dim=-1, keepdim=True) + 1
    mask = torch.arange(seq_length).repeat(batch_size, 1)
    #  Make `mask` a Boolean tensor without changing its shape. In every row, the values greater than `answer_start`
    #  AND less or equal to `answer_end` should be set to `True`. Every other element should be `False`.
    #  Hint: Use `torch.where`
    mask = None
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
            #  The length of `decoded_answer` might not be `1`. This can occur when the model predicts an answer
            #  that contains tokens from both the question and the answer or only padding tokens.
            #  Note that this behavior is normal, especially if truncation was used during tokenization.
            if len(decoded_answer) != 1:
                decoded_answer = ("NA",)
            yield decoded_input + decoded_answer


def main() -> None:
    """Main function"""
    args = get_qa_args()
    model_name = "a-ware/bart-squadv2"
    #  Load the `BartTokenizer` whose identifier is `model_name`
    tokenizer = None
    data_loader = tokenize_dataset(
        dataset=args.dataset,
        text_col_names=args.text_col_names,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length
    )
    #  Load the `BartForQuestionAnswering` model whose identifier is `model_name`
    model = None
    for triplet in get_predictions(model, data_loader, tokenizer):
        print("\t".join(triplet), end="\n\n")


if __name__ == "__main__":
    main()
