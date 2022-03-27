#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Fine-tuning on CoLA

We fine-tune a BERT model for binary classification:
The input is a sentence and the task is to predict whether
the sentence is grammatically and semantically well-formed.

We are going to use the HuCoLA dataset. It contains well-formed
anf ill-formed Hungarian sentences labelled by human annotators.

We can apply the huBERT encoder that was pre-trained on Hungarian
data. We only need to fine-tune it for this task.

HuCoLA URL: https://huggingface.co/datasets/NYTK/HuCOLA
huBERT URL: https://huggingface.co/SZTAKI-HLT/hubert-base-cc
"""

from argparse import ArgumentParser, Namespace, ArgumentTypeError
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, Union, Literal
from functools import partial

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from datasets import (
    load_dataset, Dataset,
    load_metric, Metric
)
from transformers import (
    BertTokenizer,
    PreTrainedTokenizer,
    BertForSequenceClassification,
    BatchEncoding,
    get_scheduler
)
from transformers.trainer_utils import SchedulerType

from itk_transformer_nlp.transformer_qa import check_positive_int


def check_model_save_dir(model_dir: str) -> str:
    """Check if the input is valid path for model saving"""
    model_path = Path(model_dir)
    if not model_path.parent.exists():
        raise ArgumentTypeError(
            f"Invalid path: parent directory of {model_dir} does not exist.")
    if model_path.is_file():
        raise ArgumentTypeError(
            f"Invalid path: {model_dir} is a file.")
    return model_dir


def check_float_interval(
        number: Union[str, float, int],
        interval: Tuple[float, float],
        interval_type: Literal["open", "closed", "left_closed", "right_closed"]
) -> float:
    """Check if a `float` is in a specified interval

    Args:
        number: The number to check.
        interval: A tuple of two floats that define the interval start and end.
        interval_type: One of `'open'`, `'closed'`, `'left_closed'`, `'right_closed'`.
            It refers to whether the interval start and end value should be accepted.
    """
    number = float(number)
    start, end = interval
    if interval_type == "open":
        is_correct = start < number < end
    elif interval_type == "closed":
        is_correct = start <= number <= end
    elif interval_type == "left_closed":
        is_correct = start <= number < end
    elif interval_type == "right_closed":
        is_correct = start < number <= end
    else:
        raise ValueError(f"Unknown interval type: {interval_type}")
    if not is_correct:
        raise ArgumentTypeError(f"{number} is not in the {interval_type} interval "
                                f"between {start} and {end}")
    return number


def get_hucola_training_args() -> Namespace:
    """Get command line arguments"""
    parser = ArgumentParser(
        description="Get command line arguments for fine-tuning huBERT on HuCoLA")
    parser.add_argument("--batch-size", dest="batch_size", type=check_positive_int,
                        default=8, help="Training batch size. Defaults to `8`.")
    parser.add_argument("--max-seq-length", dest="max_seq_length", type=check_positive_int,
                        default=128, help="Maximum sequence length above which inputs "
                                          "will be truncated. Defaults to `128`.")
    parser.add_argument("--num-epochs", dest="num_epochs", type=check_positive_int,
                        default=2, help="Number of training epochs. Defaults to `2`.")
    parser.add_argument("--lr", type=partial(check_float_interval, interval=(0., 1.),
                                             interval_type="open"), default=1e-6,
                        help="Learning rate parameter, a `float` between 0 and 1. "
                             "Defaults to `1e-6`.")
    parser.add_argument("--weight-decay", dest="weight_decay", default=1e-6,
                        type=partial(check_float_interval, interval=(0., 1.),
                                     interval_type="left_closed"),
                        help="Weight_decay parameter, a `float` between 0 and 1. "
                             "Defaults to `1e-6`.")
    parser.add_argument("--model-save-path", dest="model_save_path", type=check_model_save_dir,
                        help="Optional. Path to a directory where the model will be saved. "
                             "If not specified, the fine-tuned model will not be saved!")
    parser.add_argument("--hucola-sent-col", dest="hucola_sent_col", default="Sent",
                        help="Sentence column name in the `HuCoLA` dataset. Override the "
                             "default value only if you made sure that the column name "
                             "changed (it is not `'Sent'` anymore). Defaults to `'Sent'`.")
    parser.add_argument("--hucola-label-col", dest="hucola_label_col", default="Label",
                        help="Label column name in the `HuCoLA` dataset. Override the "
                             "default value only if you made sure that the column name "
                             "changed (it is not `'Label'` anymore). Defaults to `'Label'`.")
    return parser.parse_args()


def rename_column(
        dataset: Dataset,
        old_col_name: str,
        new_col_name: str,
        col_type: type
) -> Dataset:
    """Rename a column in a dataset and convert its data type

    Args:
        dataset: The dataset to be processed.
        old_col_name: The column name that is to be changed.
        new_col_name: The new column name.
        col_type: The data in the renamed column will be converted to this type.

    Returns:
        The dataset with the renamed column
    """
    return dataset.map(
        lambda example: {new_col_name: col_type(example[old_col_name])},
        remove_columns=[old_col_name]
    )


# The function below is similar to the the tokenization function
# defined in `transformer_qa.py`. However, we will now process
# single sentences, not sentence pairs.
# Note that we do not need the tokenizer to output `token_type_ids`.
# This would be a tensor with elements 0 and 1 which indicate whether a token
# comes from the first or the second input sentence. However, only single
# sentences will be tokenized now, as each data point contains only one
# sentence. The model will be able to handle this.
def tokenize_single_sent_dataset(
        dataset: Dataset,
        text_col_name: str,
        label_col_name: str,
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
        max_seq_length: Optional[int] = None,
) -> DataLoader:
    """Tokenize a dataset

    Args:
        dataset: The input data as a `datasets.Dataset` object
        text_col_name: The dataset column (which can also be called a key)
            that contains the text data. It should not be `"input_ids"` or
            `"attention_mask"` as those are the columns
            returned by the tokenizer.
        label_col_name: The dataset column that contains the labels
        tokenizer: A pre-trained tokenizer
        batch_size: Batch size for tokenization
        max_seq_length: Optional. If the number of tokens in a sequence is
            `n` and `n` > `max_seq_length`, the sequence will be truncated.
            This means cutting off the last `n - max_seq_length` tokens.
            If not specified, truncation will not be applied.

    Returns:
         The tokenized dataset as a `DataLoader`
    """
    dataset_cols = dataset.features.keys()
    if text_col_name not in dataset_cols:
        raise KeyError(f"{text_col_name} is not a dataset field.")
    if label_col_name not in dataset_cols:
        raise KeyError(f"{label_col_name} is not a dataset field.")
    tokenizer_cols = tokenizer("Dummy text", return_token_type_ids=False).keys()
    if text_col_name in tokenizer_cols:
        raise KeyError(f"Invalid text column name: {text_col_name}")
    tokenizer.model_max_length = max_seq_length

    def tok_func(example: Dict[str, Any]) -> BatchEncoding:
        return tokenizer(example[text_col_name], padding=True, truncation=True,
                         return_token_type_ids=False)

    dataset = dataset.map(tok_func, batched=True, batch_size=batch_size)
    dataset.set_format(
        type="torch", columns=list(tokenizer_cols) + [label_col_name])
    return DataLoader(dataset, batch_size=batch_size)


# The following functions are related to fine-tuning.
# Note that the architecture that we are going to use handles binary
# classification as a special case of multi-class classification.
# It uses softmax rather than sigmoid (i.e. logistic regression) for
# classification. As a result, the classifier head outputs two floating point
# numbers (one for each class) per input sequence instead of one.
def _get_loss_log_accuracy(
        model: BertForSequenceClassification,
        metric: Metric,
        device: torch.device,
        batch: Dict[str, torch.Tensor],
        label_key: str
) -> torch.Tensor:
    """Helper function to calculate loss and add predictions to a metric
    after a training or validation step

    Args:
        model: A BERT model
        metric: A metric object which logs predictions but does not calculate
            the metric (e.g. accuracy) until explicitly requested to.
        device: The same device where the model was put
        batch: A training batch as a `dict` whose values are tensors
        label_key: The key that corresponds to the label values in the
            batch `dict`. This is typically `'labels'`.

    Returns:
        The loss as a scalar tensor.
    """
    batch = {k: v.to(device) for k, v in batch.items()}
    outputs = model(**batch)
    # We get logits from the model, but the metric object needs a concrete
    # prediction. We simply use an argmax operation to achieve this.
    # For classification, one would first call the softmax, but softmax
    # does not change the argmax result.
    # Note that `outputs.logits` is of shape `(batch_size, number_of_classes)`
    preds = torch.argmax(outputs.logits, dim=-1)
    metric.add_batch(predictions=preds, references=batch[label_key])
    return outputs.loss


def log_results(
        epoch: int,
        step: int,
        train_loss: Union[float, torch.Tensor],
        train_acc: Union[float, torch.Tensor],
        val_loss: Union[float, torch.Tensor],
        val_acc: Union[float, torch.Tensor],
) -> None:
    """Helper function to log loss and accuracy scores"""
    print(f"\nTraining loss at step {step}, epoch {epoch}: "
          f"{train_loss}")
    print(f"Training accuracy at step {step}, epoch {epoch}: "
          f"{train_acc}")
    print(f"Validation loss at step {step}, epoch {epoch}: "
          f"{val_loss}")
    print(f"Validation accuracy at step {step}, epoch {epoch}: "
          f"{val_acc}")


# We assume that we already made sure that the for loop iterates at least once.
# So we can suppress warnings related to referencing possibly unassigned variables.
# This is only relevant if you use the PyCharm IDE.
# noinspection PyUnboundLocalVariable
@torch.inference_mode()
def do_evaluation(
        model: BertForSequenceClassification,
        metric: Metric,
        device: torch.device,
        data_loader: DataLoader,
        label_key: str,
        metric_type: str
) -> Tuple[float, float]:
    """Do a validation epoch

    Args:
        model: A BERT model
        metric: A metric object which logs predictions but does not calculate
            the metric (e.g. accuracy) until explicitly requested to.
        device: The device to use, the same device where the model was put.
        data_loader: A validation dataset wrapped by a `DataLoader`.
            Batches will be expected to be `dict` instances that contain the
            model inputs.
        label_key: The key that corresponds to the label values in the
            batch `dict`. This is typically `'labels'`.
        metric_type: Metric type to calculate, e.g. `'accuracy'`.

    Returns:
        The validation loss and accuracy
    """
    model.eval()
    val_loss = 0.
    for val_step, val_batch in enumerate(data_loader, start=1):
        loss = _get_loss_log_accuracy(
            model, metric, device, batch=val_batch, label_key=label_key)
        val_loss += loss
    val_loss /= val_step
    # Only now do we compute the metric score.
    # `metric.compute()` returns a `dict`, but we need only the score itself
    val_acc = metric.compute()[metric_type]
    return val_loss.item(), val_acc


# We assume that we already made sure that the for loops iterate at least once.
# So we can suppress warnings related to referencing possibly unassigned variables.
# This is only relevant if you use the PyCharm IDE.
# noinspection PyUnboundLocalVariable
def fine_tune_for_classification(
        model: BertForSequenceClassification,
        train_data_loader: DataLoader,
        val_data_loader: DataLoader,
        num_epochs: int,
        learning_rate: float = 1e-6,
        weight_decay: float = 1e-6,
        scheduler_type: Union[str, SchedulerType] = "linear",
        num_warmup_steps: int = 200,
        logging_freq: int = 100,
        metric_type: str = "accuracy"
) -> BertForSequenceClassification:
    """Fine-tune a model

     Args:
         model: A BERT model
         train_data_loader: A training dataset wrapped by a `DataLoader`.
             Batches will be expected to be `dict` instances that contain the
             model inputs.
         val_data_loader: A validation dataset wrapped by a `DataLoader`.
             Batches will be expected to be `dict` instances that contain the
             model inputs.
         num_epochs: Number of training epochs. Defaults to `2`.
         learning_rate: Learning rate argument passed to an `AdamW` optimizer.
             Defaults to `1e-6`.
         weight_decay: Weight decay argument passed to an `AdamW` optimizer.
             Defaults to `1e-6`.
         scheduler_type: `name` parameter of the `transformers.get_scheduler`
             function. Defaults to `'linear'`.
         num_warmup_steps: Number of learning rate warmup steps.
         logging_freq: How often to log expressed in term of training steps.
             Counting starts over after each epoch end.
         metric_type: Metric type to calculate, Defaults to `'accuracy'`.

     Returns:
         The fine-tuned model
     """
    num_training_steps = num_epochs * len(train_data_loader)
    if num_training_steps <= num_warmup_steps:
        raise ValueError(f"The number of training steps ({num_training_steps}) "
                         "should be larger than than the number of "
                         f"warmup steps ({num_warmup_steps}).")
    device = torch.device("cuda:0") if torch.cuda.is_available() \
        else torch.device("cpu")
    model.to(device).train()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate,
                                  weight_decay=weight_decay)
    lr_scheduler = get_scheduler(
        name=scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    train_metric = load_metric(metric_type)
    val_metric = load_metric(metric_type)
    # `label_key` is the key in the batch dictionaries
    # whose values are the labels
    label_key = "labels"

    for epoch in range(1, num_epochs + 1):
        train_loss = 0.
        print(f"Epoch {epoch} started...")
        loss_step_tracker = 0
        for train_step, train_batch in enumerate(tqdm(train_data_loader),
                                                 start=1):
            loss_step_tracker += 1
            loss = _get_loss_log_accuracy(
                model, train_metric, device,
                batch=train_batch, label_key=label_key)
            train_loss += loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if train_step % logging_freq == 0:
                # Evaluate the model on the validation data.
                # Use the `do_evaluation function` that we already
                # implemented.
                val_loss, val_acc = None
                train_acc = train_metric.compute()[metric_type]
                log_results(
                    epoch=epoch,
                    step=train_step,
                    train_loss=train_loss / loss_step_tracker,
                    train_acc=train_acc,
                    val_loss=val_loss,
                    val_acc=val_acc
                )
                train_loss = 0.
                loss_step_tracker = 0
                model.train()

    # Get the final logs after the training was completed
    if train_step % logging_freq != 0:
        # Evaluate the final model on the validation data.
        # Use the `do_evaluation function` that we already
        # implemented.
        val_loss, val_acc = None
        train_acc = train_metric.compute()[metric_type]
        log_results(
            epoch=epoch,
            step=train_step,
            train_loss=train_loss / loss_step_tracker,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc
        )
    return model.eval()


def main() -> None:
    """Main function"""
    model_name = "SZTAKI-HLT/hubert-base-cc"
    dataset_name = "NYTK/HuCOLA"
    args = get_hucola_training_args()

    # Load the Hungarian CoLA dataset.
    # This is a dataset for a binary classification task:
    # Every data point contains a Hungarian sentence and a label.
    # If the sentence is grammatically and semantically well-formed,
    # the label is `1`. Otherwise, it is `0`.
    # The next line of code downloads both the training and validation splits
    # and puts them into a list which is immediately unpacked.
    # Suppress PyCharm TypeChecker, it would complain about the list passed to `split`.
    # This is, however, correct. It is only relevant if you use the PyCharm IDE.
    # noinspection PyTypeChecker
    dataset = load_dataset(
        dataset_name, split=["train", "validation"], field="data")
    train_dataset, val_dataset = dataset
    del dataset
    # We have seen that the labels are given as strings in the column 'Label'
    # of the datasets. Let us write a function that converts these strings to
    # integers and renames the column. The new name should be 'labels', as
    # the `transformers.BertForSequenceClassification` object that we are about to
    # use requires that labels be provided through the keyword argument `labels`.
    old_label_name, new_label_name = args.hucola_label_col, "labels"
    train_dataset = rename_column(
        train_dataset, old_label_name, new_label_name, int)
    val_dataset = rename_column(
        val_dataset, old_label_name, new_label_name, int)

    # As we use a pre-trained model, a tokenizer must already exist.
    # We can simply download it from HuggingFace Hub.
    tokenizer = BertTokenizer.from_pretrained(model_name)
    train_data_loader, val_data_loader = (tokenize_single_sent_dataset(
        dataset=dataset,
        text_col_name=args.hucola_sent_col,
        label_col_name=new_label_name,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
    ) for dataset in (train_dataset, val_dataset))

    # Load the pre-trained model. The classifier head weights will be
    # initialized randomly.
    # Use a method of `BertForSequenceClassification` to load the model
    # and set the number of classes to 2.
    # Feel free to refer to
    # https://huggingface.co/docs/transformers/main/en/main_classes/model
    hu_model = None
    # Fine-tune the model.
    # If everything is all right, both the training and the
    # validation accuracy should be larger than 80% by the
    # end of the training.
    hu_model = fine_tune_for_classification(
        model=hu_model,
        train_data_loader=train_data_loader,
        val_data_loader=val_data_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay
    )
    if args.model_save_path is not None:
        hu_model.save_pretrained(args.model_save_path)


if __name__ == "__main__":
    main()
