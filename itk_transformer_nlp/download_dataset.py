"""A script to download a dataset and write it
to a file in jsonlines format
"""

from argparse import ArgumentParser, Namespace, FileType
from math import ceil

from datasets import load_dataset

from itk_transformer_nlp.transformer_qa import check_positive_int


def get_dataset_args() -> Namespace:
    """Get command line arguments"""
    parser = ArgumentParser(description="Command line arguments for downloading a dataset")
    parser.add_argument("dataset_name", help="Name of the dataset")
    parser.add_argument("target_path", type=FileType("wb"), help="Path to the file where the dataset will be written")
    parser.add_argument("--sub-dataset", dest="sub_dataset",
                        help="Optional. Name of the sub-dataset, e.g. CoLA in GLUE")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="train",
                        help="Dataset split to use, `train`, `validation` or `test`. Defaults to `train`")
    parser.add_argument("--shuffle", action="store_true", help="Specify this flag if the dataset should be shuffled")
    parser.add_argument("--sample-size", dest="sample_size", type=check_positive_int,
                        help="Optional. Get only the first `--sample-size` elements of the dataset. "
                             "This means random sampling if used together with the `--shuffle` flag")
    return parser.parse_args()


def write_dataset() -> None:
    """Main function in module: download and write a dataset"""
    args = get_dataset_args()
    dataset = load_dataset(args.dataset_name, args.sub_dataset, split=args.split)
    if args.shuffle:
        dataset = dataset.shuffle(42)
    if args.sample_size is not None:
        num_shards = ceil(len(dataset) / args.sample_size)
        dataset = dataset.shard(num_shards, 0, contiguous=True)
    dataset.to_json(args.target_path, force_ascii=False)


if __name__ == "__main__":
    write_dataset()
