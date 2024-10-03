
# modification from yakhyo
# https://github.com/Nkluge-correa/TeenyTinyLlama

import json
import argparse

from datasets import load_dataset, concatenate_datasets
from tokenizers import SentencePieceBPETokenizer
from transformers import LlamaTokenizerFast, AutoTokenizer


from typing import List

hf_datasets = ["yakhyo/uz-wiki", "yakhyo/uz-news"]


def prepare_datasets(datasets_list: List[str]):
    all_data = []
    for dataset_name in datasets_list:
        try:
            data = load_dataset(
                dataset_name,
                token=args.hub_token if args.hub_token else None,
            )
            for split in ["train", "test", "validation"]:
                try:
                    all_data.append(data[split])
                except KeyError:
                    pass
        except:
            print(f"dataset: `{dataset_name}` not found, skipping...")

    concat_data = []
    for data in all_data:
        data = data.remove_columns([col for col in data.column_names if col != "text"])
        concat_data.append(data)

    return concatenate_datasets(concat_data)


def main(args):

    dataset = prepare_datasets(hf_datasets)

    # select num_samples from the dataset
    dataset = dataset.shuffle(seed=42).select(range(len(dataset)))

    # Create a SentencePieceBPETokenizer
    tokenizer = SentencePieceBPETokenizer(
        replacement="Ġ"
    )

    # Train the SentencePieceBPETokenizer on the dataset
    tokenizer.train_from_iterator(
        iterator=dataset['text'],
        vocab_size=args.vocab_size,
        show_progress=True,
        special_tokens=["<unk>", "<s>", "</s>",  "<pad>"],
    )

    # Save the tokenizer
    tokenizer.save("new-sentencepiece-tokenizer.json", pretty=True)

    # Load reference tokenizer
    if args.reference_tokenizer is not None and args.hub_token is not None:
        reference_tokenizer = AutoTokenizer.from_pretrained(
            args.reference_tokenizer, token=args.hub_token if args.hub_token else None)
        reference_tokenizer.save_pretrained("reference-tokenizer")
    else:
        raise ValueError(
            "No tokenizer name provided or no hub token provided. Try using --reference_tokenizer 'meta-llama/Llama-2-7b-hf'")

    # Read and dump the json file for the new tokenizer and the reference tokenizer
    with open("new-sentencepiece-tokenizer.json") as f:
        new_llama_tokenizer_json = json.load(f)

    with open("reference-tokenizer/tokenizer.json") as f:
        reference_tokenizer_json = json.load(f)

    # Add the reference tokenizer's config to the new tokenizer's config
    new_llama_tokenizer_json["normalizer"] = reference_tokenizer_json["normalizer"]
    new_llama_tokenizer_json["pre_tokenizer"] = reference_tokenizer_json["pre_tokenizer"]
    new_llama_tokenizer_json["post_processor"] = reference_tokenizer_json["post_processor"]
    new_llama_tokenizer_json["decoder"] = reference_tokenizer_json["decoder"]
    new_llama_tokenizer_json["model"]['fuse_unk'] = reference_tokenizer_json["model"]['fuse_unk']
    new_llama_tokenizer_json["model"]['byte_fallback'] = reference_tokenizer_json["model"]['byte_fallback']

    # Dump the new tokenizer's config
    with open("new-sentencepiece-tokenizer.json", "w") as f:
        json.dump(new_llama_tokenizer_json, f, indent=2, ensure_ascii=False)

    # Load the new tokenizer as a LlamaTokenizerFast
    new_llama_tokenizer = LlamaTokenizerFast(
        tokenizer_file="new-sentencepiece-tokenizer.json",
        unk_token="<unk>",
        unk_token_id=0,
        bos_token="<s>",
        bos_token_id=1,
        eos_token="</s>",
        eos_token_id=2,
        pad_token="<pad>",
        pad_token_id=3,
        padding_side="right",
    )

    # Save the new tokenizer
    new_llama_tokenizer.save_pretrained("new-llama-tokenizer")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Llama Tokenizer using SentencePieceBPE")

    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to access the dataset on the hub"
    )
    parser.add_argument(
        "--reference_tokenizer",
        type=str,
        default=None,
        help="The name of the reference tokenizer to use"
    )

    parser.add_argument(
        "--vocab_size",
        type=int,
        default=None,
        help="Vocabulary size to use for the tokenizer"
    )
    args = parser.parse_args()
    main(args)

# How to run:
# python bpe_tokenizer.py  --hub_token "hf.." --reference_tokenizer "meta-llama/Llama-3.2-3b" --vocab_size 32000
