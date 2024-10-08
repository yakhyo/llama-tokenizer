
# modification from yakhyo
# https://github.com/Nkluge-correa/TeenyTinyLlama

import json
import argparse

from datasets import load_dataset, concatenate_datasets
from tokenizers import SentencePieceBPETokenizer
from transformers import LlamaTokenizerFast, AutoTokenizer, AddedToken, PreTrainedTokenizerFast

from tqdm import tqdm
from typing import List

hf_datasets = ["yakhyo/uz-wiki", "yakhyo/uz-news", "agentlans/high-quality-english-sentences"]
hf_datasets = ["yakhyo/uz-wiki"]


def normalize_text(text: str) -> str:
    """
    Normalize Uzbek characters, replacing variations of o‘, o', o`, and ’ (curved apostrophe).
    """
    return text.replace("‘", "'").replace("`", "'").replace("’", "'").replace("()", "")


def prepare_datasets(datasets_list: List[str]):
    all_data = []
    for dataset_name in datasets_list:
        try:
            data = load_dataset(dataset_name)
            for split in ["train", "test", "validation"]:
                try:
                    all_data.append(data[split])
                except KeyError:
                    pass
        except:
            print(f"dataset: `{dataset_name}` not found, skipping...")

    concat_data = []
    for data in tqdm(all_data):
        data = data.map(lambda example: {"text": normalize_text(example["text"])})
        data = data.remove_columns([col for col in data.column_names if col != "text"])
        concat_data.append(data)

    return concatenate_datasets(concat_data)


def main(args):

    dataset = prepare_datasets(hf_datasets)
    num_reserved_special_tokens = 256

    # select num_samples from the dataset
    dataset = dataset.shuffle(seed=42).select(range(50000))

    # Create a SentencePieceBPETokenizer
    tokenizer = SentencePieceBPETokenizer(replacement="Ġ")

    # Train the SentencePieceBPETokenizer on the dataset
    tokenizer.train_from_iterator(
        iterator=dataset['text'],
        vocab_size=args.vocab_size,
        show_progress=True
    )

    # Save the tokenizer
    tokenizer.save("new-llama-tokenizer.json", pretty=True)

    # Load reference tokenizer
    if args.reference_tokenizer is not None:
        reference_tokenizer = AutoTokenizer.from_pretrained(args.reference_tokenizer)
        reference_tokenizer.save_pretrained("reference-tokenizer")
    else:
        raise ValueError(
            "No tokenizer name provided or no hub token provided. \
            Try using --reference_tokenizer 'meta-llama/Llama-3.2-3b'"
        )

    # Read and dump the json file for the new tokenizer and the reference tokenizer
    with open("new-llama-tokenizer.json", "r") as f:
        new_llama_tokenizer_json = json.load(f)

    with open("reference-tokenizer/tokenizer.json", "r") as f:
        reference_tokenizer_json = json.load(f)

    # Add the reference tokenizer's config to the new tokenizer's config
    new_llama_tokenizer_json["normalizer"] = reference_tokenizer_json["normalizer"]
    new_llama_tokenizer_json["pre_tokenizer"] = reference_tokenizer_json["pre_tokenizer"]
    new_llama_tokenizer_json["post_processor"] = reference_tokenizer_json["post_processor"]
    new_llama_tokenizer_json["decoder"] = reference_tokenizer_json["decoder"]
    new_llama_tokenizer_json["model"]['fuse_unk'] = reference_tokenizer_json["model"]['fuse_unk']
    new_llama_tokenizer_json["model"]['byte_fallback'] = reference_tokenizer_json["model"]['byte_fallback']

    # Dump the new tokenizer's config
    with open("new-llama-tokenizer.json", "w") as f:
        json.dump(new_llama_tokenizer_json, f, indent=2, ensure_ascii=False)

    # Load the new tokenizer as a LlamaTokenizerFast
    new_llama_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file="new-llama-tokenizer.json",
        padding_side="right",
        model_max_length=131072,
        bos_token=AddedToken("<|begin_of_text|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        eos_token=AddedToken("<|end_of_text|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True)
    )
    # Define the special tokens you want to add
    added_tokens = [
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|reserved_special_token_0|>",
        "<|reserved_special_token_1|>",
        "<|reserved_special_token_2|>",
        "<|reserved_special_token_3|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|reserved_special_token_4|>",
        "<|eot_id|>",  # end of turn
    ] + [
        f"<|reserved_special_token_{i}|>" for i in range(5, num_reserved_special_tokens - 5)
    ]

    # Create AddedToken objects for each special token with properties similar to your expected output
    added_tokens_object = [
        AddedToken(token, rstrip=False, lstrip=False, single_word=False, normalized=False, special=True)
        for token in added_tokens
    ]

    # Add these reserved tokens to the tokenizer
    new_llama_tokenizer.add_tokens(added_tokens_object)

    # Save the new tokenizer
    new_llama_tokenizer.save_pretrained("new-llama-tokenizer")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Llama Tokenizer using SentencePieceBPE")

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
# python bpe_tokenizer.py --reference_tokenizer "meta-llama/Llama-3.2-3b" --vocab_size 128000
