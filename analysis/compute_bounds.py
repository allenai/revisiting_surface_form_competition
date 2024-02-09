import argparse
import pandas as pd
from tqdm import tqdm
import os
import yaml


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-primes", type=int, required=True, choices={0, 1, 2, 4, 8}
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices={
            "curie",
            "davinci",
            "davinci-instruct-beta",
            "text-davinci-003",
            "google/flan-t5-xxl",
            "facebook/opt-30b",
        },
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices={"openbookqa", "commonsense_qa", "mmlu"},
    )
    parser.add_argument(
        "--prompt-format",
        type=str,
        required=True,
        choices=[
            "no_answer_choices",
            "string_answer_choices",
            "enumerated_answer_choices",
        ],
    )
    parser.add_argument("--outdir", type=str, default="data")
    parser.add_argument("--random-seed", type=int, default=10)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--complete_existing",
        action="store_true",
        help="for csv files that are not complete, fill in missing datapoints where script previously errored out",
    )
    parser.add_argument(
        "--uncontextual_premise",
        action="store_true",
        help="if specified, will compute uncontextualized denominator for replicating PMI-DC method",
    )
    return parser.parse_args()


def parse_csv(dataset, num_answer_choices):
    p_mass_diff, resolved = [], []
    for _, row in tqdm(dataset.iterrows(), total=len(dataset)):
        probs_list_all_tokens = [
            row[f"all_tokens_prob_choice_{i}"] for i in range(num_answer_choices)
        ]
        probs_sum = row["sum_all_token_probs"]

        # then get the difference between top and second top predictions
        sorted_probs = sorted(probs_list_all_tokens)
        max_value = sorted_probs[-1]
        second_max_value = sorted_probs[-2]

        # store the differences
        total_sfc = 1 - probs_sum
        top_choices_diff = max_value - second_max_value
        total_diff = total_sfc - top_choices_diff
        if total_diff > 0:
            p_mass_diff.append(total_diff)
        else:
            p_mass_diff.append(0)
        # even if all extra prob. mass was placed on synonyms of the 2nd choice, it wouldn't be enough to change the prediction
        if total_sfc < top_choices_diff:
            resolved.append(1)
        else:
            resolved.append(0)

    return round(sum(resolved) / len(resolved), 4) * 100


if __name__ == "__main__":
    args = get_args()

    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    dataset_config = config[args.dataset]
    test_split = dataset_config["test_split"]
    num_answer_choices = dataset_config["num_answer_choices"]
    tasks = dataset_config["tasks"]

    if args.dataset == "commonsense_qa":
        num_datapoints = 500
    elif args.dataset == "openbookqa":
        num_datapoints = 500
    elif args.dataset == "mmlu":
        num_datapoints = 1140
    test_hf_dataset = None

    overall_percents = []
    for num_primes in [0, 1, 2, 4, 8]:
        if num_primes != 0 and args.model in {
            "curie",
            "davinci",
            "davinci-instruct-beta",
        }:
            seeds = [10, 20, 30]
        else:
            seeds = [10]
        percent_0_sfc = []
        for seed in seeds:
            mname = args.model.replace("/", "-")
            csv_file = f"./data/{args.dataset}/{args.dataset}_gpt3_generations_{num_primes}primes_labelOnly_{mname}_temp0_stop###_1samples_seed{seed}_full{test_split}Set_{args.prompt_format}Prompt.csv"
            if not os.path.isfile(csv_file):
                raise Exception("File does not exist:", csv_file)
            # load file
            dataset = pd.read_csv(csv_file)
            if len(dataset) != num_datapoints:
                raise Exception("Wrong # of datapoints:", csv_file)
            resolved = parse_csv(dataset, num_answer_choices)
            percent_0_sfc.append(resolved)

        # compute mean and std
        if len(seeds) == 3:
            overall_percents.append(
                f"${round(pd.Series(percent_0_sfc).mean(), 2)}_{{{round(pd.Series(percent_0_sfc).std(), 2)}}}$"
            )
        elif len(seeds) == 1:
            overall_percents.append(f"${round(pd.Series(percent_0_sfc).mean(), 2)}$")
        else:
            raise Exception()

    print(" & ".join(overall_percents))
