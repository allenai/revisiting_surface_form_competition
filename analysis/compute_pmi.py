import argparse
import datasets as nlp
import numpy as np
import os
import pandas as pd
import sys
import yaml
from plotting.compare_independent_variables import compute_pmi_accuracy


# get relevant prediction files
# copied & modified from plotting/compare_independent_variables.py
def get_unconditional_prediction_file_stats(
    dataset_name, model, test_split, num_datapoints, seed, num_primes, prompt_type
):
    preds_file = f"./data/{dataset_name}/uncontextual_preds/{dataset_name}_gpt3_generations_{num_primes}primes_labelOnly_{model.replace('/','-')}_temp0_stop###_1samples_seed{seed}_full{test_split.title()}Set_{prompt_type}Prompt.csv"
    preds = pd.read_csv(preds_file)
    # check that file is complete
    if len(preds) != num_datapoints:
        print("File does not have enough datapoints:")
        print(preds_file)
        raise Exception
    return preds


def get_conditional_prediction_file_stats(
    dataset_name, model, test_split, num_datapoints, seed, num_primes, prompt_type
):
    preds_file = f"./data/{dataset_name}/{dataset_name}_gpt3_generations_{num_primes}primes_labelOnly_{model.replace('/','-')}_temp0_stop###_1samples_seed{seed}_full{test_split.title()}Set_{prompt_type}Prompt.csv"
    preds = pd.read_csv(preds_file)
    # check that file is complete
    if len(preds) != num_datapoints:
        print("File does not have enough datapoints:")
        print(preds_file)
        raise Exception
    # and get logfile since we already computed seq. scoring accuracy and prob. mass
    logfile = f"./data/{dataset_name}/{num_primes}_primes_{dataset_name}_{model.replace('/','-')}_seed_{seed}_{prompt_type}Prompt.log"
    if not os.path.isfile(logfile):
        raise Exception("Logfile does not exist:", logfile)
    return preds, logfile


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, choices={"openbookqa", "commonsense_qa", "mmlu"}
    )
    parser.add_argument("--plot_label_conditional_pmi", action="store_true")
    args = parser.parse_args()

    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    dataset_config = config[args.dataset]
    test_split = dataset_config["test_split"]
    num_answer_choices = dataset_config["num_answer_choices"]
    tasks = dataset_config["tasks"]

    if args.dataset in {"openbookqa", "commonsense_qa"}:
        num_datapoints = 500
    elif args.dataset == "mmlu":
        num_datapoints = 1140

    # only load datasets once
    if args.dataset in {"openbookqa", "commonsense_qa"}:
        test_hf_dataset = nlp.load_dataset(args.dataset, split=test_split + "[:500]")
    elif args.dataset == "mmlu":
        test_hf_dataset = []
        for subtask in tasks:
            test_hf_dataset.extend(
                list(nlp.load_dataset("hendrycks_test", subtask, split="test[:20]"))
            )
        assert len(test_hf_dataset) == 1140

    # loop through each possibility
    for model in [
        "google-flan-t5-xxl",
        "curie",
        "davinci",
        "davinci-instruct-beta",
        "text-davinci-003",
        "facebook-opt-30b",
    ]:
        print("MODEL:", model)
        print("######")

        # Note PMI-denominator is always 0-shot (and therefore, seed is always 10)
        l_only_preds = get_unconditional_prediction_file_stats(
            args.dataset,
            model,
            test_split,
            num_datapoints,
            seed=10,
            num_primes=0,
            prompt_type="no_answer_choices",
        )
        if args.plot_label_conditional_pmi:
            l_cond_L_string_preds = get_unconditional_prediction_file_stats(
                args.dataset,
                model,
                test_split,
                num_datapoints,
                seed=10,
                num_primes=0,
                prompt_type="string_answer_choices",
            )
            l_cond_L_enum_preds = get_unconditional_prediction_file_stats(
                args.dataset,
                model,
                test_split,
                num_datapoints,
                seed=10,
                num_primes=0,
                prompt_type="enumerated_answer_choices",
            )

        for q_type in [
            "no_answer_choices",
            "string_answer_choices",
            "enumerated_answer_choices",
        ]:
            for num_primes in [0, 1, 2, 4, 8]:
                # only these 3 models have all 3 seeds
                if num_primes != 0 and model in {
                    "curie",
                    "davinci",
                    "davinci-instruct-beta",
                }:
                    seeds = [10, 20, 30]
                else:
                    seeds = [10]
                for seed in seeds:
                    # get numerator predictions dataframe and logfile
                    numerator_df, logfile = get_conditional_prediction_file_stats(
                        args.dataset,
                        model,
                        test_split,
                        num_datapoints,
                        seed=seed,
                        num_primes=num_primes,
                        prompt_type=q_type,
                    )

                    if args.plot_label_conditional_pmi:
                        if q_type == "string_answer_choices":
                            conditional_pmi_acc = compute_pmi_accuracy(
                                numerator_df,
                                l_cond_L_string_preds,
                                test_hf_dataset,
                                args.dataset,
                                num_answer_choices,
                            )
                        elif q_type == "enumerated_answer_choices":
                            conditional_pmi_acc = compute_pmi_accuracy(
                                numerator_df,
                                l_cond_L_enum_preds,
                                test_hf_dataset,
                                args.dataset,
                                num_answer_choices,
                            )
                        else:
                            conditional_pmi_acc = None
                    unconditional_pmi_acc = compute_pmi_accuracy(
                        numerator_df,
                        l_only_preds,
                        test_hf_dataset,
                        args.dataset,
                        num_answer_choices,
                    )

                    # append to original logfile
                    with open(logfile, "r") as f:
                        lines = f.readlines()
                    detector = [
                        it for it in lines if "Unconditional PMI Accuracy:" in it
                    ]
                    if len(detector) > 1:
                        print("TOO MANY WRITES:", logfile)
                    elif len(detector) == 1:
                        try:
                            if (
                                float(detector[0].strip().split(":")[-1])
                                != unconditional_pmi_acc
                            ):
                                breakpoint()
                        except:
                            breakpoint()
                    else:
                        # nothing written to file yet
                        with open(logfile, "a") as f:
                            print(
                                "\nUnconditional PMI Accuracy:",
                                unconditional_pmi_acc,
                                file=f,
                            )
                        print("Written to", logfile)

                    if (
                        args.plot_label_conditional_pmi
                        and conditional_pmi_acc is not None
                    ):
                        conditional_detector = [
                            it
                            for it in lines
                            if "Label-Conditional PMI Accuracy:" in it
                        ]
                        if len(conditional_detector) > 1:
                            print("TOO MANY WRITES:", logfile)
                        if len(conditional_detector) == 0:
                            with open(logfile, "a") as f:
                                print(
                                    "\nLabel-Conditional PMI Accuracy:",
                                    conditional_pmi_acc,
                                    file=f,
                                )
                            print("Written to", logfile)
