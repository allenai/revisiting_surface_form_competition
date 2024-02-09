import argparse
from collections import defaultdict
import datasets as nlp
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import yaml
import matplotlib

flatten = itertools.chain.from_iterable

matplotlib.rcParams.update({"font.size": 14})
matplotlib.rcParams["figure.dpi"] = 300


def parse_logfile(logfile, ds_size):
    uncond_pmi_acc = "n/a"
    with open(logfile, "r") as f:
        lines = [line.rstrip() for line in f]
        successfully_found = 0
        for i, it in enumerate(lines):
            if "Total:" in it:
                # check # datapoints in logfile matches
                num = int(it.split("Total:")[-1].strip())
                if num != ds_size:
                    print(
                        f"Not enough datapoints in logfile: {num} (should be {ds_size}) {logfile}"
                    )
            if "Percent " in it:
                if it.split(":")[0] == "Percent of correct sequence scoring examples":
                    seq_acc = float(it.split(":")[1])
                    successfully_found += 1
            if "Distribution of all-token 'Probability Masses':" in it:
                # also test for scientific notation format
                if str(ds_size) not in it and str("{:.6e}".format(ds_size)) not in it:
                    print(
                        f"Not enough datapoints in logfile: {it.split(':')[1].split('count')[1].strip()} (should be {ds_size}) {logfile}"
                    )
                assert "mean" in lines[i + 1]
                all_tokens_prob_mean = float(lines[i + 1].split()[1].strip()) * 100
                # also pull std
                assert "std" in lines[i + 2]
                all_tokens_prob_std = float(lines[i + 2].split()[1].strip()) * 100
                successfully_found += 1
            if "Unconditional PMI Accuracy:" in it:
                uncond_pmi_acc = float(it.split(":")[-1])
        if successfully_found != 2:
            if successfully_found == 0:
                print("Exists but is empty: ", logfile)
            else:
                print("ISSUE:")
                print(successfully_found)
                print(logfile)
            raise Exception

    return seq_acc, uncond_pmi_acc, all_tokens_prob_mean, all_tokens_prob_std


def compute_reordered_probabilities(
    row,
    test_hf_dataset,
    dataset_name,
    num_answer_choices,
):
    # manually compute sequence scoring prediction
    if "all_tokens_uncontextualized_prob_0" in row.keys():
        probs_list_all_tokens = [
            row[f"all_tokens_uncontextualized_prob_{idx}"]
            for idx in range(num_answer_choices)
        ]
    elif "all_tokens_prob_choice_0" in row.keys():
        probs_list_all_tokens = [
            row[f"all_tokens_prob_choice_{idx}"] for idx in range(num_answer_choices)
        ]
    else:
        breakpoint()

    # refer back to original dataset to get the exact answer choice list
    if dataset_name == "mmlu":
        # verify dataset instance matches
        ds_instance = test_hf_dataset[row.id]
        if row.question != ds_instance["question"]:
            breakpoint()
        if (
            ds_instance["choices"][ds_instance["answer"]] != row.gold_label
            and ds_instance["choices"][ds_instance["answer"]].replace(", ", " and ")
            != row.gold_label
        ):
            breakpoint()
        if (
            ds_instance["choices"][ds_instance["answer"]].replace(", ", " and ")
            == row.gold_label
        ):
            choice_list = [it.replace(", ", " and ") for it in ds_instance["choices"]]
        else:
            choice_list = ds_instance["choices"]
    else:
        if dataset_name == "commonsense_qa":
            choice_list = test_hf_dataset[
                test_hf_dataset["question"].index(row.question)
            ]["choices"]["text"]
        else:
            choice_list = test_hf_dataset[test_hf_dataset["id"].index(row.id)][
                "choices"
            ]["text"]

    # check basic stats of answer choice list
    assert len(choice_list) == len(probs_list_all_tokens) == num_answer_choices

    # verify choice list matches order of answers in prediction spreadsheets
    if ", ".join(choice_list[:-1]) + ", or " + choice_list[-1] != row.answer_choices:
        # have to re-order choice list and probs_list as they were scrambled
        shuffled_choices = row.answer_choices.split(", or ")[0].split(", ") + [
            row.answer_choices.split(", or ")[1]
        ]
        # create a proper choice list from the shuffled string
        if set(shuffled_choices) != set(choice_list):
            new_choice_list = []
            i = 0
            while len(new_choice_list) != num_answer_choices:
                if shuffled_choices[i] in choice_list:
                    new_choice_list.append(shuffled_choices[i])
                    i += 1
                else:
                    jk = shuffled_choices[i] + ", " + shuffled_choices[i + 1]
                    jl = (
                        shuffled_choices[i]
                        + ", "
                        + shuffled_choices[i + 1]
                        + ", "
                        + shuffled_choices[i + 2]
                    )
                    if jk in choice_list:
                        new_choice_list.append(jk)
                        i += 2
                    elif jl in choice_list:
                        new_choice_list.append(jl)
                        i += 3
                    else:
                        breakpoint()
            if (
                ", ".join(new_choice_list[:-1]) + ", or " + new_choice_list[-1]
                != row.answer_choices
            ):
                breakpoint()
            shuffled_choices = new_choice_list
            assert set(shuffled_choices) == set(choice_list)
        # then, re-order the scores using the dictionary
        score_mapping = {}
        for txt, sc in zip(shuffled_choices, probs_list_all_tokens):
            score_mapping[txt] = sc
        probs_list_all_tokens = [score_mapping[it] for it in choice_list]

    return choice_list, probs_list_all_tokens


def compute_all_tokens_probs_sum(row, choice_list, probs_list_all_tokens):
    unique_tokens = set(choice_list)
    if len(unique_tokens) != len(choice_list):
        # remove duplicates
        all_probs_sum = 0
        for xl in unique_tokens:
            # duplicate probs are not *totally* identical due to stochasticity in API; take the first occurrence
            all_probs_sum += probs_list_all_tokens[choice_list.index(xl)]
        assert all_probs_sum <= sum(probs_list_all_tokens)
    else:
        all_probs_sum = sum(probs_list_all_tokens)

    return all_probs_sum


def compute_pmi_accuracy(
    numerator, denominator, test_hf_dataset, dataset_name, num_answer_choices
):
    seq_acc = []
    for (_, numerator_row), (_, denominator_row) in zip(
        numerator.iterrows(), denominator.iterrows()
    ):
        # first, assert rows match
        if (
            numerator_row.id != denominator_row.id
            or numerator_row.question != denominator_row.question
            or numerator_row.gold_label != denominator_row.gold_label
        ):
            breakpoint()
            raise Exception("rows don't match")

        # get probability scores for both rows
        choice_list_num, num_probs_list = compute_reordered_probabilities(
            numerator_row, test_hf_dataset, dataset_name, num_answer_choices
        )
        choice_list_denom, denom_probs_list = compute_reordered_probabilities(
            denominator_row, test_hf_dataset, dataset_name, num_answer_choices
        )

        if choice_list_num != choice_list_denom:
            breakpoint()

        # compute PMI-normalized scores
        try:
            probs_list_all_tokens = [
                num / float(denom)
                for num, denom in zip(num_probs_list, denom_probs_list)
            ]
        except:
            # add some small amount of noise into the lists to avoid division by zero
            num_probs_list = [
                it if it != 0.0 else 0.000000000000001 for it in num_probs_list
            ]
            denom_probs_list = [
                it if it != 0.0 else 0.000000000000001 for it in denom_probs_list
            ]
            probs_list_all_tokens = [
                num / float(denom)
                for num, denom in zip(num_probs_list, denom_probs_list)
            ]

        # get all tokens prediction *after* normalization
        # used "all tokens" instead of "first token" as the seq. scoring accuracy in plotted graphs & tables, as done in prior work
        pred_l_all = choice_list_num[np.argmax(np.array(probs_list_all_tokens))]

        # score whether correct
        if pred_l_all == numerator_row.gold_label:
            seq_acc.append(1)
        else:
            seq_acc.append(0)

    return round((sum(seq_acc) / float(len(seq_acc))) * 100, 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, choices={"openbookqa", "commonsense_qa", "mmlu"}
    )
    args = parser.parse_args()
    dataset_name = args.dataset

    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    dataset_config = config[dataset_name]
    test_split = dataset_config["test_split"]
    num_answer_choices = dataset_config["num_answer_choices"]
    tasks = dataset_config["tasks"]

    if dataset_name in {"openbookqa", "commonsense_qa"}:
        num_datapoints = 500
    elif dataset_name == "mmlu":
        num_datapoints = 1140
    else:
        raise Exception("dataset not supported")

    # only load datasets once
    if dataset_name == "commonsense_qa":
        test_hf_dataset = nlp.load_dataset(dataset_name, split=test_split + "[:500]")
    elif dataset_name == "openbookqa":
        test_hf_dataset = nlp.load_dataset(
            dataset_name, "main", split=test_split + "[:500]"
        )
    elif dataset_name == "mmlu":
        test_hf_dataset = []
        for subtask in tasks:
            test_hf_dataset.extend(
                list(nlp.load_dataset("hendrycks_test", subtask, split="test[:20]"))
            )
        assert len(test_hf_dataset) == 1140

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

        # get relevant unconditional prediction files
        def get_unconditional_prediction_file_stats(prompt_type):
            preds_file = f"./data/{dataset_name}/uncontextual_preds/{dataset_name}_gpt3_generations_0primes_labelOnly_{model.replace('/','-')}_temp0_stop###_1samples_seed10_full{test_split.title()}Set_{prompt_type}Prompt.csv"
            preds = pd.read_csv(preds_file)
            # check that file is complete
            if len(preds) != num_datapoints:
                print("File does not have enough datapoints:")
                print(preds_file)
                raise Exception
            return preds

        l_only_preds = get_unconditional_prediction_file_stats("no_answer_choices")
        l_cond_L_string_preds = get_unconditional_prediction_file_stats(
            "string_answer_choices"
        )
        l_cond_L_enum_preds = get_unconditional_prediction_file_stats(
            "enumerated_answer_choices"
        )

        # get relevant conditional prediction files
        def get_conditional_prediction_file_stats(prompt_type):
            preds_file = f"./data/{dataset_name}/{dataset_name}_gpt3_generations_0primes_labelOnly_{model.replace('/','-')}_temp0_stop###_1samples_seed10_full{test_split.title()}Set_{prompt_type}Prompt.csv"
            preds = pd.read_csv(preds_file)
            # check that file is complete
            if len(preds) != num_datapoints:
                print("File does not have enough datapoints:")
                print(preds_file)
                raise Exception
            # and get logfile since we already computed seq. scoring accuracy and prob. mass
            logfile = f"./data/{dataset_name}/0_primes_{dataset_name}_{model.replace('/','-')}_seed_10_{prompt_type}Prompt.log"
            if not os.path.isfile(logfile):
                raise Exception("Logfile does not exist:", logfile)
            return preds, parse_logfile(logfile, num_datapoints)

        l_cond_Q_preds, (
            seq_acc_l_cond_Q,
            _,
            pmass_mean_l_cond_Q,
            pmass_std_l_cond_Q,
        ) = get_conditional_prediction_file_stats("no_answer_choices")
        l_cond_both_string_preds, (
            seq_acc_l_cond_both_string,
            _,
            pmass_mean_l_cond_both_string,
            pmass_std_l_cond_both_string,
        ) = get_conditional_prediction_file_stats("string_answer_choices")
        l_cond_both_enum_preds, (
            seq_acc_l_cond_both_enum,
            _,
            pmass_mean_l_cond_both_enum,
            pmass_std_l_cond_both_enum,
        ) = get_conditional_prediction_file_stats("enumerated_answer_choices")

        # loop over all 6 cases to compute 2 metrics
        raw_results = {
            r"$None$": l_only_preds,
            r"$\mathcal{L}_{string}$": l_cond_L_string_preds,
            r"$\mathcal{L}_{enum}$": l_cond_L_enum_preds,
            r"$q$": l_cond_Q_preds,
            r"$q + \mathcal{L}_{string}$": l_cond_both_string_preds,
            r"$q + \mathcal{L}_{enum}$": l_cond_both_enum_preds,
        }
        all_pmass_means = defaultdict(float)
        all_pmass_sterrs = defaultdict(float)
        all_accuracies = defaultdict(float)
        all_unconditional_pmi_accuracies = defaultdict(float)
        all_labelConditional_pmi_accuracies = defaultdict(float)
        for name, df in raw_results.items():
            seq_acc = []
            pmasses = []
            for i, row in df.iterrows():
                choice_list, probs_list_all_tokens = compute_reordered_probabilities(
                    row, test_hf_dataset, dataset_name, num_answer_choices
                )

                # score whether correct
                if "all_tokens_correct" in row.keys():
                    if row.all_tokens_correct:
                        seq_acc.append(1)
                    else:
                        seq_acc.append(0)
                else:
                    # manually compute sequence scoring prediction
                    # get all tokens prediction
                    pred_l_all = choice_list[np.argmax(np.array(probs_list_all_tokens))]

                    # score whether correct
                    if pred_l_all == row.gold_label:
                        seq_acc.append(1)
                    else:
                        seq_acc.append(0)

                # assert file has been re-parsed correctly
                if "first_tokens" not in row.keys():
                    breakpoint()

                # write summed prob mass to list
                probs_sum = compute_all_tokens_probs_sum(
                    row, choice_list, probs_list_all_tokens
                )
                pmasses.append(probs_sum)

            # compute accuracy and report
            total_acc = round((sum(seq_acc) / float(len(seq_acc))) * 100, 2)
            pmass_mean = pd.Series(pmasses).mean() * 100
            pmass_std = pd.Series(pmasses).std() * 100
            # append to dictionaries
            all_accuracies[name] = total_acc
            all_pmass_means[name] = round(pmass_mean, 2)
            all_pmass_sterrs[name] = round(pmass_std, 2)

            # compute pmi and append to dictionaries
            if name != r"$None$":
                all_unconditional_pmi_accuracies[name] = compute_pmi_accuracy(
                    df, l_only_preds, test_hf_dataset, dataset_name, num_answer_choices
                )
            else:
                all_unconditional_pmi_accuracies[name] = 0

            if name == r"$q + \mathcal{L}_{string}$":
                all_labelConditional_pmi_accuracies[name] = compute_pmi_accuracy(
                    df,
                    l_cond_L_string_preds,
                    test_hf_dataset,
                    dataset_name,
                    num_answer_choices,
                )
            elif name == r"$q + \mathcal{L}_{enum}$":
                all_labelConditional_pmi_accuracies[name] = compute_pmi_accuracy(
                    df,
                    l_cond_L_enum_preds,
                    test_hf_dataset,
                    dataset_name,
                    num_answer_choices,
                )
            else:
                all_labelConditional_pmi_accuracies[name] = 0

            # sanity-check computations against existing logfile
            if name == r"$q + \mathcal{L}_{enum}$":
                if not math.isclose(
                    total_acc, seq_acc_l_cond_both_enum, rel_tol=0.0005
                ):
                    breakpoint()
                if not math.isclose(
                    pmass_mean, pmass_mean_l_cond_both_enum, rel_tol=0.0005
                ):
                    breakpoint()
                if not math.isclose(
                    pmass_std, pmass_std_l_cond_both_enum, rel_tol=0.0005
                ):
                    breakpoint()
            elif name == r"$q + \mathcal{L}_{string}$":
                if not math.isclose(
                    total_acc, seq_acc_l_cond_both_string, rel_tol=0.0005
                ):
                    breakpoint()
                if not math.isclose(
                    pmass_mean, pmass_mean_l_cond_both_string, rel_tol=0.0005
                ):
                    breakpoint()
                if not math.isclose(
                    pmass_std, pmass_std_l_cond_both_string, rel_tol=0.0005
                ):
                    breakpoint()
            elif name == r"$q$" and seq_acc_l_cond_Q is not None:
                if not math.isclose(total_acc, seq_acc_l_cond_Q, rel_tol=0.0005):
                    breakpoint()
                if not math.isclose(pmass_mean, pmass_mean_l_cond_Q, rel_tol=0.0005):
                    breakpoint()
                if not math.isclose(pmass_std, pmass_std_l_cond_Q, rel_tol=0.0005):
                    breakpoint()

        df = pd.DataFrame(
            {
                "Model": [model] * 12,
                "Probabilistic Quantity": list(
                    flatten(
                        zip(
                            list(raw_results.keys()),
                            list(raw_results.keys()),
                        )
                    )
                ),
                "Metric": [
                    r"Acc. ($\mathbb{1}[argmax_{\ell\in\mathcal{L}}p(\ell|x) = y]$)",
                    r"Avg. $\mathrm{PMV}$ ($\sum_{\ell \in \mathcal{L}} p(\ell|x)$)",
                ]
                * 6,
                "Score": list(
                    flatten(
                        [
                            (
                                all_accuracies[name],
                                all_pmass_means[name],
                            )
                            for name in raw_results.keys()
                        ]
                    )
                ),
                "Std Err": list(
                    flatten(
                        [(0, all_pmass_sterrs[name]) for name in raw_results.keys()]
                    )
                ),
            }
        )

        # plot data
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(data=df, x="Probabilistic Quantity", y="Score", hue="Metric", ax=ax)
        if model == "facebook-opt-30b":
            ax.set_title("OPT 30B", fontsize=18)
        elif model == "google-flan-t5-xxl":
            ax.set_title("flan-t5-xxl", fontsize=18)
        else:
            ax.set_title(model, fontsize=18)
        ax.set_ylim([0, 105])
        ax.set_ylabel("")
        plt.xticks(rotation=20)
        ax.set_xlabel(r"Model Input $x$")

        if model == "curie":
            ax.legend(
                loc="upper right", bbox_to_anchor=(1, 1), prop={"size": 14}, ncol=2
            )
        else:
            ax.get_legend().remove()
        # show rounded labels on each bar
        for container in ax.containers:
            ax.bar_label(container, fmt="{:,.2f}")

        if not os.path.isdir("./plotting/independent_var_plots"):
            os.mkdir("./plotting/independent_var_plots")
        plt.savefig(
            f"./plotting/independent_var_plots/{model}_{dataset_name}_0shot.pdf",
            bbox_inches="tight",
        )
