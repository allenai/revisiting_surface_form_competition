import argparse
import pandas as pd
import os
import numpy as np


def format(x):
    if len(x) > 1 and -100 not in x:
        return "$" + str(round(np.mean(x), 2)) + "_{" + str(round(np.std(x), 2)) + "}$"
    elif -100 not in x:
        return "$" + str(round(np.mean(x), 2)) + "$"
    else:
        return ""


def bold(row, maximize=True):
    for i, el in enumerate(row):
        if maximize and el == row.values.max():
            try:
                row[i] = "$\mathbf{" + row[i].split("$")[1] + "}$"
            except:
                if row.name[0] == "google-flan-t5-xxl" and row.index[0][0] in {
                    "CommonsenseQA",
                    "OpenbookQA",
                }:
                    pass
                else:
                    breakpoint()
        if not maximize and el == row.values.min():
            try:
                row[i] = "$\mathbf{" + row[i].split("$")[1] + "}$"
            except:
                if row.name[0] == "google-flan-t5-xxl" and row.index[0][0] in {
                    "CommonsenseQA",
                    "OpenbookQA",
                }:
                    pass
                else:
                    breakpoint()
    return row


def create_table(df, table_desc):
    # don't give a confidence interval if there is only one seed
    if "accuracy" in table_desc:
        new = pd.pivot_table(
            df[df.Metric == table_desc],
            values="Value",
            index=["Model", "Prompt", "Decoder"],
            columns=["Dataset", "# In-Context Demonstrations"],
            aggfunc=lambda x: format(x),
            fill_value=0,
        )
    elif table_desc == "prob. mass":
        new = pd.pivot_table(
            df[(df.Metric == table_desc) & (df.Decoder == "greedy")],
            values="Value",
            index=["Model", "Prompt"],
            columns=["Dataset", "# In-Context Demonstrations"],
            aggfunc=lambda x: format(x),
            fill_value=0,
        )
    elif table_desc == "invalid answers":
        new = pd.pivot_table(
            df[df.Metric == table_desc],
            values="Value",
            index=["Model", "Prompt"],
            columns=["Dataset", "# In-Context Demonstrations"],
            aggfunc=lambda x: format(x),
            fill_value=0,
        )

    new.sort_values(by=["Model", "Prompt"], ascending=[True, False], inplace=True)

    # bold top values in each row for each dataset
    if table_desc != "invalid answers":
        for dataset in {"OpenbookQA", "CommonsenseQA", "MMLU"}:
            try:
                new.loc[:, pd.IndexSlice[dataset, :]] = new.loc[
                    :, pd.IndexSlice[dataset, :]
                ].apply(lambda row: bold(row), axis=1)
            except:
                pass
    else:
        for dataset in {"OpenbookQA", "CommonsenseQA", "MMLU"}:
            try:
                new.loc[:, pd.IndexSlice[dataset, :]] = new.loc[
                    :, pd.IndexSlice[dataset, :]
                ].apply(lambda row: bold(row, maximize=False), axis=1)
            except:
                pass

    return new.style.to_latex()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metric",
        type=str,
        choices={"accuracy", "PMI accuracy", "invalid answers", "prob. mass"},
    )
    args = parser.parse_args()

    dataset_list = ["openbookqa", "commonsense_qa", "mmlu"]
    cleaned_dataset_list = ["OpenbookQA", "CommonsenseQA", "MMLU"]
    clean_dataset_names = {
        "openbookqa": "OpenbookQA",
        "commonsense_qa": "CommonsenseQA",
        "mmlu": "MMLU",
    }

    dataframe_items = []
    for model in [
        "curie",
        "davinci",
        "davinci-instruct-beta",
        "facebook-opt-30b",
        "google-flan-t5-xxl",
        "text-davinci-003",
    ]:
        for prompt in [
            "_no_answer_choicesPrompt",
            "_string_answer_choicesPrompt",
            "_enumerated_answer_choicesPrompt",
        ]:
            for dataset in dataset_list:
                for num_primes in [0, 1, 2, 4, 8]:
                    # note that value for each seed should be added to a different row
                    if num_primes != 0 and model in {
                        "curie",
                        "davinci",
                        "davinci-instruct-beta",
                    }:
                        seeds = [10, 20, 30]
                    else:
                        seeds = [10]
                    for seed in seeds:
                        logfile = f"./data/{dataset}/{num_primes}_primes_{dataset}_{model}_seed_{seed}{prompt}.log"
                        if prompt == "_enumerated_answer_choicesPrompt":
                            clean_prompt = "enum ans."
                        elif prompt == "_string_answer_choicesPrompt":
                            clean_prompt = "string ans."
                        elif prompt == "_no_answer_choicesPrompt":
                            clean_prompt = "no ans."
                        clean_dataset = clean_dataset_names[dataset]
                        if os.path.isfile(logfile):
                            with open(logfile, "r") as f:
                                lines = [line.rstrip() for line in f]
                                successfully_found = 0
                                for i, it in enumerate(lines):
                                    if "Total:" in it:
                                        num = int(it.split("Total:")[-1].strip())
                                        if num != 500 and num != 1140:
                                            print(
                                                f"FILE HAS {num} DATAPOINTS INSTEAD OF 500/1140:",
                                                logfile,
                                            )
                                    if "Percent " in it:
                                        if (
                                            it.split(":")[0]
                                            == "Percent of correct greedy examples"
                                        ):
                                            g_acc = float(it.split(":")[1])
                                            successfully_found += 1
                                        elif (
                                            it.split(":")[0]
                                            == "Percent of correct sequence scoring examples"
                                        ):
                                            seq_acc = float(it.split(":")[1])
                                            successfully_found += 1
                                        elif (
                                            it.split(":")[0]
                                            == "Percent of correct PMI examples"
                                        ):
                                            pmi_acc = float(it.split(":")[1])
                                            successfully_found += 1
                                        elif (
                                            it.split(":")[0]
                                            == "Percent of invalid greedy examples"
                                        ):
                                            inv_pct = float(it.split(":")[1])
                                            successfully_found += 1
                                    if (
                                        "Distribution of FIRST-TOKEN Probability Masses:"
                                        in it
                                    ):
                                        if not (
                                            "500.0" in it
                                            or "1140.0" in it
                                            or "5.000000e+02" in it
                                            or "1.140000e+03" in it
                                        ):
                                            print("INVALID PROMPT LOG: ", logfile)
                                        assert "mean" in lines[i + 1]
                                        first_prob_mean = (
                                            float(lines[i + 1].split()[1].strip()) * 100
                                        )
                                        successfully_found += 1
                                if successfully_found != 5:
                                    if successfully_found == 0:
                                        print("Exists but is empty: ", logfile)
                                    elif successfully_found == 4:
                                        pmi_acc = None
                                        print("DOES NOT HAVE PMI:", logfile)
                                    else:
                                        print("ISSUE-- SKIPPING:", logfile)
                                        continue

                            datapoint = [
                                model,
                                clean_prompt,
                                str(num_primes),
                                clean_dataset,
                                "greedy",
                                "accuracy",
                                g_acc,
                            ]
                            dataframe_items.append(datapoint)
                            datapoint = [
                                model,
                                clean_prompt,
                                str(num_primes),
                                clean_dataset,
                                "greedy",
                                "prob. mass",
                                first_prob_mean,
                            ]
                            dataframe_items.append(datapoint)
                            datapoint = [
                                model,
                                clean_prompt,
                                str(num_primes),
                                clean_dataset,
                                "ans. scoring",
                                "accuracy",
                                seq_acc,
                            ]
                            dataframe_items.append(datapoint)
                            datapoint = [
                                model,
                                clean_prompt,
                                str(num_primes),
                                clean_dataset,
                                "ans. scoring",
                                "prob. mass",
                                first_prob_mean,
                            ]
                            dataframe_items.append(datapoint)
                            datapoint = [
                                model,
                                clean_prompt,
                                str(num_primes),
                                clean_dataset,
                                "greedy",
                                "invalid answers",
                                inv_pct,
                            ]
                            dataframe_items.append(datapoint)
                            if pmi_acc is not None:
                                datapoint = [
                                    model,
                                    clean_prompt,
                                    str(num_primes),
                                    clean_dataset,
                                    "ans. scoring",
                                    "PMI accuracy",
                                    pmi_acc,
                                ]
                                dataframe_items.append(datapoint)
                        else:
                            print(f"Does not exist: {logfile}")

    df = pd.DataFrame(
        dataframe_items,
        columns=[
            "Model",
            "Prompt",
            "# In-Context Demonstrations",
            "Dataset",
            "Decoder",
            "Metric",
            "Value",
        ],
    )

    # given dataframe, create Latex tables
    print(create_table(df, args.metric))
