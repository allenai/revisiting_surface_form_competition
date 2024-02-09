import argparse
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import os
from compare_independent_variables import parse_logfile

if __name__ == "__main__":
    task_sizes = {}
    task_sizes["openbookqa"] = 500
    task_sizes["commonsense_qa"] = 500
    task_sizes["mmlu"] = 1140

    sns.set_style(
        "whitegrid",
        rc={
            "ytick.left": True,
            "xtick.bottom": True,
            "axes.edgecolor": "black",
        },
    )

    dataset_list = ["openbookqa", "commonsense_qa", "mmlu"]
    cleaned_dataset_list = ["OpenbookQA", "CommonsenseQA", "MMLU"]
    clean_dataset_names = {
        "openbookqa": "OpenbookQA",
        "commonsense_qa": "CommonsenseQA",
        "mmlu": "MMLU",
    }
    clean_model_names = {
        "text-davinci-003": "tdv-003",
        "google-flan-t5-xxl": "flan-t5-xxl",
        "davinci-instruct-beta": "dv-ins-beta",
        "davinci": "davinci",
        "curie": "curie",
        "facebook-opt-30b": "OPT 30B",
    }

    dataframe_items = []
    for model in clean_model_names.keys():
        for prompt in [
            "_no_answer_choicesPrompt",
            "_string_answer_choicesPrompt",
            "_enumerated_answer_choicesPrompt",
        ]:
            for dataset in dataset_list:
                for num_primes in [0, 1, 2, 4, 8]:
                    # note that value for each seed should be added to a different row
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
                        logfile = f"./data/{dataset}/{num_primes}_primes_{dataset}_{model}_seed_{seed}{prompt}.log"
                        clean_dataset = clean_dataset_names[dataset]
                        if os.path.isfile(logfile):
                            (
                                seq_acc,
                                _,
                                all_prob_mean,
                                _,
                            ) = parse_logfile(logfile, task_sizes[dataset])

                            # add computed values to dataframe
                            datapoint = [
                                clean_model_names[model],
                                prompt,
                                str(num_primes),
                                clean_dataset,
                                "Accuracy",
                                seq_acc,
                            ]
                            dataframe_items.append(datapoint)
                            datapoint = [
                                clean_model_names[model],
                                prompt,
                                str(num_primes),
                                clean_dataset,
                                "Avg. PMA",
                                all_prob_mean,
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
            "Metric",
            "Value",
        ],
    )
    palette_colors = sns.color_palette("tab10")
    palette_dict = {
        model_name: color
        for model_name, color in zip(clean_model_names.values(), palette_colors)
    }
    matplotlib.rcParams.update({"font.size": 14})
    matplotlib.rcParams["figure.dpi"] = 300
    for ds in cleaned_dataset_list:
        for prompt in [
            "_no_answer_choicesPrompt",
            "_string_answer_choicesPrompt",
            "_enumerated_answer_choicesPrompt",
        ]:
            subset = df[(df["Dataset"] == ds) & (df["Prompt"] == prompt)]
            # subset should have a consistent shape now that all files have been run
            # if has 3 seeds, it's 26 points per prompt + dataset * 3 models plus 10 points * 3 models (single seed)
            fig, ax = plt.subplots(figsize=(6, 4))
            if subset.shape[0] != 108:
                print("Total datapoints should be 108, but is:", subset.shape[0])
            if prompt == "_no_answer_choicesPrompt":
                # plot legend
                ax = sns.lineplot(
                    x="# In-Context Demonstrations",
                    y="Value",
                    data=subset,
                    hue="Model",
                    style="Metric",
                    errorbar="se",
                    palette=palette_dict,
                )
                plt.legend(
                    loc="upper right", borderaxespad=0, prop={"size": 10}, ncol=2
                )
            else:
                ax = sns.lineplot(
                    x="# In-Context Demonstrations",
                    y="Value",
                    data=subset,
                    hue="Model",
                    style="Metric",
                    errorbar="se",
                    legend=None,
                    palette=palette_dict,
                )
            if prompt == "_enumerated_answer_choicesPrompt":
                ax.set_xlabel("# In-Context Demonstrations")
            else:
                ax.set_xlabel("")
            ax.set_xticks(["0", "1", "2", "4", "8"])
            ax.set_ylabel("Accuracy/Avg. PMA")

            plt.ylim(-5, 105)
            if not os.path.isdir("./plotting/line_plots"):
                os.mkdir("./plotting/line_plots")
            plt.savefig(
                f"./plotting/line_plots/{ds.lower()}_answer_scoring_accuracy_vs_probability{prompt}.pdf",
                bbox_inches="tight",
            )
            plt.clf()
