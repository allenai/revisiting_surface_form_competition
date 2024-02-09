import argparse
from compare_independent_variables import parse_logfile
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import scipy as sp
from sklearn.linear_model import LinearRegression


def int_or_str(value):
    try:
        return int(value)
    except:
        return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-primes", type=int_or_str, choices={"all", 0, 1, 2, 4, 8})
    parser.add_argument(
        "--dataset", type=str, choices={"openbookqa", "commonsense_qa", "mmlu"}
    )
    args = parser.parse_args()

    if args.dataset in {"openbookqa", "commonsense_qa"}:
        num_datapoints = 500
    elif args.dataset == "mmlu":
        num_datapoints = 1140
    else:
        raise Exception("dataset not supported")

    prompt_lookup = {
        "no_answer_choices": r"$q$",
        "string_answer_choices": r"$q + \mathcal{L}_{string}$",
        "enumerated_answer_choices": r"$q + \mathcal{L}_{enum}$",
    }

    data = {
        "Model": [],
        "Prompt Format": [],
        "Accuracy": [],
        "Avg. PMA": [],
    }

    clean_model_names = {
        "text-davinci-003": "tdv-003",
        "google-flan-t5-xxl": "flan-t5-xxl",
        "davinci-instruct-beta": "dv-ins-beta",
        "davinci": "davinci",
        "curie": "curie",
        "facebook-opt-30b": "OPT 30B",
    }

    for model in clean_model_names.keys():
        for prompt_type in [
            "no_answer_choices",
            "string_answer_choices",
            "enumerated_answer_choices",
        ]:
            if args.num_primes != "all":
                clean_prompt_type = prompt_lookup[prompt_type]
                logfile = f"./data/{args.dataset}/{args.num_primes}_primes_{args.dataset}_{model.replace('/','-')}_seed_10_{prompt_type}Prompt.log"
                if not os.path.isfile(logfile):
                    raise Exception("Logfile does not exist:", logfile)
                (
                    seq_acc,
                    _,
                    all_prob_mean,
                    _,
                ) = parse_logfile(logfile, num_datapoints)

                data["Model"].append(clean_model_names[model])
                data["Prompt Format"].append(clean_prompt_type)
                data["Accuracy"].append(seq_acc)
                data["Avg. PMA"].append(all_prob_mean)

            else:
                for np in [0, 1, 2, 4, 8]:
                    clean_prompt_type = prompt_lookup[prompt_type]
                    logfile = f"./data/{args.dataset}/{np}_primes_{args.dataset}_{model.replace('/','-')}_seed_10_{prompt_type}Prompt.log"
                    if not os.path.isfile(logfile):
                        print("Logfile does not exist:", logfile)
                    else:
                        (
                            seq_acc,
                            _,
                            all_prob_mean,
                            _,
                        ) = parse_logfile(logfile, num_datapoints)

                        data["Model"].append(clean_model_names[model])
                        data["Prompt Format"].append(clean_prompt_type)
                        data["Accuracy"].append(seq_acc)
                        data["Avg. PMA"].append(all_prob_mean)

    assert (
        len(data["Model"])
        == len(data["Prompt Format"])
        == len(data["Accuracy"])
        == len(data["Avg. PMA"])
    )
    df = pd.DataFrame.from_dict(data)

    plt.rc("axes", titlesize=14)  # fontsize of the axes title
    plt.rc("axes", labelsize=14)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=14)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=14)  # fontsize of the tick labels
    plt.rc("legend", fontsize=12)  # legend fontsize

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7, 4.5))
    if args.dataset == "commonsense_qa":
        g = sns.scatterplot(
            x="Avg. PMA",
            y="Accuracy",
            data=df,
            hue="Model",
            style="Prompt Format",
            ax=axes,
            legend=None,
        )
    else:
        g = sns.scatterplot(
            x="Avg. PMA",
            y="Accuracy",
            data=df,
            hue="Model",
            style="Prompt Format",
            ax=axes,
        )

    if args.dataset == "openbookqa":
        title = "OpenbookQA"
    elif args.dataset == "mmlu":
        title = "MMLU"
    elif args.dataset == "commonsense_qa":
        title = "CommonsenseQA"
    else:
        raise Exception("invalid dataset name")
    plt.title("")
    axes.set_ylabel("Accuracy", fontsize=16)
    axes.set_xlabel("Avg. PMA", fontsize=16)
    plt.ylim([-5, 105])
    plt.xlim([-5, 105])
    plt.grid()
    if args.dataset != "commonsense_qa":
        plt.legend(loc="upper left", prop={"size": 10}, ncol=2)

    # print statistics for Table 1
    for n in df["Model"].unique():
        r, p = sp.stats.pearsonr(
            df[df["Model"] == n]["Avg. PMA"],
            df[df["Model"] == n]["Accuracy"],
        )
        print(n, "r={:.2f}, p={:.2g}".format(r, p))

    if not os.path.isdir("./plotting/scatterplots"):
        os.mkdir("./plotting/scatterplots")
    plt.savefig(
        f"./plotting/scatterplots/{args.dataset}_{args.num_primes}Shot.pdf",
        bbox_inches="tight",
    )
