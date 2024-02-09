import ast
import argparse
from collections import defaultdict
import datasets as nlp
import sys
import os
import openai
import csv
import yaml
import functools, itertools
import pandas as pd
import random
import time
from tqdm import tqdm
import numpy as np
import utils.flanT5_utils as t5
import utils.opt_utils as opt
from transformers import GPT2TokenizerFast

openai.api_key = os.getenv("OPENAI_API_KEY")

stem_mmlu = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "electrical_engineering",
    "elementary_mathematics",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_mathematics",
    "high_school_physics",
    "high_school_statistics",
    "machine_learning",
]
social_science_mmlu = [
    "econometrics",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_microeconomics",
    "high_school_psychology",
    "human_sexuality",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
]
humanities_mmlu = [
    "formal_logic",
    "high_school_european_history",
    "high_school_us_history",
    "high_school_world_history",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "moral_disputes",
    "moral_scenarios",
    "philosophy",
    "prehistory",
    "professional_law",
    "world_religions",
]
other_mmlu = [
    "business_ethics",
    "clinical_knowledge",
    "college_medicine",
    "global_facts",
    "human_aging",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "nutrition",
    "professional_accounting",
    "professional_medicine",
    "virology",
]

key_lookup = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
openai_models = [
    "curie",
    "davinci",
    "davinci-instruct-beta",
    "text-davinci-003",
]


def get_attributes(item, dataset_name, i, p_type):
    if dataset_name == "openbookqa":
        question = item["question_stem"]
    elif dataset_name in {"commonsense_qa", "mmlu"}:
        question = item["question"]
    else:
        raise Exception("invalid dataset name")

    # don't shuffle answer choices
    if dataset_name in {"openbookqa", "commonsense_qa"}:
        choice_list = item["choices"]["text"]
        answer = item["choices"]["text"][
            item["choices"]["label"].index(item["answerKey"])
        ]
        symbolic_answer = item["answerKey"]
        idx = item["id"]
    elif dataset_name == "mmlu":
        choice_list = item["choices"]
        choice_list = [it.replace(", ", " and ") for it in choice_list]
        answer = choice_list[item["answer"]]
        symbolic_answer = key_lookup[item["answer"]]
        idx = i
    else:
        raise Exception("invalid dataset name")
    choice_string = ", ".join(choice_list[:-1]) + ", or " + choice_list[-1]

    return question, answer, symbolic_answer, choice_string, choice_list, idx


def complete_gpt3(
    prompt,
    target,
    generation_length=None,
    model=None,
    num_log_probs=1,
    echo=False,
    temperature=0,
):
    """
    This function extends 1 or more GPT-3 prompts

    Inputs:
        prompt: str or [str] is a prompt or list of prompts

        generation_length: maximum length of output in tokens

        model: GPT-3 version

        num_log_probs: (*misnomer* for T5 compatibility) number of generations to produce for each prompt

        top_p: for nucleus sampling in generation

        stop: stop token

        echo: whether or not to include the input in the GPT-3 response

    Output:
        response: the raw response from GPT-3, with simple json format
            note that the 'choices' field is where generations are in the format
            [c_00, c01, ... cnm] where c_ij is the jth generation (based on input n)
            of the ith prompt
    """

    # call GPT-3 API until result is provided and then return it
    if target is not None:
        prompt = " ".join([prompt, target])
    response = None
    received = False
    while not received:
        try:
            response = openai.Completion.create(
                engine=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=generation_length,
                top_p=0,  # keep default for now
                frequency_penalty=0,
                presence_penalty=0,
                stop=["###"],
                n=num_log_probs,
                echo=echo,
                logprobs=5,
            )
            received = True
        except:
            error = sys.exc_info()[0]
            if error == openai.error.InvalidRequestError:
                # something is wrong: e.g. prompt too long
                return "error"
            else:
                print("API error:", error)
                time.sleep(0.2)
    return response


def get_answer_logprobs_qa(
    response, prompt, item, tokenizer, string_format=False, uncontextualized=False
):
    out_toks = response["choices"][0]["logprobs"]["tokens"]
    if string_format:
        if uncontextualized:
            # prompt was a single space token, so return the entire output minus one
            if "".join(out_toks[1:]) != f" {item}" and "bytes" not in "".join(
                out_toks[1:]
            ):
                breakpoint()
            probs_subset = response["choices"][0]["logprobs"]["token_logprobs"][1:]
            words_subset = out_toks[1:]
            assert len(probs_subset) == len(words_subset)
            return probs_subset, words_subset
        else:
            # return the last token position
            idx = len(out_toks) - 1

            # if answer is not just one word
            if "".join(out_toks[idx:]) != f" {item}":
                # isolate answer by re-tokenizing prompt
                idx = len(tokenizer.tokenize(prompt))

            if "".join(out_toks[idx:]) != f" {item}" and "bytes" not in "".join(
                out_toks[idx:]
            ):
                if "".join(out_toks[idx - 1 :]) == f" {item}":
                    idx = idx - 1
                else:
                    breakpoint()
            # get logprobs for the answer indices only
            probs_subset = response["choices"][0]["logprobs"]["token_logprobs"][idx:]
            words_subset = out_toks[idx:]
            assert len(probs_subset) == len(words_subset)
    else:
        indices = [i for i in range(len(out_toks)) if out_toks[i] == " correct"]
        try:
            inx1 = indices[-1]
        except:
            breakpoint()
        # validate have isolated answer
        if "".join(out_toks[inx1 + 4 :]) != f" {item}" and "bytes" not in "".join(
            out_toks[inx1 + 4 :]
        ):
            if item == "correct":
                inx1 = inx1 - 4
            elif len(indices) > 1:
                inx1 = indices[-2]
            if "".join(out_toks[inx1 + 4 :]) != f" {item}":
                print("same?")
                print("".join(out_toks[inx1 + 4 :]))
                print(f" {item}")
                breakpoint()
        # get logprobs for the same indices
        probs_subset = response["choices"][0]["logprobs"]["token_logprobs"][inx1 + 4 :]
        words_subset = out_toks[inx1 + 4 :]
    return probs_subset, words_subset


def augment_prompt(
    prompt,
    q_type,
    quest,
    choice_string,
    choice_list,
    num_answer_choices,
    uncontextual_premise=False,
):
    if q_type == "string_answer_choices":
        if uncontextual_premise:
            # don't include in-context examples or question
            custom_prompt = f"answer choices: {choice_string}\nThe correct answer is:"
        else:
            custom_prompt = (
                prompt
                + f"question: {quest}\nanswer choices: {choice_string}\nThe correct answer is:"
            )
    elif q_type == "enumerated_answer_choices":
        if uncontextual_premise:
            # don't include in-context examples or question
            custom_prompt = "Choices:\n"
        else:
            custom_prompt = prompt + f"Question: {quest}\nChoices:\n"
        assert len(choice_list) == num_answer_choices
        if num_answer_choices == 4:
            custom_prompt += f" A: {choice_list[0]}\n B: {choice_list[1]}\n C: {choice_list[2]}\n D: {choice_list[3]}\n"
        elif num_answer_choices == 5:
            custom_prompt += f" A: {choice_list[0]}\n B: {choice_list[1]}\n C: {choice_list[2]}\n D: {choice_list[3]}\n E: {choice_list[4]}\n"
        custom_prompt += f"Answer:"
    elif q_type == "no_answer_choices":
        if uncontextual_premise:
            # don't include in-context examples or question
            # however GPT-3 API requires at least one token in prompt to score answer probability correctly
            custom_prompt = "?"
        else:
            custom_prompt = prompt + quest

    return custom_prompt


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


if __name__ == "__main__":
    args = get_args()

    # get args/define key values
    num_primes = args.num_primes
    model = args.model
    complete = (
        functools.partial(opt.complete, model)
        if "facebook" in model
        else functools.partial(t5.complete, model, tuple(range(args.gpus)))
        if "google" in model
        else complete_gpt3
    )
    random_seed = args.random_seed
    dataset_name = args.dataset
    q_type = args.prompt_format
    if (args.uncontextual_premise and num_primes != 0) or (
        args.uncontextual_premise and random_seed != 10
    ):
        raise Exception(
            "No need to run the uncontextual premise with >0 shots or other random seeds, as it doesn't take in-context examples."
        )
    if model == "text-davinci-003" and random_seed != 10:
        raise Exception("bad seed-model combo")

    if type(random_seed) is int:
        random.seed(random_seed)
    else:
        random.seed(10)

    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    dataset_config = config[dataset_name]
    test_split = dataset_config["test_split"]
    num_answer_choices = dataset_config["num_answer_choices"]
    keyword = dataset_config["keyword"]
    keylist = dataset_config["keylist"]
    instruction = dataset_config["instruction"]
    tasks = dataset_config["tasks"]

    if args.uncontextual_premise:
        outfile = f"{args.outdir}/{dataset_name}/uncontextual_preds/{dataset_name}_gpt3_generations_{num_primes}primes_labelOnly_{model.replace('/','-')}_temp0_stop###_1samples_seed{random_seed}_full{test_split.title()}Set_{q_type}Prompt.csv"
        logfile = f"{args.outdir}/{dataset_name}/uncontextual_preds/{num_primes}_primes_{dataset_name}_{model.replace('/','-')}_seed_{random_seed}_{q_type}Prompt.log"
    else:
        outfile = f"{args.outdir}/{dataset_name}/{dataset_name}_gpt3_generations_{num_primes}primes_labelOnly_{model.replace('/','-')}_temp0_stop###_1samples_seed{random_seed}_full{test_split.title()}Set_{q_type}Prompt.csv"
        logfile = f"{args.outdir}/{dataset_name}/{num_primes}_primes_{dataset_name}_{model.replace('/','-')}_seed_{random_seed}_{q_type}Prompt.log"

    if dataset_name in {"openbookqa", "commonsense_qa"}:
        num_datapoints = 500
    elif dataset_name == "mmlu":
        num_datapoints = 1140
    else:
        raise Exception("dataset not supported")

    if args.complete_existing and os.path.isfile(outfile):
        # load existing CSV file as pandas dataframe
        no_load = False
        try:
            current_datapoints = pd.read_csv(outfile)
        except:
            no_load = True
        if no_load:
            print(f"Computing {num_datapoints} (all) datapoints for file {outfile}")
            current_datapoints = None
            if (
                os.path.isfile(outfile) or os.path.isfile(logfile)
            ) and not args.overwrite:
                val = input("File already exists (but is empty). Overwrite? [y/n] ")
                if val.lower() not in ("yes", "y"):
                    raise Exception("Exit")
        else:
            if len(current_datapoints["id"].unique()) >= num_datapoints:
                raise Exception(f"file is already complete: {outfile}")
            else:
                x = num_datapoints - len(current_datapoints["id"].unique())
                print(f"Computing {x} more datapoints for file {outfile}")
    else:
        current_datapoints = None
        print(f"Computing {num_datapoints} (all) datapoints for file {outfile}")
        if (os.path.isfile(outfile) or os.path.isfile(logfile)) and not args.overwrite:
            val = input("File already exists. Overwrite? [y/n] ")
            if val.lower() not in ("yes", "y"):
                raise Exception("Exit")

        os.makedirs(os.path.dirname(outfile), exist_ok=True)

    # write out for annotation
    if args.complete_existing and current_datapoints is not None:
        # append to existing file; don't rewrite existing header
        f = open(outfile, "a")
        writer = csv.writer(f)
    elif args.uncontextual_premise:
        f = open(outfile, "w")
        writer = csv.writer(f)
        writer.writerow(
            [
                "id",
                "task",
                "question",
                "answer_choices",
                "first_tokens",
                "gold_label",
            ]
            + [
                f"first_token_uncontextualized_prob_{i}"
                for i in range(num_answer_choices)
            ]
            + [
                f"all_tokens_uncontextualized_prob_{i}"
                for i in range(num_answer_choices)
            ]
            + ["prompt"]
        )
    else:
        # write to new file with header
        f = open(outfile, "w")
        writer = csv.writer(f)
        writer.writerow(
            [
                "id",
                "task",
                "question",
                "answer_choices",
                "first_tokens",
                "gold_label",
                "predicted_label_greedy_unprocessed",
                "predicted_label_greedy_processed",
                "greedy_processed_invalid",
                "greedy_processed_correct",
                "predicted_label_first_token",
                "first_token_correct",
                "predicted_label_all_tokens",
                "all_tokens_correct",
                "greedy_and_all_tokens_seq_pred_same",
                "greedy_prob_string_unprocessed",
                "greedy_prob_string_processed",
                "greedy_total_prob_processed",
                "greedy_top_5_logprobs",
            ]
            + [f"first_token_prob_choice_{i}" for i in range(num_answer_choices)]
            + ["sum_first_token_probs"]
            + [f"all_tokens_prob_choice_{i}" for i in range(num_answer_choices)]
            + ["sum_all_token_probs", "prompt"]
        )

    # load dataset, minus instances (maybe) used as primes
    if dataset_name in {"commonsense_qa", "openbookqa"}:
        prime_set = nlp.load_dataset(dataset_name, split="train")
        dataset = nlp.load_dataset(dataset_name, split=test_split + "[:500]")
        assert len(dataset) == 500
        prompt_template = None

    # initialize storage vectors for results
    greedy_acc, seq_acc, seq_fo_acc, inv_cnt = [], [], [], []
    # store probability mass sums
    p_masses_first, p_masses_all = [], []
    confusion_matrix = defaultdict(int)
    cnt_same = 0

    # load tokenizer
    if q_type == "no_answer_choices" and not args.uncontextual_premise:
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    else:
        tokenizer = None

    # iterate through all sub-tasks (1 for all except MMLU, which has 57)
    for subtask_idx, subtask in tqdm(enumerate(tasks), total=len(tasks)):
        if args.complete_existing and current_datapoints is not None:
            if dataset_name == "mmlu":
                subset = current_datapoints[current_datapoints["task"] == subtask]
                if len(subset) == 20:
                    continue
            else:
                subset = current_datapoints

        if dataset_name == "mmlu":
            # load sub-dataset and prompt subset
            try:
                val_sets = nlp.load_dataset(
                    "hendrycks_test", subtask, split=["dev", "validation"]
                )
            except:
                raise Exception(
                    "upgrade datasets package with `pip install -U datasets`"
                )
            prime_set = nlp.concatenate_datasets([val_sets[0], val_sets[1]])
            try:
                dataset = nlp.load_dataset("hendrycks_test", subtask, split="test[:20]")
            except:
                raise Exception(
                    "upgrade datasets package with `pip install -U datasets`"
                )
            assert len(dataset) == 20

        # construct prompt (shared by all instances for this task)
        if q_type == "string_answer_choices":
            prompt = f"Let's answer {keyword} questions.\n\n"
        elif q_type == "enumerated_answer_choices":
            if dataset_name == "mmlu":
                task_specific_instruction = subtask.replace("_", " ")
                prompt = f"The following are multiple-choice questions about {task_specific_instruction}. {instruction}\n\n"
            else:
                prompt = f"{instruction}\n\n"
        elif q_type == "no_answer_choices":
            prompt = ""

        if prime_set is not None:
            # select prime indices
            # randomly shuffle the whole list for consistency across runs of script with different # of prompt examples
            shuffled_prime_indices = random.sample(
                [i for i in range(len(prime_set))], len(prime_set)
            )
            selected_prime_indices = shuffled_prime_indices[:num_primes]
            selected_primes = [prime_set[i] for i in selected_prime_indices]

            for item in selected_primes:
                (
                    question,
                    answer,
                    symbolic_answer,
                    choice_string,
                    choice_list,
                    _,
                ) = get_attributes(
                    item,
                    dataset_name,
                    0,
                    q_type,
                )
                if q_type == "string_answer_choices":
                    prompt += f"question: {question}\nanswer choices: {choice_string}\nThe correct answer is: {answer}\n\n"
                elif q_type == "enumerated_answer_choices":
                    prompt += f"Question: {question}\nChoices:\n"
                    assert len(choice_list) == num_answer_choices
                    if num_answer_choices == 4:
                        prompt += f" A: {choice_list[0]}\n B: {choice_list[1]}\n C: {choice_list[2]}\n D: {choice_list[3]}\n"
                    elif num_answer_choices == 5:
                        prompt += f" A: {choice_list[0]}\n B: {choice_list[1]}\n C: {choice_list[2]}\n D: {choice_list[3]}\n E: {choice_list[4]}\n"
                    else:
                        raise Exception()
                    prompt += f"Answer: {symbolic_answer}\n\n"
                elif q_type == "no_answer_choices":
                    prompt += f"{question} {answer}\n"

        for row_idx, inst in tqdm(enumerate(dataset), total=len(dataset)):
            if (
                args.complete_existing
                and current_datapoints is not None
                and len(subset) != 0
            ):
                # skip already-run indices
                if row_idx + (20 * subtask_idx) in subset.index:
                    if dataset_name in {"commonsense_qa", "mmlu"}:
                        if not inst["question"] in subset["question"].values:
                            breakpoint()
                    elif dataset_name == "openbookqa":
                        if not inst["id"] in subset["id"].values:
                            breakpoint()
                    continue

            if dataset_name in {"commonsense_qa", "openbookqa", "mmlu"}:
                (
                    quest,
                    gold_l,
                    symbolic_answer,
                    choice_string,
                    choice_list,
                    idx,
                ) = get_attributes(
                    inst, dataset_name, row_idx + (20 * subtask_idx), q_type
                )
                # construct prompt
                if args.uncontextual_premise:
                    custom_prompt = augment_prompt(
                        prompt,
                        q_type,
                        quest,
                        choice_string,
                        choice_list,
                        num_answer_choices,
                        uncontextual_premise=True,
                    )
                else:
                    custom_prompt = augment_prompt(
                        prompt,
                        q_type,
                        quest,
                        choice_string,
                        choice_list,
                        num_answer_choices,
                        uncontextual_premise=False,
                    )

            if row_idx == 0 and subtask_idx == 0:
                print("#######################")
                print(custom_prompt)
                print("#######################")
                og_custom_prompt = custom_prompt

            # for each answer choice, score string probability under the model
            probs_list_all_tokens = []
            probs_list_first_tokens = []
            words_first_tokens = []
            if q_type in {"string_answer_choices", "no_answer_choices"}:
                for item in choice_list:
                    response = complete(
                        custom_prompt,
                        item,
                        generation_length=0,
                        model=model,
                        num_log_probs=1,
                        echo=True,
                        temperature=0,
                    )
                    while response == "error":
                        if not args.uncontextual_premise:
                            # reformat prompt (too long) and try again
                            print("Pruning prompt...")
                            prompt = "question:".join(prompt.split("question:")[:-1])
                            print("New prompt length =", len(prompt))
                            custom_prompt = augment_prompt(
                                prompt,
                                q_type,
                                quest,
                                choice_string,
                                choice_list,
                                num_answer_choices,
                            )
                            response = complete(
                                custom_prompt,
                                item,
                                generation_length=0,
                                model=model,
                                num_log_probs=1,
                                echo=True,
                                temperature=0,
                            )
                        else:
                            print("Uncontextual premise has an issue:", custom_prompt)
                            breakpoint()

                    # find indices for log probs of the answer choice alone
                    if (
                        dataset_name in {"commonsense_qa", "openbookqa", "mmlu"}
                        and model in openai_models
                    ):
                        probs_subset, words_subset = get_answer_logprobs_qa(
                            response,
                            custom_prompt,
                            item,
                            tokenizer,
                            string_format=True
                            if q_type == "no_answer_choices"
                            else False,
                            uncontextualized=args.uncontextual_premise,
                        )
                    elif "/" in model:
                        probs_subset = response["choices"][0]["logprobs"][
                            "token_logprobs"
                        ]
                        words_subset = response["choices"][0]["logprobs"]["tokens"]
                    else:
                        raise Exception(
                            "function not build for current model + task combination"
                        )
                    probs_list_all_tokens.append(np.exp(sum(probs_subset)))
                    probs_list_first_tokens.append(np.exp(probs_subset[0]))
                    words_first_tokens.append(words_subset[0])

                if not args.uncontextual_premise:
                    # also get greedy answer
                    response = complete(
                        custom_prompt,
                        None,
                        generation_length=50,
                        model=model,
                        num_log_probs=1,
                        echo=False,
                        temperature=0,
                    )
                    while response == "error":
                        # reformat prompt (too long) and try again
                        print("Pruning prompt...")
                        prompt = "question:".join(prompt.split("question:")[:-1])
                        custom_prompt = augment_prompt(
                            prompt,
                            q_type,
                            quest,
                            choice_string,
                            choice_list,
                            num_answer_choices,
                        )
                        response = complete(
                            custom_prompt,
                            None,
                            generation_length=50,
                            model=model,
                            num_log_probs=1,
                            echo=False,
                            temperature=0,
                        )

                    if len(response["choices"]) > 1:
                        raise Exception("code can currently only handle 1 sample")
                    choice = response["choices"][0]
                    pred_l_greedy_unprocessed = choice["text"].replace("\n", "\\n")
                    greedy_probs_unprocessed = choice["logprobs"]["token_logprobs"]
                    pred_l_greedy_processed = (
                        choice["text"].split("\n")[0].strip("\n").strip(".").strip()
                    )
                    # get greedy prob string
                    out_toks = choice["logprobs"]["tokens"]
                    if "/" not in model:
                        if "<|endoftext|>" in out_toks and "\n" in out_toks:
                            inxE = out_toks.index("<|endoftext|>")
                            inxL = out_toks.index("\n") - 1
                            inx1 = min([inxE, inxL])
                        else:
                            try:
                                inx1 = out_toks.index("\n") - 1
                            except:
                                try:
                                    inx1 = out_toks.index("<|endoftext|>")
                                except:
                                    inx1 = len(out_toks)
                        # validate have isolated answer
                        if "".join(
                            out_toks[:inx1]
                        ) != f" {pred_l_greedy_processed}" and "bytes" not in "".join(
                            out_toks[:inx1]
                        ):
                            # fix specific parsing errors
                            if (
                                "".join(out_toks[: inx1 + 1])
                                == f" {pred_l_greedy_processed}"
                            ):
                                # no period in generated answer output, index one term more
                                inx1 = inx1 + 1
                            elif (
                                not "".join(out_toks[: inx1 + 1])
                                and not pred_l_greedy_processed
                            ):
                                # valid match (both empty strings), continue
                                inx1 = inx1
                            elif (
                                "".join(out_toks[: inx1 - 1])
                                == f" {pred_l_greedy_processed}"
                            ) and (out_toks[inx1 - 1] == "."):
                                # last character is a period not in original answer
                                inx1 = inx1 - 1
                            elif (
                                "".join(out_toks[:inx1])
                                == f" {pred_l_greedy_processed}"[:-1]
                            ):
                                inx1 = inx1
                            elif (
                                "".join(out_toks[: inx1 - 1])
                                == f" {pred_l_greedy_processed}"
                            ):
                                inx1 = inx1 - 1
                            else:
                                print("Considered same")
                                print("".join(out_toks[:inx1]))
                                print(f" {pred_l_greedy_processed}")
                        # get logprobs for the same indices
                        greedy_probs_processed = greedy_probs_unprocessed[:inx1]
                        greedy_total_prob = np.exp(sum(greedy_probs_processed))
                        top_logprobs_subset = choice["logprobs"]["top_logprobs"][:inx1]
                        greedy_top_5_logprobs = [
                            dict(item) for item in top_logprobs_subset
                        ]
                    elif "/" in model:
                        out_toks = choice["logprobs"]["tokens"]
                        logprobs = choice["logprobs"]["token_logprobs"]
                        top_logprobs = choice["logprobs"]["top_logprobs"]
                        drop_pad = itertools.dropwhile(
                            lambda x, *_: x in ("<pad>",),
                            zip(out_toks, logprobs, top_logprobs),
                        )
                        toks, probs, top_probs = zip(
                            *itertools.takewhile(
                                lambda x, *_: x not in ("</s>",), drop_pad
                            )
                        )
                        print(toks, probs, top_probs)
                        greedy_probs_processed = probs
                        greedy_total_prob = np.exp(sum(greedy_probs_processed))
                        top_logprobs_subset = top_probs
                        greedy_top_5_logprobs = top_probs

            elif q_type == "enumerated_answer_choices":
                # only run 1-token greedy decoding
                response = complete(
                    custom_prompt,
                    None,
                    generation_length=1,
                    model=model,
                    num_log_probs=1,
                    echo=False,
                    temperature=0,
                )
                while response == "error":
                    if not args.uncontextual_premise:
                        # reformat prompt (too long) and try again
                        prompt = "Question:".join(prompt.split("Question:")[:-1])
                        custom_prompt = augment_prompt(
                            prompt,
                            q_type,
                            quest,
                            choice_string,
                            choice_list,
                            num_answer_choices,
                        )
                        response = complete(
                            custom_prompt,
                            None,
                            generation_length=1,
                            model=model,
                            num_log_probs=1,
                            echo=False,
                            temperature=0,
                        )
                    else:
                        print("Uncontextual premise too long:", custom_prompt)
                        breakpoint()

                if len(response["choices"]) > 1:
                    raise Exception("code can currently only handle 1 sample")
                choice = response["choices"][0]
                pred_l_greedy_unprocessed = choice["text"].replace("\n", "\\n")
                greedy_probs_unprocessed = choice["logprobs"]["token_logprobs"]
                try:
                    pred_l_greedy_processed = choice_list[
                        keylist.index(
                            choice["text"].split("\n")[0].strip("\n").strip(".").strip()
                        )
                    ]
                except:
                    pred_l_greedy_processed = (
                        choice["text"].split("\n")[0].strip("\n").strip(".").strip()
                    )
                greedy_probs_processed = greedy_probs_unprocessed
                greedy_total_prob = np.e ** (sum(greedy_probs_processed))

                greedy_top_5_logprobs = dict(choice["logprobs"]["top_logprobs"][0])
                for sym in keylist:
                    words_first_tokens.append(f" {sym}")
                    if f" {sym}" in greedy_top_5_logprobs:
                        tmp_val = np.e ** greedy_top_5_logprobs[f" {sym}"]
                        probs_list_first_tokens.append(tmp_val)
                        probs_list_all_tokens.append(tmp_val)
                    else:
                        # run custom answer-specific scoring under LM
                        response = complete(
                            custom_prompt,
                            sym,
                            generation_length=0,
                            model=model,
                            num_log_probs=1,
                            echo=True,
                            temperature=0,
                        )
                        out_toks = response["choices"][0]["logprobs"]["tokens"]
                        if out_toks[-1] not in (f" {sym}", sym):
                            breakpoint()
                        sc = response["choices"][0]["logprobs"]["token_logprobs"][-1]
                        probs_list_first_tokens.append(np.e**sc)
                        probs_list_all_tokens.append(np.e**sc)

                # format identically to other prompt case
                greedy_top_5_logprobs = [greedy_top_5_logprobs]

                if not (
                    len(probs_list_all_tokens)
                    == len(probs_list_first_tokens)
                    == num_answer_choices
                ):
                    breakpoint()

            if not args.uncontextual_premise:
                # get sequence scoring predictions
                pred_l_all = choice_list[np.argmax(np.array(probs_list_all_tokens))]
                pred_l_first = choice_list[np.argmax(np.array(probs_list_first_tokens))]

                normalized_pred_l_greedy_processed = pred_l_greedy_processed
                if dataset_name == "mmlu":
                    normalized_choice_list = [
                        it.lower().strip(".") for it in choice_list
                    ]
                else:
                    normalized_choice_list = choice_list
                normalized_gold_l = gold_l

                # add accuracy scores
                if normalized_pred_l_greedy_processed not in normalized_choice_list:
                    greedy_invalid = True
                    inv_cnt.append(1)
                    # automatically assume false
                    greedy_correct = False
                    greedy_acc.append(0)
                else:
                    greedy_invalid = False
                    inv_cnt.append(0)
                    if normalized_pred_l_greedy_processed == normalized_gold_l:
                        greedy_correct = True
                        greedy_acc.append(1)
                    else:
                        greedy_correct = False
                        greedy_acc.append(0)

                if pred_l_all == gold_l:
                    pred_l_all_correct = True
                    seq_acc.append(1)
                else:
                    pred_l_all_correct = False
                    seq_acc.append(0)

                if pred_l_first == gold_l:
                    pred_l_first_correct = True
                    seq_fo_acc.append(1)
                else:
                    pred_l_first_correct = False
                    seq_fo_acc.append(0)

                if pred_l_all == pred_l_greedy_processed:
                    cnt_same += 1
                    same = True
                else:
                    same = False

                # compare predictions for confusion matrix
                if greedy_correct and pred_l_all_correct:
                    confusion_matrix["both correct"] += 1
                elif greedy_correct and not pred_l_all_correct:
                    confusion_matrix["only greedy correct"] += 1
                elif pred_l_all_correct and not greedy_correct:
                    if greedy_invalid:
                        confusion_matrix["seq correct; greedy invalid"] += 1
                    elif not greedy_invalid:
                        confusion_matrix["seq correct; greedy incorrect but valid"] += 1
                    else:
                        breakpoint()
                elif not greedy_correct and not pred_l_all_correct:
                    if greedy_invalid:
                        confusion_matrix["both incorrect; greedy invalid"] += 1
                    elif not greedy_invalid:
                        confusion_matrix["both incorrect; greedy valid"] += 1
                    else:
                        breakpoint()
                else:
                    breakpoint()

                # write summed prob mass to list-- first tokens
                unique_tokens = set(words_first_tokens)
                if len(unique_tokens) != len(words_first_tokens):
                    # remove duplicates
                    first_probs_sum = 0
                    for xl in unique_tokens:
                        # duplicate probs are not *totally* identical due to stochasticity in API; take the first occurrence
                        first_probs_sum += probs_list_first_tokens[
                            words_first_tokens.index(xl)
                        ]
                    p_masses_first.append(first_probs_sum)
                else:
                    first_probs_sum = sum(probs_list_first_tokens)
                    p_masses_first.append(first_probs_sum)
                if 0 > first_probs_sum > 1:
                    breakpoint()

                # all tokens
                unique_tokens = set(choice_list)
                if len(unique_tokens) != len(choice_list):
                    # remove duplicates
                    all_probs_sum = 0
                    for xl in unique_tokens:
                        # duplicate probs are not *totally* identical due to stochasticity in API; take the first occurrence
                        all_probs_sum += probs_list_all_tokens[choice_list.index(xl)]
                    p_masses_all.append(all_probs_sum)
                else:
                    all_probs_sum = sum(probs_list_all_tokens)
                    p_masses_all.append(all_probs_sum)
                if 0 > all_probs_sum > 1:
                    breakpoint()

            if args.uncontextual_premise:
                writer.writerow(
                    [
                        idx,
                        subtask,
                        quest,
                        choice_string,
                        words_first_tokens,
                        gold_l,
                    ]
                    + [probs_list_first_tokens[i] for i in range(num_answer_choices)]
                    + [probs_list_all_tokens[i] for i in range(num_answer_choices)]
                    + [custom_prompt.replace("\n", "\\n")]
                )
            else:
                writer.writerow(
                    [
                        idx,
                        subtask,
                        quest,
                        choice_string,
                        words_first_tokens,
                        gold_l,
                        pred_l_greedy_unprocessed,
                        pred_l_greedy_processed,
                        greedy_invalid,
                        greedy_correct,
                        pred_l_first,
                        pred_l_first_correct,
                        pred_l_all,
                        pred_l_all_correct,
                        same,
                        greedy_probs_unprocessed,
                        greedy_probs_processed,
                        greedy_total_prob,
                        greedy_top_5_logprobs,
                    ]
                    + [probs_list_first_tokens[i] for i in range(num_answer_choices)]
                    + [first_probs_sum]
                    + [probs_list_all_tokens[i] for i in range(num_answer_choices)]
                    + [all_probs_sum, custom_prompt.replace("\n", "\\n")]
                )

    f.close()

    if not args.uncontextual_premise:
        # compute final accuracy
        with open(logfile, "a") as f:
            print("Output file: %s" % outfile, file=f)
            try:
                print("#######################", file=f)
                print(og_custom_prompt, file=f)
                print("#######################", file=f)
            except:
                pass
            print(
                "Number of examples where predictions are the same:", cnt_same, file=f
            )
            print(
                "Percent of examples where predictions are the same:",
                round((cnt_same / float(len(greedy_acc))) * 100, 2),
                file=f,
            )
            print("Number of examples:", len(greedy_acc), file=f)
            print("Number of correct greedy examples:", sum(greedy_acc), file=f)
            print("Number of invalid greedy examples:", sum(inv_cnt), file=f)
            print("Number of correct sequence scoring examples:", sum(seq_acc), file=f)
            print(
                "Number of correct sequence scoring examples-- First Token Only:",
                sum(seq_fo_acc),
                file=f,
            )
            print(
                "Percent of correct greedy examples:",
                round((sum(greedy_acc) / float(len(greedy_acc))) * 100, 2),
                file=f,
            )
            print(
                "Percent of invalid greedy examples:",
                round((sum(inv_cnt) / float(len(inv_cnt))) * 100, 2),
                file=f,
            )
            print(
                "Percent of correct sequence scoring examples:",
                round((sum(seq_acc) / float(len(seq_acc))) * 100, 2),
                file=f,
            )
            print(
                "Percent of correct sequence scoring examples-- First Token Only:",
                round((sum(seq_fo_acc) / float(len(seq_fo_acc))) * 100, 2),
                file=f,
            )
            print("Confusion Matrix of Predictions: ", file=f)
            print(
                "                             |------------------|------------------|------------------|",
                file=f,
            )
            print(
                "          Method:            |                    Sequence Scoring                    |",
                file=f,
            )
            print(
                "                             |--------------------------------------------------------|",
                file=f,
            )
            print(
                "                             |     Correct      | Incorrect Valid  | Incorr. Invalid* |",
                file=f,
            )
            print(
                "----------|------------------|------------------|------------------|------------------|",
                file=f,
            )
            print(
                f"          |      Correct     |       {confusion_matrix['both correct']}        |       {confusion_matrix['only greedy correct']}        |       0        |",
                file=f,
            )
            print(
                "  Greedy  |------------------|------------------|------------------|------------------|",
                file=f,
            )
            print(
                f" Decoding | Incorrect Valid  |       {confusion_matrix['seq correct; greedy incorrect but valid']}        |       {confusion_matrix['both incorrect; greedy valid']}        |       0        |",
                file=f,
            )
            print(
                "          |------------------|------------------|------------------|------------------|",
                file=f,
            )
            print(
                f"          | Incorr. Invalid* |       {confusion_matrix['seq correct; greedy invalid']}        |       {confusion_matrix['both incorrect; greedy invalid']}        |       0        |",
                file=f,
            )
            print(
                "----------|------------------|------------------|------------------|------------------|",
                file=f,
            )
            print(
                "*Sequence Scoring is always valid. Currently not manually evaluating greedy answers for validity.",
                file=f,
            )
            print("Total:", len(greedy_acc), file=f)
            try:
                assert (
                    len(greedy_acc)
                    == len(seq_acc)
                    == len(seq_fo_acc)
                    == sum(confusion_matrix.values())
                )
            except:
                breakpoint()

            print("Confusion Matrix (Percentages): ", file=f)
            print(
                "                             |------------------|------------------|------------------|",
                file=f,
            )
            print(
                "          Method:            |                    Sequence Scoring                    |",
                file=f,
            )
            print(
                "                             |--------------------------------------------------------|",
                file=f,
            )
            print(
                "                             |     Correct      | Incorrect Valid  | Incorr. Invalid* |",
                file=f,
            )
            print(
                "----------|------------------|------------------|------------------|------------------|",
                file=f,
            )
            print(
                f"          |      Correct     |       {round((confusion_matrix['both correct'] / float(len(greedy_acc))) * 100, 2)}        |       {round((confusion_matrix['only greedy correct'] / float(len(greedy_acc))) * 100, 2)}        |       0        |",
                file=f,
            )
            print(
                "  Greedy  |------------------|------------------|------------------|------------------|",
                file=f,
            )
            print(
                f" Decoding | Incorrect Valid  |       {round((confusion_matrix['seq correct; greedy incorrect but valid'] / float(len(greedy_acc))) * 100, 2)}        |       {round((confusion_matrix['both incorrect; greedy valid'] / float(len(greedy_acc))) * 100, 2)}        |       0        |",
                file=f,
            )
            print(
                "          |------------------|------------------|------------------|------------------|",
                file=f,
            )
            print(
                f"          | Incorr. Invalid* |       {round((confusion_matrix['seq correct; greedy invalid'] / float(len(greedy_acc))) * 100, 2)}        |       {round((confusion_matrix['both incorrect; greedy invalid'] / float(len(greedy_acc))) * 100, 2)}        |       0        |",
                file=f,
            )
            print(
                "----------|------------------|------------------|------------------|------------------|",
                file=f,
            )
            print(
                "Total Percent:",
                sum(
                    [
                        (it / float(len(greedy_acc))) * 100
                        for it in confusion_matrix.values()
                    ]
                ),
                file=f,
            )
            print(
                "Distribution of FIRST-TOKEN Probability Masses:",
                pd.Series(p_masses_first).describe(),
                file=f,
            )
            print(
                "Distribution of all-token 'Probability Masses':",
                pd.Series(p_masses_all).describe(),
                file=f,
            )
