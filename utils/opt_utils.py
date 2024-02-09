import functools, operator, torch
from transformers import AutoTokenizer, AutoModelForCausalLM


@functools.cache
def load_model(checkpoint: str = "facebook/opt-1.3b"):
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        device_map="auto",
        load_in_8bit=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.truncation_side = "left"
    return {"model": model, "tokenizer": tokenizer}


def complete(
    model_name_or_path,
    prompt,
    target=None,
    generation_length=None,
    model=None,
    num_log_probs=1,
    echo=False,
    temperature=0,
):
    model = load_model(model_name_or_path)
    if generation_length == 0:
        choices = score_target(model, prompt, target, num_log_probs)
    else:
        choices = generate(model, prompt, num_log_probs, generation_length)
    return dict(model=model_name_or_path, choices=[choices])


def score_target(model, prompt, target, n_logprobs):
    text = " ".join((prompt, target))
    prompt_len = len(model["tokenizer"](text=prompt)[0])
    full_len = len(model["tokenizer"](text=text)[0])
    target_len = prompt_len - full_len
    input_ids = model["tokenizer"](
        text=text, return_tensors="pt", max_length=1024, truncation=True
    ).input_ids
    target_ids = input_ids[0, target_len:].tolist()
    with torch.inference_mode():
        out = model["model"](input_ids=input_ids.to(0))
    target_log_distributions = (
        out.logits.softmax(-1).log().roll(1, dims=1)[0, target_len:, :]
    )
    target_logprobs = list(
        map(operator.getitem, target_log_distributions.tolist(), target_ids)
    )
    target_tokens = model["tokenizer"].batch_decode(target_ids)
    topk = torch.topk(target_log_distributions, n_logprobs)
    top_logprobs = list(
        dict(zip(model["tokenizer"].batch_decode(ids), vals.tolist()))
        for ids, vals in zip(topk.indices, topk.values)
    )
    return dict(
        text=target,
        logprobs=dict(
            token_logprobs=target_logprobs,
            tokens=target_tokens,
            top_logprobs=top_logprobs,
        ),
    )


def generate(model, prompt, n_logprobs, generation_length):
    prompt = model["tokenizer"](
        text=prompt, return_tensors="pt", max_length=1024, truncation=True
    )
    with torch.inference_mode():
        out = model["model"].generate(
            inputs=prompt.input_ids.to(0),
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=generation_length,
        )
    token_list = model["tokenizer"].batch_decode(out.sequences[0])
    logprobs_ = [s[0].softmax(-1).log() for s in out.scores]
    logprobs = list(map(operator.getitem, logprobs_, out.sequences[0]))
    topk = [torch.topk(s, n_logprobs) for s in logprobs_]
    top_logprobs = list(
        dict(zip(model["tokenizer"].batch_decode(t.indices), t.values.tolist()))
        for t in topk
    )
    return dict(
        text=model["tokenizer"].decode(out.sequences[0], skip_special_tokens=True),
        logprobs=dict(
            token_logprobs=list(map(float, logprobs)),
            tokens=token_list,
            top_logprobs=top_logprobs,
        ),
    )
