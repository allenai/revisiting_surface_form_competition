import functools, operator, torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# Load model and tokenizer, also return the cuda device used for input to model
@functools.cache
def load_model(model_name_or_path: str, cuda_devices: tuple[int, ...] = None):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, device_map="auto")
    return {"model": model, "tokenizer": tokenizer}


def complete(
    model_name_or_path,
    cuda_devices,
    prompt,
    target=None,
    generation_length=None,
    model=None,
    num_log_probs=1,
    echo=False,
    temperature=0,
):
    model = load_model(model_name_or_path, cuda_devices)
    prompt = model["tokenizer"](
        text=prompt,
        truncation=False,
        return_tensors="pt",
        padding=True,
        add_special_tokens=True,
        max_length=None,
        return_length=False,
    ).to("cuda")
    if not generation_length:
        choices = score_target(model, prompt, target, num_log_probs)
    else:
        choices = generate(model, prompt, num_log_probs)
    # Mimic openai response object
    return dict(model=model_name_or_path, choices=[choices])


def score_target(model, prompt, target, n_logprobs):
    target_tokens = model["tokenizer"](
        text=target,
        truncation=False,
        return_tensors="pt",
        padding=True,
        add_special_tokens=True,
        max_length=None,
        return_length=False,
    ).to("cuda")
    ids = target_tokens.input_ids[0].tolist()
    with torch.inference_mode():
        out = model["model"](**prompt, labels=target_tokens.input_ids)
    logprobs = out.logits.softmax(-1).log()
    target_logprobs = torch.tensor(list(map(operator.getitem, logprobs[0], ids)))
    token_list = model["tokenizer"].batch_decode(ids)
    topk = torch.topk(logprobs[0], n_logprobs)
    top_logprobs = list(
        dict(zip(model["tokenizer"].batch_decode(ids), vals.tolist()))
        for ids, vals in zip(topk.indices, topk.values)
    )
    return dict(
        text=target,
        logprobs=dict(
            token_logprobs=target_logprobs.tolist()[:-1],
            tokens=token_list[:-1],
            top_logprobs=top_logprobs[:-1],
        ),
    )


def generate(model, prompt, n_logprobs):
    out = model["model"].generate(
        inputs=prompt["input_ids"], return_dict_in_generate=True, output_scores=True
    )
    token_list = model["tokenizer"].batch_decode(out.sequences[0])
    logprobs = list(map(operator.getitem, [s[0] for s in out.scores], out.sequences[0]))
    topk = [torch.topk(s[0], n_logprobs) for s in out.scores]
    top_logprobs = list(
        dict(zip(model["tokenizer"].batch_decode(t.indices), map(float, t.values)))
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
