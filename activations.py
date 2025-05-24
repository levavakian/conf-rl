# ---------- editable constants ----------
LAYER              = 20                     # transformer layer (0‑based)
MODEL_NAME         = "Qwen/Qwen3-14B"
OUTPUT_DIR         = "qwen_acts_idx"      # folder created if needed
DEVICE             = "cuda"
OUTPUT_GENERATION = True

SEQ_LEN            = 8192
MAX_TOKENS         = 40_000_000
TEST = False
if TEST:
    BATCH_SIZE = 1
    TEST_BATCHES = 1
    TRAIN_BATCHES = 1
    MAX_SHARDS = 2
    OUTPUT_GENERATION = True
else:
    BATCH_SIZE = 4
    TEST_BATCHES = 1
    TRAIN_BATCHES = 9
    MAX_SHARDS = 0
    OUTPUT_GENERATION = True
# ---------------------------------------

import itertools, torch, types
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from unsloth import FastLanguageModel
from collections import deque

def find_num_existing_shards():
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    max_shard_idx = -1
    for f_path in output_path.iterdir():
        if f_path.suffix == ".pt":
            parts = f_path.stem.split("_") # e.g. layer20_0_train
            if len(parts) == 3 and parts[0] == f"layer{LAYER}" and parts[2] == "train":
                try:
                    shard_idx = int(parts[1])
                    if shard_idx > max_shard_idx:
                        max_shard_idx = shard_idx
                except ValueError:
                    pass
    return max_shard_idx + 1

def generate_doc_response(tokenizer, model, doc):
    conversation = [
        {"role": "user", "content": doc},
    ]

    formatted_str = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        formatted_str,
        return_tensors="pt",
        padding=False, 
        truncation=True, 
        max_length=SEQ_LEN
    )

    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    current_prompt_seq_len = input_ids.shape[1]
    num_new_tokens = max(0, SEQ_LEN - current_prompt_seq_len)
    
    with torch.no_grad():
        resp_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=num_new_tokens,
            pad_token_id=tokenizer.eos_token_id
        )
    output = tokenizer.decode(resp_ids[0], skip_special_tokens=False)

    if OUTPUT_GENERATION:
        print("\n\n\n", output, "\n\n\n")

    return output


def generate_activations(tokenizer, model, docs):
    """
    For every document returns a dict with:
        text         – the original document string
        tokens       – list[str]   (token-level strings)
        token_ids    – torch.LongTensor (seq_len)
        activations  – torch.FloatTensor (seq_len, d_model)
    """
    formatted_docs = [generate_doc_response(tokenizer, model, doc) for doc in docs]

    batch = tokenizer(
        formatted_docs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=SEQ_LEN,
    )
    input_ids  = batch["input_ids"].to(DEVICE)
    attn_mask  = batch["attention_mask"].to(DEVICE)

    # 2. Capture the chosen activation -----------------------------------
    captured = []
    def _hook(_, __, out):
        # Unsloth returns (hidden_states, *extras)
        if isinstance(out, tuple):
            out = out[0]
        captured.append(out.detach())

    # Hook the layer residual stream so that we can later use the activations to train an interp SAE.
    handle = model.model.layers[LAYER].register_forward_hook(_hook)

    with torch.no_grad():
        # Incur cost of an extra forward pass to get the activations, but keep the code simpler.
        # The single forward pass with the total context should still generate all of the activations we would
        # have computed in the model.generate inside of generate_doc_response regardless.
        _ = model(input_ids=input_ids, attention_mask=attn_mask)

    handle.remove()

    # 3. Build per-document records --------------------------------------
    acts_batch      = captured[0].bfloat16().cpu()
    input_ids_cpu   = input_ids.cpu()
    attn_mask_cpu   = attn_mask.cpu()

    records = []
    for i, doc in enumerate(docs):
        seq_len   = int(attn_mask_cpu[i].sum())
        ids       = input_ids_cpu[i, :seq_len]
        acts      = acts_batch[i, :seq_len]

        tokens = tokenizer.convert_ids_to_tokens(ids.tolist(), skip_special_tokens=False)

        records.append(
            {
                "tokens": tokens,
                "token_ids": ids,
                "activations": acts,
            }
        )
    return records

# 1. Load 4‑bit Unsloth model + tokenizer
print("loading Unsloth model …")
model, tokenizer = FastLanguageModel.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,
    max_seq_length=SEQ_LEN,
    gpu_memory_utilization = 0.85, # Reduce if out of memory
)
model.eval()
D_MODEL = model.config.hidden_size
print("d_model =", D_MODEL)
print("model device =", model.device)

stream = load_dataset("Goodfire/r1-collect", split="train",
                      streaming=True, trust_remote_code=True)
text_iter = (row["question"] for row in stream)

# 3. Find existing shards to resume
next_shard_idx = find_num_existing_shards()
if next_shard_idx > 0:
    print(f"Resuming from shard {next_shard_idx}")
    text_iter = itertools.islice(text_iter, next_shard_idx * (TEST_BATCHES + TRAIN_BATCHES) * BATCH_SIZE, None)
else:
    print("Starting from scratch")

# 4. Main processing loop
try:
    docs_finished = False
    for shard_idx in itertools.count(start=next_shard_idx):
        if (MAX_SHARDS > 0 and shard_idx >= MAX_SHARDS) or docs_finished:
            break

        print(f"Processing shard {shard_idx}...")
        train_res = []
        for _ in range(TRAIN_BATCHES):
            train_docs = list(itertools.islice(text_iter, BATCH_SIZE))
            if not train_docs:
                docs_finished = True
                break
            train_res.extend(generate_activations(tokenizer, model, train_docs))
        
        test_res = []
        for _ in range(TEST_BATCHES):
            test_docs = list(itertools.islice(text_iter, BATCH_SIZE))
            if not test_docs:
                docs_finished = True
                break
            test_res.extend(generate_activations(tokenizer, model, test_docs))
        
        torch.save(train_res, Path(OUTPUT_DIR) / f"layer{LAYER}_{shard_idx}_train.pt")
        torch.save(test_res, Path(OUTPUT_DIR) / f"layer{LAYER}_{shard_idx}_test.pt")
        
except KeyboardInterrupt:
    print("\nInterrupted by user. Exiting.")

print("Finished processing.")





