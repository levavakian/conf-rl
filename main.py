from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
from trl import GRPOConfig, GRPOTrainer
import os

from helpers import (
    extract_xml_answer,
    extract_xml_confidence,
    confidence_format_reward_func,
    answer_format_reward_func,
    correctness_reward_func,
    load_and_split_markets,
    create_markets_dataset,
)

train_markets, test_markets = load_and_split_markets("all_markets_2024-08-01_2025-05-24.json", seed=42)
PREDICTION_SYSTEM_PROMPT = """
You are an expert at analyzing prediction markets. Given a market question, provide your reasoning and prediction as well as a percentage indicating how likely you are to be correct.
Your confidence percentage should be a number between 0.5 and 0.999...
Your confidence percentage will be converted to log odds, and your reward will be based on the log odds.

If you predict very confidently and are correct, you will be given a large reward, if you are wrong, it will be a large penalty.
If you predict very unconfidently and are correct, you will be given a small reward, if you are wrong, it will be a small penalty.

Notice that the confidence percentage is a number between 0.5 and 0.999..., as any confidence percentage below 0.5 would indicate you should switch your answer to the other option.
Your confidence should use as many significant figures as appropriate, you do not have to limit to a fixed number of decimal places.
You can also never be 100% confident, even about an event that has already happened, so your confidence percentage should be less than 1.0.

Your answer should be either YES or NO in <answer>...</answer> and your confidence should be in <confidence>...</confidence>

For example, a valid response would be:
<answer>
YES
</answer>
<confidence>
0.999995
</confidence>
"""
train_dataset = create_markets_dataset(train_markets, PREDICTION_SYSTEM_PROMPT)
test_dataset = create_markets_dataset(test_markets, PREDICTION_SYSTEM_PROMPT)

max_seq_length = 4096 # Increased to handle longer sequences
lora_rank = 128 # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen3-14B",
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = False, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.6, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)

training_args = GRPOConfig(
    use_vllm = False,
    learning_rate = 5e-6,  # Reduced from 1e-5 to be more conservative
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "paged_adamw_8bit",
    logging_steps = 1,
    bf16 = is_bfloat16_supported(),
    fp16 = not is_bfloat16_supported(),
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1,
    num_generations = 6,
    max_completion_length = 2048,
    num_train_epochs = 1,
    save_steps = 25,  # Save checkpoint every 25 steps
    save_total_limit = 3,  # Keep only the 3 most recent checkpoints
    save_strategy = "steps",  # Save based on steps
    load_best_model_at_end = False,
    max_grad_norm = 0.5,
    report_to = "tensorboard",
    log_completions = True,
    output_dir = "outputs",
)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        confidence_format_reward_func,
        answer_format_reward_func,
        correctness_reward_func,
    ],
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = test_dataset,
)

# Check if there's an existing checkpoint to resume from
checkpoint_dir = None
if os.path.exists(training_args.output_dir):
    checkpoints = [d for d in os.listdir(training_args.output_dir) if d.startswith("checkpoint-")]
    if checkpoints:
        # Sort checkpoints by step number and get the latest
        checkpoints.sort(key=lambda x: int(x.split("-")[1]))
        checkpoint_dir = os.path.join(training_args.output_dir, checkpoints[-1])
        print(f"Found checkpoint: {checkpoint_dir}")

# Train with checkpoint resuming
if checkpoint_dir and os.path.exists(checkpoint_dir):
    print(f"Resuming training from checkpoint: {checkpoint_dir}")
    trainer.train(resume_from_checkpoint=checkpoint_dir)
else:
    print("Starting training from scratch")
    trainer.train()

# Save the final model
trainer.save_model("outputs/final_model")
print("Training completed! Final model saved to outputs/final_model")