# from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
from trl import GRPOConfig, GRPOTrainer
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from helpers import (
    confidence_format_reward_func,
    answer_format_reward_func,
    correctness_reward_func,
    load_and_split_markets,
    create_markets_dataset,
)

train_markets, test_markets = load_and_split_markets("all_markets_2024-08-01_2025-05-24.json", seed=55)
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

max_seq_length = 2048 # Can increase for longer reasoning traces
lora_rank = 32 # Larger rank = smarter, but slower

# Configure 8-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16,
)

# Load tokenizer with proper configuration
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3-14B",
    model_max_length=max_seq_length,
    padding_side="left",
    truncation_side="left"
)

# Add padding token if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model in 8-bit
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-14B",
    # quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

# Prepare model for 8-bit training
model = prepare_model_for_kbit_training(model)

# Configure LoRA
peft_config = LoraConfig(
    lora_alpha=lora_rank*2,
    lora_dropout=0.05,
    r=lora_rank,
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    task_type="CAUSAL_LM", 
)

# Apply LoRA to the 8-bit model
model = get_peft_model(model, peft_config)

max_prompt_length = 512
max_completion_length = max_seq_length - max_prompt_length

training_args = GRPOConfig(
    temperature = 1.0,
    learning_rate = 5e-6,
    weight_decay = 0.01,
    warmup_ratio = 0.1,
    lr_scheduler_type = "linear",
    optim = "adamw_8bit",
    logging_steps = 1,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 4, # Increase to 4 for smoother training
    num_generations = 4, # Decrease if out of memory
    generation_batch_size = 1,
    max_prompt_length = max_prompt_length,
    max_completion_length = max_completion_length,
    num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 100,
    save_steps = 10,
    report_to = "tensorboard", # Can use Weights & Biases
    output_dir = "outputs",

    # For optional training + evaluation
    # fp16_full_eval = True,
    # per_device_eval_batch_size = 4,
    # eval_accumulation_steps = 1,
    # eval_strategy = "steps",
    # eval_steps = 1,
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
