# %%capture
# Installs Unsloth, Xformers (Flash Attention) and all other packages!
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# We have to check which Torch version for Xformers (2.3 -> 0.0.27)
from torch import __version__; from packaging.version import Version as V
xformers = "xformers==0.0.27" if V(__version__) < V("2.4.0") else "xformers"
# !pip install --no-deps {xformers} trl peft accelerate bitsandbytes triton

from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    "unsloth/llama-2-7b-bnb-4bit",
    "unsloth/llama-2-13b-bnb-4bit",
    "unsloth/codellama-34b-bnb-4bit",
    "unsloth/tinyllama-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit", # New Google 6 trillion tokens model 2.5x faster!
    "unsloth/gemma-2b-bnb-4bit",
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/mistral-7b-v0.3", # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

from constants import entity_unit_map

# Updated custom prompt
custom_prompt = """Extract the {entity} from the following product text. 
Your answer should be in the format "x unit" where x is a float number and unit is one of the allowed units for this entity.

Allowed units for {entity}: {allowed_units}

If no value is found, return an empty string.

Product Text: {text}
Group ID: {group_id}

{entity}: """

def formatting_prompts_func(examples):
    texts = examples["text"]
    group_ids = examples["group_id"]
    entities = examples["entity"]
    values = examples["value"]  # Assuming this is the correct value for the entity
    formatted_texts = []
    for text, group_id, entity, value in zip(texts, group_ids, entities, values):
        allowed_units = ", ".join(entity_unit_map[entity])
        formatted_text = custom_prompt.format(
            entity=entity,
            allowed_units=allowed_units,
            text=text,
            group_id=group_id
        )
        formatted_text += f"{value}{tokenizer.eos_token}"
        formatted_texts.append(formatted_text)
    return {"text": formatted_texts}

from datasets import load_dataset
dataset = load_dataset("path/to/your/dataset", split="train")  # Replace with your dataset path
dataset = dataset.map(formatting_prompts_func, batched=True)

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60, # Set num_train_epochs = 1 for full training runs
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

trainer_stats = trainer.train()

# alpaca_prompt = Copied from above
# FastLanguageModel.for_inference(model)
# sample_text = "ModelBL-27B1 Li-Polymerbattery 1ICP4/60/75 3.8V 2700mAh 10.26Wh Limited charge voltage4.35V Ningbo Veken Sattery Co.,LTD S/N:206090180172465452"
# sample_group_id = "123"
# sample_entity = "voltage"

# inputs = tokenizer(
# [
#     custom_prompt.format(
#         entity=sample_entity,
#         text=sample_text,
#         group_id=sample_group_id
#     )
# ], return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)
print(tokenizer.batch_decode(outputs)[0])

model.save_pretrained("lora_model") # Local saving
tokenizer.save_pretrained("lora_model")
# model.push_to_hub("your_name/lora_model", token = "...") # Online saving
# tokenizer.push_to_hub("your_name/lora_model", token = "...") # Online saving