import os

import torch
from compressed_tensors.offload import dispatch_model
from datasets import load_dataset
from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration, AutoModelForVision2Seq

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier



MODEL_ID = "/workspace/models/huihui-ai/Huihui-Qwen3-VL-30B-A3B-Instruct-abliterated"
NUM_CALIBRATION_SAMPLES = 1
MAX_SEQUENCE_LENGTH = 8192
LOCAL_DATA_PATH = "/workspace/data/merged_dataset.jsonl"

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.set_grad_enabled(False)

max_memory_mapping = {
    0: "50GiB",     
    "cpu": "200GiB"  
}
model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16, 
    device_map="cpu", 
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)


ds = load_dataset("json", data_files={"train": LOCAL_DATA_PATH}, split="train")

if len(ds) > NUM_CALIBRATION_SAMPLES:
    ds = ds.select(range(NUM_CALIBRATION_SAMPLES))

ds = ds.shuffle(seed=42)


def preprocess_function(example):
    messages = []
    for message in example["messages"]:
        messages.append(
            {
                "role": message["role"],
                "content": [{"type": "text", "text": message["content"]}],
            }
        )

    return processor.apply_chat_template(
        messages,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
        tokenize=True,
        add_special_tokens=False,
        return_dict=True,
        add_generation_prompt=False,
    )


ds = ds.map(preprocess_function, batched=False, remove_columns=ds.column_names)


def data_collator(batch):
    assert len(batch) == 1
    return {
        key: (
            torch.tensor(value)
            if key != "pixel_values"
            else torch.tensor(value, dtype=torch.bfloat16).squeeze(0)
        )
        for key, value in batch[0].items()
    }


# Configure AWQ quantization with smoothing and balancing
# NOTE: This recipe uses W4A16 quantization with group_size=32
# rather than the default preset with group_size=128
recipe = AWQModifier(
    ignore=[
        "re:.*embed_tokens",
        "re:.*input_layernorm$",
        "re:.*mlp[.]gate$",
        "re:.*post_attention_layernorm$",
        "re:.*norm$",
        "re:model[.]visual.*",
        "re:visual.*",
        "lm_head",
    ],
    duo_scaling=True,
    config_groups={
        "group_0": {
            "targets": ["Linear"],
            "weights": {
                "num_bits": 4,
                "type": "int",
                "symmetric": True,
                "group_size": 32,
                "strategy": "group",
                "dynamic": False,
                "actorder": None,
                "observer": "mse",
            },
        }
    },
)

torch.cuda.empty_cache()

# Apply AWQ quantization.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-AWQ-W4A16-mse-seq-back"
oneshot(
    model=model,
    processor=processor,
    recipe=recipe,
    dataset=ds.select(range(1)),
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    data_collator=data_collator,
    output_dir=SAVE_DIR,
)

print("========== SAMPLE GENERATION ==============")
dispatch_model(model)
input_ids = processor(text="Hello my name is", return_tensors="pt").input_ids.to("cuda")
output = model.generate(input_ids, max_new_tokens=20)
print(processor.decode(output[0]))
print("==========================================")

# Save to disk in compressed-tensors format.

model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)
