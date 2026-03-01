from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import AWQModifier
from transformers import AutoTokenizer, AutoProcessor
from datasets import load_dataset

model_path = "/workspace/models/Huihui-Qwen3-VL-32B-Thinking-abliterated"
quant_path = "/workspace/models/Huihui-Qwen3-VL-32B-Thinking-abliterated-AWQ"

tokenizer = AutoTokenizer.from_pretrained(model_path)
processor = AutoProcessor.from_pretrained(model_path)

calib_ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train[:64]')  # 降到 64
calib_texts = [item['text'] for item in calib_ds if item['text'].strip()]

print(f"校准文本数量: {len(calib_texts)}")

recipe = [
    AWQModifier(
        ignore=["lm_head"],
        scheme="W4A16_ASYM",
        targets=["Linear"]
    )
]

oneshot(
    model=model_path,
    dataset=calib_texts,
    recipe=recipe,
    output_dir=quant_path,
    num_calibration_samples=len(calib_texts),
    max_seq_length=1024,              # 关键：降到 1024
    trust_remote_code=True,
)

tokenizer.save_pretrained(quant_path)
processor.save_pretrained(quant_path)

print("量化完成！模型保存在:", quant_path)