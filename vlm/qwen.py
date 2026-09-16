import sys
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
MAX_NEW_TOKENS = 128

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16, device_map="cuda"
)

class TokenCounter:
    def __init__(self, on_token):
        self.on_token = on_token
        self.count = -1  # generate() passes the prompt in first; skip it

    def put(self, value):
        self.count += 1
        if self.count > 0:
            self.on_token(self.count)

    def end(self):
        pass

def run_vlm(image, prompt, on_token=None):
    image = image.convert("RGB")
    print("call 0")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")

    with torch.inference_mode():
        print("call 1")
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            streamer=TokenCounter(on_token) if on_token else None,
        )

    trimmed = out[0][inputs.input_ids.shape[1]:]
    return processor.decode(trimmed, skip_special_tokens=True).strip()

if __name__ == "__main__":
    print("call 1")
    image = Image.open(sys.argv[1])
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Describe this image."
    print(run_vlm(image, prompt))
    print(f"\npeak VRAM: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")