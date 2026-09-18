import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
MAX_NEW_TOKENS = 1024
MAX_SIDE = 1024


_processor = None
_model = None

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


def _load():
    global _processor, _model
    if _model is None:
        _processor = AutoProcessor.from_pretrained(MODEL_ID)
        _model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16, device_map="cuda"
        )
    return _processor, _model


def generate(image, prompt, max_new_tokens=128, on_token=None, temperature=0.0):
    """image: PIL.Image or path. Returns (reply_text, prompt_token_count)."""
    processor, model = _load()
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image = image.convert("RGB")
    image.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)

    messages = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": prompt}],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

    # Greedy for classify/judge, where the answer should be repeatable.
    # Sampled for captions, where greedy picks the blandest phrasing every time.
    if temperature > 0:
        sampling = {"do_sample": True, "temperature": temperature, "top_p": 0.9}
    else:
        sampling = {"do_sample": False}

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            streamer=TokenCounter(on_token) if on_token else None,
            **sampling,
        )


    n_prompt = inputs.input_ids.shape[1]
    reply = processor.decode(out[0][n_prompt:], skip_special_tokens=True).strip()
    return reply, n_prompt