import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
MAX_NEW_TOKENS = 1024

_processor = None
_model = None


def _load():
    global _processor, _model
    if _model is None:
        _processor = AutoProcessor.from_pretrained(MODEL_ID)
        _model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16, device_map="cuda"
        )
    return _processor, _model


def generate(image, prompt, max_new_tokens=128):
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

    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    n_prompt = inputs.input_ids.shape[1]
    reply = processor.decode(out[0][n_prompt:], skip_special_tokens=True).strip()
    return reply, n_prompt