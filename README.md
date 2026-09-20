## Instagram Caption Bridge

This project uses [Upriver.ai's](https://upriver.ai/) api to creat captions for instagram
posts. [Upriver.ai's](https://upriver.ai/) is a monitioring layer for social media watching
current trends. This projects takes that information along side three other inputs. The
first is the post(a single image) itself, the secound is the context behind the post, and 
third a sample of your voice via text. It works to ground captions in what's in the photo, what
you say about it, and what's actually trending. In the case that the context behind the post
is not applicable to a current trend, the Caption Bridge will generate a plain one instead.
Additionally forcing an unrelated trend onto a photo is the main failure mode it's built to 
avoid.

# Setup

Needs an NVIDIA GPU with ~8 GB free. `vlm/model.py` hardcodes `device_map="cuda"`,
so CPU-only torch fails at load.

```
python -m venv venv && venv\Scripts\activate  #virtual env
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128 #torch 5060 ti
pip install -r requirements.txt #installs packages

echo UPRIVER_API_KEY=your_key_here > .env

python main.py
```

# Model weights (~8 GB, [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct))
hf download Qwen/Qwen3-VL-4B-Instruct    

## How it works

One 4B vision model does three passes: describe, judge, write.

```
photo ──▶ describe ──┬──▶ retrieve ──▶ gate ──▶ judge ──▶ write ──▶ clean ──▶ caption
                     │     (2 vector    (score)   (VLM)    (VLM)   (regex)
story ───────────────┘      queries)
voice ─────────────────────────────────────────────────────▶
```

1. **Describe:** a sentence or two about the photo: setting, subject, kind of post.
2. **Retrieve:** the description and your story go to Upriver as two separate
   queries, merged by topic id.
3. **Gate:** drops anything under `min_score=0.10`, keeps the top 5. A junk
   filter, not a relevance test.
4. **Judge:** the model sees the photo and each candidate with its citation
   snippet, and answers with numbers or `NONE`. Relevance is decided here.
5. **Write:** voice first, then your story, then surviving trends as optional
   colour. Sampled at temperature 0.8.
6. **Clean:** strips hashtags, emoji, em dashes and "Here's a caption:"
   preambles. Retries once if the caption just echoes your story.

## Why it's built this way

**The judge exists because Upriver does no relevance filtering.** A nonsense query
returns five confident topics. A latte photo matched *"DOJ Probes NYC Coffee Shop
for Banning Pro-Israel Lawmaker"* the real score, keyed on "coffee shop".

**The gate is 0.10, not 0.30, because score stopped meaning relevance.** Short
queries separated cleanly at 0.30. Longer ones inflated everything: a coffee
photo's best *noise* hit went 0.108 -> 0.523, indistinguishable from a correct
match at 0.557.

**The description and story are searched separately because one drowned the
other.** "lets build a data center" returned *"Midjourney vs Krea AI"* when glued
to a 60-word description; searched alone it returns four on-topic data-center
trends.

**There's no vision-encode cache**, though the original design called for one. The
encode measured ~70 ms of a ~7 s run, the token generation dominates.

## Instrumentation

Every run appends a JSON line to `runs/runs.jsonl`: both queries, every candidate
with scores, gate and judge outcomes, the full prompt, timings, and the thresholds
in force.

```
python json_to_table.py           # one row per run
python json_to_table.py -d last   # full trace of the latest run
python test_upriver.py            # API diagnostic
```

## Limitations

- **Coverage.** Upriver serves tech, sports, politics and an undocumented
  `creative` vertical. Food, travel, pets and fashion return nothing usable, so for
  most photos the trend layer correctly does nothing.
- **Single image.** No carousel support.
