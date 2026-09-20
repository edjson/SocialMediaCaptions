# Instagram Caption Bridge

This project uses [Upriver.ai's](https://upriver.ai/) API to create captions for Instagram
posts. Upriver.ai is a monitoring layer for social media that watches current trends. This
project takes that information alongside three other inputs. The first is the post (a single
image) itself, the second is the context behind the post, and the third is a sample of your
voice via text. It works to ground captions in what's in the photo, what you say about it,
and what's actually trending. In the case that the context behind the post is not applicable
to a current trend, the Caption Bridge will generate a plain one instead. Forcing an
unrelated trend onto a photo is the main failure mode it's built to avoid.

## Setup

Needs an NVIDIA GPU with ~8 GB free. `vlm/model.py` hardcodes `device_map="cuda"`,
so CPU-only torch fails at load.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1

# CUDA 12.8 wheel. The default CPU wheel installs fine, then fails at model load.
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

Set-Content .env -Encoding utf8 -Value 'UPRIVER_API_KEY=your_key_here'

python main.py   # Gradio UI on http://127.0.0.1:7860
```

## Model weights

~8 GB, [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct). Transformers
downloads them on first run, so this step is only to pre-fetch and avoid the wait at launch:

```
hf download Qwen/Qwen3-VL-4B-Instruct
```

`hf` ships with `huggingface_hub`, which transformers already pulls in.

## How it works

One 4B vision model does three passes: describe, judge, write.

```
photo ──▶ describe ──┬──▶ retrieve ──▶ gate ──▶ judge ──▶ write ──▶ clean ──▶ caption
                     │     (2 vector    (score)   (VLM)    (VLM)   (regex)
story ───────────────┘      queries)
voice ─────────────────────────────────────────────────────▶
```

1. **Describe:** a sentence or two about the photo: setting, subject, kind of post.
2. **Retrieve:** two queries go to Upriver, the description and your story
   together, and your story on its own. Results merge by topic id, keeping the
   higher score. The Vertical dropdown pins both queries to tech, sports,
   politics or creative; Auto leaves it unset.
3. **Gate:** drops anything under `min_score=0.10`, keeps the top 5. A junk
   filter, not a relevance test.
4. **Judge:** the model sees the photo and each candidate with its citation
   snippet, and answers with numbers or `NONE`. Relevance is decided here.
5. **Write:** voice first, then your story, then surviving trends as optional
   colour. Sampled at temperature 0.8.
6. **Clean:** strips hashtags, emoji and "Here's a caption:" preambles, turns em
   dashes into commas, and unwraps quoted output. If the caption just echoes your
   story it retries once at temperature 1.0, not 0.8, since an echo is mode
   collapse and retrying at the same setting reproduces it.

## Why it's built this way

**The judge exists because Upriver does no relevance filtering.** A nonsense query
returns five confident topics. A latte photo matched *"DOJ Probes NYC Coffee Shop
for Banning Pro-Israel Lawmaker"* the real score, keyed on "coffee shop".

**The gate is 0.10, not 0.30, because score stopped meaning relevance.** Short
queries separated cleanly at 0.30. Longer ones inflated everything: a coffee
photo's best *noise* hit went 0.108 -> 0.523, indistinguishable from a correct
match at 0.557.

**The story is searched on its own, and not only glued to the description,
because the description drowned it.** "lets build a data center" returned
*"Midjourney vs Krea AI"* when appended to a 60-word description; searched alone
it returns four on-topic data-center trends.

**There's no vision-encode cache**, though the original design called for one. The
encode measured ~70 ms of a ~7 s run, the token generation dominates.

## Instrumentation

Every run appends a JSON line to `runs/runs.jsonl`: both queries, every candidate
with scores, gate and judge outcomes, the full prompt, timings, and the thresholds
in force.

```
python json_to_table.py           # one row per run
python json_to_table.py -n 5      # last 5 runs
python json_to_table.py -d last   # full trace of the latest run
python test_upriver.py            # API diagnostic
```

## Limitations

- **Coverage.** Upriver serves tech, sports, politics and an undocumented
  `creative` vertical. Food, travel, pets and fashion return nothing usable, so for
  most photos the trend layer correctly does nothing.
- **Single image.** No carousel support.
