# Caption Bridge

An Instagram captioning tool that grounds captions in what's actually
in your photo and what's trending right now.

Most caption generators look at the image alone and produce generic
description. This one adds two things: your account's own posting
history as voice context, and live trend data from the Upriver API.
A fine-tuned VLM maps the photo into Upriver's category taxonomy so
the trend lookup is targeted instead of generic.

If no current trend fits the photo, the tool says so and writes a
plain caption. Forcing an unrelated trend onto an image is the main
failure mode this is built to avoid.

## Pipeline

1. **Classify** — the post photo goes to a fine-tuned Qwen3-VL-4B,
   which emits Upriver category IDs plus a short free-text query.
2. **Retrieve** — the query and the account's vertical go to Upriver's
   breakout topics endpoint. Results come back with momentum scores.
3. **Gate** — if nothing comes back, or everything is low-momentum,
   the trend layer is skipped.
4. **Write** — Qwen3-VL-8B-Instruct writes the caption from the photo,
   the account context, and any surviving trends.

Two models by design. The small tuned one classifies into a fixed
vocabulary. The large base one writes. Neither does the other's job.

Carousel posts are handled as multi-image input to a single caption.
Vision tokens are the dominant cost there, so images are downscaled
against a configurable budget.

## Training

The classifier is distilled, not trained from scratch.

- Pull the category taxonomy from Upriver's `/media_categories`.
- Label a few thousand COCO images with a frontier model against
  that taxonomy. These are the teacher labels.
- LoRA fine-tune Qwen3-VL-4B on the pairs using NeMo Automodel
  (`FinetuneRecipeForVLM`), reusing the multimodal collate path
  from the shipped VLM recipes.
- Serve with vLLM.

Why fine-tune instead of prompting the category list: the list costs
prompt tokens on every image, forever. Baking it into weights removes
it. That saving is one of the two numbers this project reports.

Constrained decoding is used regardless of the fine-tune — output is
grammar-restricted to valid category IDs, so malformed output is not
a failure mode the training has to solve.

## Evaluation

Two numbers, both against a prompted baseline with no fine-tune:

1. **Teacher agreement** — how often the distilled 4B picks the same
   category as the frontier teacher, on held-out images.
2. **Prompt-token delta** — tokens saved per request by dropping the
   category list from the prompt.

Plus one ablation for the retrieval layer: same photo, same account,
captions generated with and without trend injection. If the outputs
are indistinguishable, the trend data isn't earning its API call.

Held-out eval uses real creator images, not COCO. See limitations.

## Known limitations

- **Domain gap.** COCO is everyday object photography. Real creator
  content is thumbnails, screenshots, product shots, and text overlays.
  The classifier is weakest on exactly the images it sees at runtime.
  The held-out creator eval set exists to measure this gap, not hide it.

- **Small accounts get no enrichment.** Upriver skips bio, audience,
  and brand safety below a follower threshold. Account context for a
  personal page therefore comes from the user's own post history, not
  from Upriver's creator endpoint. Upriver supplies trends only.

- **Noisy supervision if using creator labels.** Upriver's labels are
  creator-level, not image-level. They are not used as image ground
  truth here for that reason.

- **Serial latency.** Classify, retrieve, then write. The vision pass
  is cached per image so re-running against fresh trends does not
  re-encode the photo.

## Status

- [ ] Prompted baseline, end to end, no training
- [ ] Held-out creator eval set
- [ ] COCO labeling run
- [ ] LoRA fine-tune on Pinnacles
- [ ] Benchmark + ablation

## Stack

Qwen3-VL-8B-Instruct, Qwen3-VL-4B-Instruct, NVIDIA NeMo Automodel,
vLLM, Upriver API.

