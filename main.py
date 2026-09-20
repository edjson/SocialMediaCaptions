import os

os.environ["GRADIO_TEMP_DIR"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gradio_tmp")
import gradio as gr

import time
from inspect import signature

from vlm.model import generate, MODEL_ID
from pipeline import (
    describe, build_query, retrieve, retrieve_many, gate, judge, title_of, keywords_of,
    score_of, build_prompt, clean_caption, is_echo, debug_note,
    ECHO_RETRY, CAPTION_TOKENS,
)
import runlog


from upriver.upriver import VERTICALS

# Read off gate() itself, so a logged run stays interpretable after the
# thresholds change.
GATE_SETTINGS = {k: v.default for k, v in signature(gate).parameters.items()
                 if v.default is not v.empty}


def topic_row(topic):
    """Compact enough to log every candidate, complete enough to re-run the
    gate offline at any threshold."""
    return {
        "id": topic.get("topic_id"),
        "score": round(score_of(topic), 4),
        "title": title_of(topic),
        "vertical": topic.get("vertical"),
        "status": topic.get("status"),
        "cites": len(topic.get("citations") or []),
    }


def start_generate(image):
    if image is None:
        raise gr.Error("Please give an image")
    return gr.Textbox(interactive=False), gr.Button(visible=False), gr.HTML(visible=True)

def generate_caption(image, story, vertical, voice, progress=gr.Progress()):
    run_id = runlog.new_run_id()

    progress(0, desc="Reading image")
    started = time.perf_counter()
    description = describe(image)
    t_describe = time.perf_counter() - started
    query = build_query(description, story)

    progress(0, desc="Checking trends")
    started = time.perf_counter()
    queries = [q for q in (query, story.strip()) if q]
    raw = retrieve_many(queries, vertical=vertical or None)
    candidates = gate(raw)
    t_retrieve = time.perf_counter() - started

    progress(0, desc="Checking trend relevance")
    started = time.perf_counter()
    topics, verdict = judge(image, candidates)
    t_judge = time.perf_counter() - started

    keywords = list(dict.fromkeys(k for topic in topics for k in keywords_of(topic)))
    if topics:
        note = "**Trends used:** " + ", ".join(title_of(t) for t in topics)
        if keywords:
            note += "  \n**Keywords offered:** " + ", ".join(keywords)
    elif candidates:
        note = f"{len(candidates)} trends found, none fit the photo \u2014 plain caption."
    else:
        note = f"No trend fit \u201c{query[:60]}\u2026\u201d \u2014 wrote a plain caption."
    note += "\n\n" + debug_note(query, raw, candidates, topics, verdict)

    prompt = build_prompt(story, topics, voice)
    started = time.perf_counter()
    caption, _ = generate(
        image,
        prompt,
        max_new_tokens=CAPTION_TOKENS,
        temperature=0.8,
        on_token=lambda n: progress((n, CAPTION_TOKENS), desc="writing caption", unit="tokens"),
    )
    t_caption = time.perf_counter() - started
    cleaned = clean_caption(caption)

    # A prompt rule the model can ignore needs a guarantee in code, the same
    # reason clean_caption exists. Temperature goes up, not back to 0.8: the
    # echo is mode collapse, so retrying at the same setting repeats it.
    regenerated = False
    caption_first = None
    if is_echo(cleaned, story):
        regenerated = True
        caption_first = cleaned
        progress(0, desc="caption echoed the story, retrying")
        caption, _ = generate(
            image, prompt + ECHO_RETRY, max_new_tokens=CAPTION_TOKENS,
            temperature=1.0,
        )
        cleaned = clean_caption(caption)


    runlog.log(
        "generate",
        run_id=run_id,
        image=runlog.image_key(image),
        image_size=list(image.size),
        story=story,
        voice=voice,
        vertical=vertical or None,
        description=description,
        query=query,
        queries=queries,
        raw=[topic_row(t) for t in raw],
        gate_ids=[t.get("topic_id") for t in candidates],
        judge_verdict=verdict,
        kept_ids=[t.get("topic_id") for t in topics],
        keywords=keywords,
        prompt=prompt,
        caption_raw=caption,
        caption=cleaned,
        regenerated=regenerated,
        caption_first=caption_first,
        settings={"model": MODEL_ID, "temperature": 0.8,
                  "caption_tokens": CAPTION_TOKENS, **GATE_SETTINGS},
        timings={"describe": round(t_describe, 2), "retrieve": round(t_retrieve, 2),
                 "judge": round(t_judge, 2), "caption": round(t_caption, 2)},
    )

    return cleaned, note, run_id, gr.Column(visible=False), gr.Column(visible=True)


def finish_generate():
    return gr.Textbox(interactive=True), gr.Button(visible=True), gr.HTML(visible=False)

def go_back(caption, run_id):
    """Leaving the edit page is the accept signal; caption_out is editable, so
    this captures whatever the user changed it to."""
    runlog.log("accepted", run_id=run_id, caption_final=caption)
    return gr.Column(visible=True), gr.Column(visible=False)

with gr.Blocks(title="Caption Bridge") as demo:
    gr.Markdown("# Caption Bridge")

    with gr.Row():
        with gr.Column() as upload_page:
            image_in = gr.Image(type="pil", sources=["upload"], label="Post image", height=400)
            story_in = gr.Textbox(
                label="Context",
                placeholder="What's the story behind this post?",
                lines=3,
            )
            voice_in = gr.Textbox(
                label="Your voice",
                placeholder="Paste a few captions you've written before, one per line",
                lines=4,
            )

            vertical_in = gr.Dropdown(
                choices=[("Auto", "")] + [(v.title(), v) for v in VERTICALS],
                value="",
                label="Vertical",
            )
            go = gr.Button("Generate", variant="primary")
            progress_box = gr.HTML(visible=False, min_height=60)

        with gr.Column(visible=False) as edit_page:
            caption_out = gr.Textbox(label="Caption", lines=8, interactive=True)
            trend_note = gr.Markdown()
            back = gr.Button("Back")
            run_state = gr.State("")

    go.click(
        start_generate,
        inputs=image_in,
        outputs=[story_in, go, progress_box],
        show_progress="hidden",
    ).success(
        generate_caption,
        inputs=[image_in, story_in, vertical_in, voice_in],
        outputs=[caption_out, trend_note, run_state, upload_page, edit_page],
        show_progress_on=progress_box,
    ).then(
        finish_generate,
        outputs=[story_in, go, progress_box],
        show_progress="hidden",
    )
    back.click(go_back, inputs=[caption_out, run_state], outputs=[upload_page, edit_page])

if __name__ == "__main__":
    demo.queue().launch()