import os

os.environ["GRADIO_TEMP_DIR"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gradio_tmp")
import gradio as gr

from vlm.model import generate
from pipeline import (
    classify, retrieve, gate, judge, title_of, keywords_of,
    build_prompt, strip_hashtags, CAPTION_TOKENS,
)

from upriver.upriver import VERTICALS


def start_generate(image):
    if image is None:
        raise gr.Error("Please give an image")
    return gr.Textbox(interactive=False), gr.Button(visible=False), gr.HTML(visible=True)

def generate_caption(image, story, vertical, voice, progress=gr.Progress()):
    progress(0, desc="Reading image")
    query = classify(image)

    progress(0, desc=f"Checking trends for \u201c{query}\u201d")
    candidates = gate(retrieve(query, vertical=vertical or None))

    progress(0, desc="Checking trend relevance")
    topics = judge(image, candidates)

    if topics:
        note = "**Trends used:** " + ", ".join(title_of(t) for t in topics)
        keywords = [k for t in topics for k in keywords_of(t)]
        if keywords:
            note += "  \n**Keywords offered:** " + ", ".join(keywords)

    elif candidates:
        note = f"{len(candidates)} trends found for \u201c{query}\u201d, none fit the photo \u2014 plain caption."
    else:
        note = f"No trend fit \u201c{query}\u201d \u2014 wrote a plain caption."


    caption, _ = generate(
        image,
        build_prompt(story, topics, voice),
        max_new_tokens=CAPTION_TOKENS,
        temperature=0.8,

        on_token=lambda n: progress((n, CAPTION_TOKENS), desc="writing caption", unit="tokens"),
    )
    return strip_hashtags(caption), note, gr.Column(visible=False), gr.Column(visible=True)



def finish_generate():
    return gr.Textbox(interactive=True), gr.Button(visible=True), gr.HTML(visible=False)

def go_back():
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

    go.click(
        start_generate,
        inputs=image_in,
        outputs=[story_in, go, progress_box],
        show_progress="hidden",
    ).success(
        generate_caption,
        inputs=[image_in, story_in, vertical_in, voice_in],
        outputs=[caption_out, trend_note, upload_page, edit_page],
        show_progress_on=progress_box,
    ).then(
        finish_generate,
        outputs=[story_in, go, progress_box],
        show_progress="hidden",
    )
    back.click(go_back, outputs=[upload_page, edit_page])

if __name__ == "__main__":
    demo.queue().launch()