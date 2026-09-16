import os

os.environ["GRADIO_TEMP_DIR"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gradio_tmp")
import gradio as gr

from vlm.model import generate, MAX_NEW_TOKENS

def start_generate(image):
    if image is None:
        raise gr.Error("Please give an image")
    return gr.Textbox(interactive=False), gr.Button(visible=False), gr.HTML(visible=True)

def generate_caption(image, story, progress=gr.Progress()):
    progress(0, desc="Reading image")
    caption = generate(
        image,
        f"write an Instagram caption for this photo. Context: {story}",
        on_token=lambda n: progress((n, MAX_NEW_TOKENS), desc="Writing caption", unit="tokens"),
    )
    return caption, gr.Column(visible=False), gr.Column(visible=True)

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
            go = gr.Button("Generate", variant="primary")
            progress_box = gr.HTML(visible=False, min_height=60)

        with gr.Column(visible=False) as edit_page:
            caption_out = gr.Textbox(label="Caption", lines=8, interactive=True)
            back = gr.Button("Back")
    go.click(
        start_generate,
        inputs=image_in,
        outputs=[story_in, go, progress_box],
        show_progress="hidden",
    ).success(
        generate_caption,
        inputs=[image_in, story_in],
        outputs=[caption_out, upload_page, edit_page],
        show_progress_on=progress_box,
    ).then(
        finish_generate,
        outputs=[story_in, go, progress_box],
        show_progress="hidden",
    )
    back.click(go_back, outputs=[upload_page, edit_page])

if __name__ == "__main__":
    demo.launch()