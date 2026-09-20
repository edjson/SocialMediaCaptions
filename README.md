## Instagram Caption Bridge

This project uses [Upriver.ai's](https://upriver.ai/) api to creat captions for instagram
posts. [Upriver.ai's](https://upriver.ai/) is a monitioring layer for social media watching
current trends. This projects takes that information along side three other inputs. The
first is the post(a single image) itself, the secound is the context behind the post, and 
third a sample of your voice. It works to ground captions in what's in the photo, what
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
