import re 
from upriver.upriver import search_topics, UpriverError
from vlm.model import generate
JUDGE_TOKENS = 16

JUDGE_PROMPT = (
    "Candidate trending topics:\n{listed}\n\n"
    "Which of these topics, if any, genuinely relate to what is shown in this photo? "
    "Be strict: a shared word is not a real connection. "
    "Reply with matching numbers only, comma separated, or NONE."
)


def judge(image, topics):
    """Drop topics that pass the gate but don't match the photo."""
    if not topics:
        return []
    listed = "\n".join(f"{i}. {title_of(t)}" for i, t in enumerate(topics, 1))
    verdict, _ = generate(
        image, JUDGE_PROMPT.format(listed=listed), max_new_tokens=JUDGE_TOKENS
    )
    if "NONE" in verdict.upper():
        return []
    picked = {int(n) for n in re.findall(r"\d+", verdict) if 1 <= int(n) <= len(topics)}
    return [t for i, t in enumerate(topics, 1) if i in picked]

QUERY_TOKENS = 32
CAPTION_TOKENS = 220

CLASSIFY_PROMPT = (
    "Describe what this photo is about in 3 to 6 words, as a search query "
    "for finding trending topics. Output only the query."
)


def classify(image):
    """Photo -> short free-text query for Upriver."""
    query, _ = generate(image, CLASSIFY_PROMPT, max_new_tokens=QUERY_TOKENS)
    return query.strip().strip('"').splitlines()[0]


def title_of(topic):
    name = topic.get("topic_name") or topic.get("canonical_name")
    return name.strip() if isinstance(name, str) and name.strip() else None


def momentum_of(topic):
    value = (topic.get("trend") or {}).get("momentum")
    return float(value) if isinstance(value, (int, float)) else 0.0


def score_of(topic):
    value = topic.get("score")
    return float(value) if isinstance(value, (int, float)) else 0.0


def retrieve(query, vertical=None, limit=8):
    """Never raises. An API failure degrades to no trends."""
    try:
        return search_topics(query, vertical=vertical, limit=limit)
    except (UpriverError, OSError):
        return []


def gate(topics, min_score=0.30, keep=3):
    """Score >= 0.30 separates real matches from fuzzy keyword noise.

    Measured over 61 topics / 8 queries: every query whose best topic cleared
    0.30 had a genuinely related topic; every query at or below 0.15 returned
    unrelated news. Momentum and status are NOT filtered on - momentum is
    missing on ~60% of topics and 87% of the corpus is status="declining".
    """
    kept = [t for t in topics if title_of(t) and score_of(t) >= min_score]
    kept.sort(key=score_of, reverse=True)
    return kept[:keep]



CAPTION_PROMPT = (
    "Write one Instagram caption for this photo.\n"
    "- One or two sentences, under 25 words.\n"
    "- Dry and specific. Say something only this photo could prompt.\n"
    "- No hashtags, no emoji, no exclamation marks.\n"
    "- No rhetorical questions, and do not address the reader.\n"
    "- Do not describe or explain what is in the photo.\n"
    "- Never use: vibes, core, obsessed, living for, POV, dive into, chef's kiss.\n"
    "Output the caption text only."
)


def build_prompt(story, topics):
    parts = [CAPTION_PROMPT]
    if story:
        parts.append(f"The person posting this says: {story}")
    if topics:
        listed = "\n".join(f"- {title_of(t)}" for t in topics)
        parts.append(
            "These subjects are getting attention right now:\n"
            f"{listed}\n"
            "Borrow at most one as an angle, and only if it genuinely connects to "
            "the image. Write it as your own passing thought, never as news or a "
            "headline. If none connect, ignore this list completely."
        )
    return "\n\n".join(parts)

