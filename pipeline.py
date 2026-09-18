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


KEYWORD_CONFIDENCE = 0.8


def keywords_of(topic, limit=4):
    """Searchable names Upriver resolved for one topic, in Upriver's order.

    Drops low-confidence entities (Sony's "35mm GM" lens resolved to General
    Motors at 0.655) and Wikipedia-style disambiguations like
    "The Tech (newspaper)".
    """
    seen, keywords = set(), []
    for entity in topic.get("entities") or []:
        name = (entity.get("canonical_name") or "").strip()
        if (not name or "(" in name or name.lower() in seen
                or (entity.get("confidence") or 0) < KEYWORD_CONFIDENCE):
            continue
        seen.add(name.lower())
        keywords.append(name)
    return keywords[:limit]


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



CLASSIFY_PROMPT = (
    "What kind of post is this? Answer as a 2 to 5 word search query naming "
    "the place, if you can tell, and the type of content: a travel "
    "destination, a sport, a gadget, a hobby, or a style of photography. "
    "Name the genre, not the objects in the frame. Output only the query."
)


def voice_examples(voice, limit=6):
    """The user's own past captions, one per line, used as style examples."""
    return [line.strip() for line in (voice or "").splitlines() if line.strip()][:limit]


def build_prompt(story, topics, voice=""):
    parts = [CLASSIFY_PROMPT]
    examples = voice_examples(voice)
    if examples:
        listed = "\n".join(f"- {e}" for e in examples)
        parts.append(
            "Captions this person has written before:\n"
            f"{listed}\n"
            "Match their voice: the same length, punctuation, capitalization and "
            "sense of humor. Do not reuse their wording. Where their style "
            "conflicts with the rules above, follow their style, except never "
            "use hashtags."
        )

    if story:
        parts.append(f"The person posting this says: {story}")
    if topics:
        lines = []
        for t in topics:
            kw = keywords_of(t)
            lines.append(f"- {title_of(t)}" + (f" (keywords: {', '.join(kw)})" if kw else ""))
        parts.append(
            "These subjects are getting attention right now:\n"
            + "\n".join(lines) + "\n"
            "Borrow at most one as an angle, and only if it genuinely connects to "
            "the image. Write it as your own passing thought, never as news or a "
            "headline. If you borrow one, work one or two of its keywords into the "
            "sentence as plain words, using the everyday short name people actually "
            "search rather than the formal one. Never as hashtags. "
            "If none connect, ignore this list completely."
        )
    return "\n\n".join(parts)


HASHTAG = r"#[^\W\d_]\w*"


def strip_hashtags(caption):
    """Drop a trailing hashtag block, turn inline hashtags into plain words.

    Only tags starting with a letter count, so "#1", "Room #204" and "C#"
    survive.
    """
    caption = re.sub(rf"(\s*{HASHTAG})+\s*$", "", caption)
    return re.sub(rf"#([^\W\d_]\w*)", r"\1", caption).strip()


