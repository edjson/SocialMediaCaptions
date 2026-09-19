import re 
from upriver.upriver import search_topics, UpriverError
from vlm.model import generate
JUDGE_TOKENS = 16

JUDGE_PROMPT = (
    "Candidate trending topics:\n{listed}\n\n"
    "Which of these could a person plausibly mention in a caption for this photo? "
    "The line in brackets is what the topic is actually about; trust it over the "
    "title. A shared place, activity or subject counts as a real connection. A "
    "word that matches in a different context does not. "
    "Reply with matching numbers only, comma separated, or NONE."
)


def judge(image, topics):
    """Drop topics that pass the gate but don't match the photo."""
    if not topics:
        return [], ""
    lines = []
    for i, t in enumerate(topics, 1):
        line = f"{i}. {title_of(t)}"
        snippet = snippet_of(t)
        if snippet:
            line += f"\n   ({snippet})"
        lines.append(line)
    listed = "\n".join(lines)

    verdict, _ = generate(
        image, JUDGE_PROMPT.format(listed=listed), max_new_tokens=JUDGE_TOKENS
    )
    if "NONE" in verdict.upper():
        return [], verdict
    picked = {int(n) for n in re.findall(r"\d+", verdict) if 1 <= int(n) <= len(topics)}
    return [t for i, t in enumerate(topics, 1) if i in picked], verdict

QUERY_TOKENS = 80   # a sentence or two of description, not a short query
CAPTION_TOKENS = 220

DESCRIBE_PROMPT = (
    "Describe this photo in one or two sentences for someone who cannot see "
    "it: the place or setting, what is happening, and the kind of post it "
    "would be, such as travel, food, sport, a gadget, or a style of "
    "photography. Output the description only."
)


def describe(image):
    """Photo -> a sentence or two. The semantic half of the Upriver query."""
    text, _ = generate(image, DESCRIBE_PROMPT, max_new_tokens=QUERY_TOKENS)
    return " ".join(text.split())


def build_query(description, story=""):
    """What goes to Upriver: what the photo shows plus what the person said."""
    return " ".join(part for part in (description, (story or "").strip()) if part)



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


def snippet_of(topic, limit=140):
    """First citation's text: what the topic is actually about.

    Titles are often too vague to judge. "User Shares Creative Project
    Outcome" reads as plausibly relevant until you see its snippet,
    "Pretty happy with how this turned out, what do you think?".
    """
    for citation in topic.get("citations") or []:
        text = (citation.get("snippet") or citation.get("title") or "").strip()
        if text:
            return " ".join(text.split())[:limit]
    return ""


def debug_note(query, raw, candidates, kept, verdict, limit=8):
    """Markdown trace: what was searched, what came back, what survived where."""
    gate_ids = {t.get("topic_id") for t in candidates}
    kept_ids = {t.get("topic_id") for t in kept}
    lines = [
        "---",
        f"**Query sent to Upriver**  \n{query}",
        "",
        f"**{len(raw)} returned · {len(candidates)} passed gate · "
        f"{len(kept)} kept by judge**",
        "",
    ]
    for t in sorted(raw, key=score_of, reverse=True)[:limit]:
        topic_id = t.get("topic_id")
        if topic_id in kept_ids:
            mark = "KEPT"
        elif topic_id in gate_ids:
            mark = "judge dropped"
        else:
            mark = "gate dropped"
        lines.append(f"- `{score_of(t):.3f}` **{title_of(t)}** — {mark}")
        snippet = snippet_of(t)
        if snippet:
            lines.append(f"    - {snippet}")
        kw = keywords_of(t)
        if kw:
            lines.append(f"    - keywords: {', '.join(kw)}")
    lines += ["", f"**Judge replied:** `{verdict.strip() or '(nothing)'}`"]
    return "\n".join(lines)


def retrieve(query, vertical=None, limit=12):
    """Never raises. An API failure degrades to no trends."""
    try:
        return search_topics(query, vertical=vertical, limit=limit)
    except (UpriverError, OSError):
        return []


def gate(topics, min_score=0.10, keep=5):
    """Cheap junk filter. Relevance is judge()'s job, not this one's.

    The old 0.30 threshold tracked relevance when queries were 3-6 words.
    Description-length queries inflate scores across the board (a coffee
    photo's best noise hit went 0.108 -> 0.523), so score no longer separates
    fit from noise. Kept low so real candidates reach the judge, which now
    sees citation snippets. Momentum and status are not filtered on: momentum
    is missing on ~60% of topics and 87% of the corpus is "declining".
    """
    
    kept = [t for t in topics if title_of(t) and score_of(t) >= min_score]
    kept.sort(key=score_of, reverse=True)
    return kept[:keep]



CAPTION_PROMPT = (
    "Write one Instagram caption for this photo.\n"
    "What the person says about the moment is the subject of the caption; "
    "the photo is the setting for it. Any trending subjects listed below are "
    "optional colour, not the point.\n"
    "Aim for one or two sentences that sound like a person talking to "
    "friends, not a brand.\n"
    "Plain punctuation: no em dashes, no emoji, no hashtags. Skip stock "
    "caption phrases and the \"not just X, but Y\" construction.\n"
    "Output the caption text only."
)


def voice_examples(voice, limit=6):
    """The user's own past captions, one per line, used as style examples."""
    return [line.strip() for line in (voice or "").splitlines() if line.strip()][:limit]


def build_prompt(story, topics, voice=""):
    parts = [CAPTION_PROMPT]
    examples = voice_examples(voice)

    if examples:
        listed = "\n".join(f"- {e}" for e in examples)
        parts.append(
            "THE VOICE. Captions this person wrote themselves:\n"
            f"{listed}\n"
            "Write as this person. Everything below is material to write about "
            "in this voice, never a style to copy. Match their length, "
            "punctuation, capitalization and humour. Do not reuse their "
            "wording. Where their voice conflicts with anything above, their "
            "voice wins."
        )

    if story:
        parts.append(f"WHAT IT IS ABOUT. The person says:\n{story}")

    if topics:
        lines = []
        for t in topics:
            kw = keywords_of(t)
            lines.append(f"- {title_of(t)}" + (f" (keywords: {', '.join(kw)})" if kw else ""))
        in_voice = "the voice above" if examples else "a real person"
        parts.append(
            "OPTIONAL COLOUR. Subjects getting attention right now:\n"
            + "\n".join(lines) + "\n"
            "Use at most one, only if it genuinely connects to the photo and "
            f"still sounds like {in_voice}. Write it as a passing thought, "
            "never as news or a headline. If you use one, work one or two of "
            "its keywords in as plain words, in the everyday short form people "
            "search. If none connect, ignore this list."
        )

    parts.append(
        "Write the caption in their voice." if examples
        else "Write it the way a real person would, not a brand."
    )
    return "\n\n".join(parts)
    return "\n\n".join(parts)


HASHTAG = r"#[^\W\d_]\w*"


def strip_hashtags(caption):
    """Drop a trailing hashtag block, turn inline hashtags into plain words.

    Only tags starting with a letter count, so "#1", "Room #204" and "C#"
    survive.
    """
    caption = re.sub(rf"(\s*{HASHTAG})+\s*$", "", caption)
    return re.sub(rf"#([^\W\d_]\w*)", r"\1", caption).strip()


EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF"      # pictographs, faces, transport, symbols
    "\U00002600-\U000027BF"       # misc symbols and dingbats
    "\U00002B00-\U00002BFF"       # arrows and stars
    "\U0001F1E6-\U0001F1FF"       # regional indicators (flags)
    "\uFE0F\u200D]+"              # variation selector, zero-width joiner
)

PREAMBLE = re.compile(
    r"^\s*(?:sure[,!.]?\s*)?(?:here(?:'s| is)[^:\n]{0,40}|caption)\s*[:\-]\s*",
    re.I,
)


def clean_caption(caption):
    """Strip the tells that mark a caption as machine-written.

    The prompt asks for these too, but a 4B ignores instructions often enough
    that the guarantee has to live in code, like the hashtag rule.
    """
    text = strip_hashtags(caption or "")
    text = PREAMBLE.sub("", text)                              # "Here's a caption:"
    text = EMOJI.sub("", text)
    text = re.sub(r"\s*(?:[\u2014\u2013]|--+)\s*", ", ", text)  # em/en dash -> comma
    text = re.sub(r",\s*(?=[,.;:!?])", "", text)               # ", ." -> "."
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)               # no space before punct
    text = re.sub(r"[ \t]+", " ", text).strip()
    if len(text) >= 2 and text[0] in '"\u201c' and text[-1] in '"\u201d':
        text = text[1:-1].strip()                              # unwrap quoted output
    return text
