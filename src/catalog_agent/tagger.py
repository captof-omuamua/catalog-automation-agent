"""Ask the fine-tuned vision model for one attribute of one product: value + confidence.

Same question, answer schema and confidence as the visual-product-tagging project, so its
validated cut-offs still apply:
  - the answer is forced (JSON schema) to be one of the attribute's allowed values
  - confidence = the model's probability for the value it picked: exp(sum of the log-probs
    of the value's tokens); the JSON around the value is forced, so it carries no signal
"""

import base64
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path

import httpx
from PIL import Image

PROMPT = (
    "Product category: {category}\nTitle: {title}\n"
    "Question: Looking at the photo and the title, what is the product's {attribute}?\n"
    "Options: {options}\nAnswer in JSON."
)


@dataclass
class Tag:
    value: str | None  # None if the call failed or the answer could not be parsed
    confidence: float  # 0..1; 0 when value is None
    error: str = ""


def image_data_url(image: bytes | str | Path, size: int) -> str:
    """Shrink so the longer side is at most `size` px; inline it as a JPEG data URL."""
    img = Image.open(io.BytesIO(image) if isinstance(image, bytes) else image).convert("RGB")
    img.thumbnail((size, size))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def request_body(model: str, product: dict, attribute: str, options: list[str], image_url: str):
    prompt = PROMPT.format(
        category=product["category"],
        title=product["title"],
        attribute=attribute,
        options=", ".join(options),
    )
    schema = {
        "type": "object",
        "properties": {"value": {"enum": options}},
        "required": ["value"],
        "additionalProperties": False,
    }
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": 32,
        "temperature": 0.0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "attributes", "schema": schema},
        },
        "logprobs": True,
        "top_logprobs": 5,
    }


def value_confidence(tokens: list[tuple[str, float]], field: str = "value") -> float | None:
    """exp(sum of log-probs) of the tokens inside this field's value."""
    text = "".join(t for t, _ in tokens)
    key = text.find(f'"{field}"')
    if key < 0:
        return None
    a = text.find('"', text.find(":", key)) + 1  # first character of the value
    b = text.find('"', a)  # closing quote
    lp, pos = [], 0
    for token, logprob in tokens:
        if pos < b and pos + len(token) > a:  # token overlaps the value's characters
            lp.append(logprob)
        pos += len(token)
    return round(math.exp(sum(lp)), 4) if lp else None


def parse_reply(reply: dict) -> Tag:
    choice = reply["choices"][0]
    try:
        value = json.loads(choice["message"]["content"])["value"]
    except (ValueError, TypeError, KeyError):
        return Tag(None, 0.0, "unparseable answer")
    content = (choice.get("logprobs") or {}).get("content") or []
    tokens = [(c.get("token", ""), float(c.get("logprob", 0.0))) for c in content]
    return Tag(value, value_confidence(tokens) or 0.0)


async def tag(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    product: dict,
    attribute: str,
    options: list[str],
    image: bytes | str | Path,
    size: int = 512,
) -> Tag:
    """One attribute of one product. Raises httpx errors so the caller can retry."""
    body = request_body(model, product, attribute, options, image_data_url(image, size))
    r = await client.post(f"{base_url}/v1/chat/completions", json=body, timeout=120)
    r.raise_for_status()
    return parse_reply(r.json())
