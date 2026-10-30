import json

from catalog_agent.tagger import parse_reply, request_body, value_confidence


def test_confidence_uses_only_the_value_tokens():
    tokens = [
        ('{"', 0.0),
        ("value", -0.1),
        ('":', 0.0),
        (' "', 0.0),
        ("Short", -0.2),
        (" Sleeve", -0.3),
        ('"}', -5.0),
    ]
    assert value_confidence(tokens) == round(2.718281828**-0.5, 4)


def test_schema_forces_an_allowed_value():
    body = request_body("m", {"category": "Tops", "title": "Tee"}, "Sleeve", ["A", "B"], "u")
    schema = body["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["value"]["enum"] == ["A", "B"]
    assert "Options: A, B" in body["messages"][0]["content"][1]["text"]


def test_unparseable_reply_gives_no_value():
    reply = {"choices": [{"message": {"content": "not json"}}]}
    assert parse_reply(reply).value is None
    good = {"choices": [{"message": {"content": json.dumps({"value": "A"})}, "logprobs": None}]}
    assert parse_reply(good).value == "A"
