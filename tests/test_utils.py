from src.utils import (
    get_required_params,
    parse_argline,
    parse_docstring,
    build_prompt,
    count_number_of_tokens,
)


def test_get_required_params():
    def f(x, y=1):
        return x + y

    assert get_required_params(f) == ["x"]


def test_parse_argline():
    argline = "param (type): param description"
    assert parse_argline(argline) == ("param", "type", "param description")


def test_parse_docstring():
    docstring = """
    function description

    Args:
        param (type): param description

    Returns:
        type: return description
    """

    assert parse_docstring(docstring) == {
        "description": "function description",
        "parameters": {
            "type": "object",
            "properties": {
                "param": {"type": "type", "description": "param description"}
            },
        },
    }


def test_build_prompt():
    messages = [
        {"role": "system", "content": "my system prompt"},
        {"role": "user", "content": "user question"},
        {"role": "assistant", "content": "assistant response"},
    ]

    prompt = build_prompt(messages)
    assert prompt == (
        "<|im_start|>system\nmy system prompt<|im_end|>\n<|im_start|>user\n"
        "user question<|im_end|>\n<|im_start|>assistant\n"
        "assistant response<|im_end|>\n"
    )


def test_count_number_of_tokens():
    assert count_number_of_tokens("hello world") == 2
