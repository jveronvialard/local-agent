import re
import inspect
from typing import Callable

from tiktoken import get_encoding


def get_required_params(f: Callable) -> list:
    """Get the required arguments of a function"""
    params = inspect.signature(f).parameters
    return [arg_name for arg_name, v in params.items() if v.default is inspect._empty]


def parse_argline(argline: str) -> tuple[str]:
    """Parse an argument description in a Google-style docstring."""
    i = argline.index("(")
    j = argline.index(")")
    argname, argtype, argdesc = argline[: i - 1], argline[i + 1 : j], argline[j + 3 :]
    argname, argtype, argdesc = (
        argname.replace("\n", ""),
        argtype.replace("\n", ""),
        argdesc.replace("\n", ""),
    )
    argname, argtype, argdesc = (
        re.sub(" +", " ", argname),
        re.sub(" +", " ", argtype),
        re.sub(" +", " ", argdesc),
    )
    argname, argtype, argdesc = (
        argname.strip(" "),
        argtype.strip(" "),
        argdesc.strip(" "),
    )
    return argname, argtype, argdesc


def parse_docstring(docstring: str) -> dict:
    """Parse a Google-style docstring."""
    d = {"description": "", "parameters": {"type": "object", "properties": {}}}

    i = docstring.index("Args:")
    j = docstring.index("Returns:")

    d["description"] = " ".join(
        [e.strip(" ") for e in docstring[:i].split("\n") if len(e.strip(" ")) > 0]
    )

    for e in docstring[i + 5 + 1 : j].split("\n"):
        if e.startswith("        "):
            argname, argtype, argdesc = parse_argline(e)
            d["parameters"]["properties"].update(
                {argname: {"type": argtype, "description": argdesc}}
            )

    return d


def build_prompt(messages: list) -> str:
    """Build chatML prompt."""
    # Note: https://huggingface.co/docs/transformers/main/chat_templating 
    # at the cost of an extra dependency
    user_offset = 0
    if messages[0]["role"] == "system":
        user_offset = 1
        prompt = "<|im_start|>system\n{system_prompt}<|im_end|>\n".format(
            system_prompt=messages[0]["content"]
        )
    elif messages[0]["role"] == "user":
        prompt = "<|im_start|>user\n{user_message}<|im_end|>\n".format(
            user_message=messages[0]["content"]
        )
    else:
        raise ValueError

    for idx, m in enumerate(messages[1:]):
        if ((idx + user_offset) % 2 == 1) and m["role"] == "user":
            prompt += "<|im_start|>user\n{user_message}<|im_end|>\n".format(
                user_message=m["content"]
            )
        elif ((idx + user_offset) % 2 == 0) and m["role"] == "assistant":
            prompt += "<|im_start|>assistant\n{user_message}<|im_end|>\n".format(
                user_message=m["content"]
            )
        else:
            raise ValueError
    return prompt


def count_number_of_tokens(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a string."""
    encoding = get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
