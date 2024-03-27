import argparse
import logging
import os

import numpy as np
from tqdm import tqdm

from src.agent import LocalAgent
from src.tools import (
    get_stock_price,
    download_youtube_audio_as_mp3,
    convert_from_mp3_to_wav,
    get_wav_transcript,
    get_most_recent_reddit_submissions,
    get_current_temperature,
)


def evaluate(local_agent: LocalAgent, test_dataset: list) -> None:
    """Evaluate local agent on a list of workflows."""
    scores = {
        "response keywords - and": np.zeros(len(test_dataset)),
        "response keywords - or": np.zeros(len(test_dataset)),
        "actions": np.zeros(len(test_dataset)),
        "attempts": np.zeros(len(test_dataset)),
    }

    for index in tqdm(range(len(test_dataset))):
        test_data = test_dataset[index]

        # Inference
        response, history, _ = local_agent.ask(
            question=test_data["question"], messages=test_data.get("messages", [])
        )

        # Evaluate final response
        if (
            "response keywords - and" not in test_data.keys()
            or len(test_data["response keywords - and"]) == 0
        ):
            scores["response keywords - and"][index] = np.nan
        else:
            scores["response keywords - and"][index] = sum(
                [
                    keyword.lower() in response.lower()
                    for keyword in test_data["response keywords - and"]
                ]
            ) / len(test_data["response keywords - and"])

        if (
            "response keywords - or" not in test_data.keys()
            or len(test_data["response keywords - or"]) == 0
        ):
            scores["response keywords - or"][index] = np.nan
        else:
            scores["response keywords - or"][index] = max(
                [
                    keyword.lower() in response.lower()
                    for keyword in test_data["response keywords - or"]
                ]
            )

        # Evaluate flow
        ## correct sequence * (1 - % of incorrect steps)
        i = 0
        for t in history:
            if t["name"] == test_data["actions"][i]:
                i += 1
            if i == len(test_data["actions"]):
                break
        correct_actions = (i == len(test_data["actions"])) * (
            1 - (len(history) - len(test_data["actions"])) / len(test_data["actions"])
        )
        scores["actions"][index] = correct_actions

        # Evaluate correctness of each step
        # TODO: based on `local_agent.history`

        # Number of attempts
        scores["attempts"][index] = np.array([e["attempts"] for e in history]).mean()     

    print(
        "{}: {:.0f} / 100".format(
            "response keywords - and",
            np.nanmean(scores["response keywords - and"]) * 100,
        )
    )
    print(
        "{}: {:.0f} / 100".format(
            "response keywords - or", np.nanmean(scores["response keywords - or"]) * 100
        )
    )
    print("{}: {:.0f} / 100".format("actions", np.nanmean(scores["actions"]) * 100))
    print("{}: {:.0f}".format("attempts", np.nanmean(scores["attempts"])))

    return scores


def main(args):
    """Setup local agent and run evaluations."""
    # Logger
    logging.basicConfig(
        filename="evaluation.log", format="%(asctime)s %(message)s", filemode="w"
    )

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Local Agent
    environ = {}
    for k in [
        "REDDIT_CLIENT_ID",
        "REDDIT_CLIENT_SECRET",
        "REDDIT_USER_AGENT",
        "OPENWEATHERMAP_API_KEY",
    ]:
        v = os.environ.get(k, None)
        if v:
            environ.update({k: v})

    # Tools - Basic
    ## Get Quote
    ### Fake Service - Fixed price
    print("# Tools - Basic")
    print("## Get Quote")
    print("### Fake Service - Fixed price")
    local_agent = LocalAgent(
        model_path=args.model_path,
        functions=[get_stock_price],
        environ=environ,
        logger=logger,
    )
    local_agent.function_dict["get_stock_price"]["function"] = lambda ticker: "70.23"

    test_dataset = [
        {
            "question": "What's the current stock price of Amazon?",
            "response keywords - and": ["70.23"],
            "actions": ["get_stock_price", "reply_with_answer"],
        },
        {
            "question": "Get me the current stock price of NVIDIA",
            "response keywords - and": ["70.23"],
            "actions": ["get_stock_price", "reply_with_answer"],
        },
    ]

    _ = evaluate(local_agent, test_dataset)

    ## Fake Service - Unexpected Results
    print("### Fake Service - Unexpected Results")
    local_agent = LocalAgent(
        model_path=args.model_path,
        functions=[get_stock_price],
        environ=environ,
        logger=logger,
    )
    local_agent.function_dict["get_stock_price"][
        "function"
    ] = lambda ticker: "Unable to reach Yahoo Finance"
    test_dataset = [
        {
            "question": "What's the current stock price of Amazon?",
            "response keywords - and": ["unable"],
            "actions": ["get_stock_price", "reply_with_answer"],
        },
    ]

    _ = evaluate(local_agent, test_dataset)

    # Tools - Basic
    ## YouTube
    print("## YouTube")
    local_agent = LocalAgent(
        model_path=args.model_path,
        functions=[
            download_youtube_audio_as_mp3,
            convert_from_mp3_to_wav,
            get_wav_transcript,
        ],
        environ=environ,
        logger=logger,
    )
    test_dataset = [
        {
            "question": (
                'Which event is mentionned in this video? '
                '"https://www.youtube.com/watch?v=n9R1ts_xfgc"'
            ),
            "response keywords - and": ["GTC"],
            "actions": [
                "download_youtube_audio_as_mp3",
                "convert_from_mp3_to_wav",
                "get_wav_transcript",
                "reply_with_answer",
            ],
        },
        {
            "question": (
                'How does Dario think about model scaling? '
                '"https://www.youtube.com/watch?v=Nlkk3glap_U"'
            ),
            "response keywords - and": ["Dario"],
            "actions": [
                "download_youtube_audio_as_mp3",
                "convert_from_mp3_to_wav",
                "get_wav_transcript",
                "reply_with_answer",
            ],
        },
        {
            "question": "What is Claude? https://www.youtube.com/watch?v=5GtVrk00eck",
            "response keywords - and": ["Anthropic"],
            "actions": [
                "download_youtube_audio_as_mp3",
                "convert_from_mp3_to_wav",
                "get_wav_transcript",
                "reply_with_answer",
            ],
        },
    ]

    _ = evaluate(local_agent, test_dataset)

    # Tools - Basic
    ## Reddit
    print("## Reddit")
    local_agent = LocalAgent(
        model_path=args.model_path,
        functions=[get_most_recent_reddit_submissions],
        environ=environ,
        logger=logger,
    )
    test_dataset = [
        {
            "question": "Summarize the most recent topics of discussion on r/LocalLLaMA",
            "actions": ["get_most_recent_reddit_submissions", "reply_with_answer"],
        },
    ]

    _ = evaluate(local_agent, test_dataset)

    # Tools - Basic
    ## Weather
    print("## Weather")
    local_agent = LocalAgent(
        model_path=args.model_path,
        functions=[get_current_temperature],
        environ=environ,
        logger=logger,
    )
    test_dataset = [
        {
            "question": "Should I take a warm coat before going out?",
            "actions": ["get_current_temperature", "reply_with_answer"],
        },
    ]

    _ = evaluate(local_agent, test_dataset)

    # Tools - Advanced
    ## Follow-up & Ability to correct itself
    print("# Tools - Advanced")
    print("## Follow-up & Ability to correct itself")
    local_agent = LocalAgent(
        model_path=args.model_path,
        functions=[get_stock_price],
        environ=environ,
        logger=logger,
    )
    local_agent.function_dict["get_stock_price"]["function"] = lambda ticker: "80.29"

    test_dataset = [
        # followup
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        'You are an helpful assistant.\n\n'
                        'You have access to the following functions:\n'
                        'def reply_with_answer(answer):\n   """\n'
                        '   This function lets you reply to the user question when you have '
                        'sufficient information.\n'
                        '   Args:\n      answer (string): Your response '
                        'to the user question.\n   """\n'
                        'def get_stock_price(ticker):\n   """\n'
                        '   This function queries Yahoo Finance to get the current price for a '
                        'provided stock ticker.\n'
                        '   Args:\n      ticker (str): The stock ticker'
                        ' e.g. AAPL for Apple.\n   """\n\n'
                        'Your responses must always be formatted as:\n'
                        '{\n   "thought": "Think about what you need to do and which function'
                        ' you need to call."'
                        '\n   "name": "The name of the function you thought of calling.",'
                        '\n   "arguments": "The function arguments as '
                        '{"argument":"value"}."\n}\n\n'
                        'Important:\n'
                        '1/ You have already memorized a lot of information '
                        'and should be able to answer some questions directly.\n'
                        '2/ Use `reply_with_answer` to reply to the user question when you have '
                        'sufficient information.\n'
                        '3/ Pay special attention to function descriptions. Do not use'
                        ' functions that are not relevant and do not invent new functions.\n'
                        '4/ If asked to summarize text, aim for a 2 or 3 lines summary.\n'
                        '5/ If you need to save files, use "./tmp/" as temporary directory.'
                    ),
                },
                {
                    "role": "user",
                    "content": "Task:\nWhat's the current stock price of Amazon?\n\n",
                },
                {
                    "role": "assistant",
                    "content": (
                        '\n{\n    "thought": "I need to find out the current stock price of '
                        'Amazon.",\n    "name": "get_stock_price",\n    "arguments": '
                        '{"ticker": "AMZN"}\n}'
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Task:\nWhat's the current stock price of Amazon?\n\nYou previously ran"
                        " ```get_stock_price(**{'ticker': 'AMZN'})``` and got ```176.555```\n"
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        '\n{\n    "thought": "I already got the current stock price of '
                        'Amazon.",\n    "name": "reply_with_answer",\n    "arguments": '
                        '{"answer": "The current stock price of Amazon is $176.555."}\n}'
                    ),
                },
            ],
            "question": (
                "And what about Google?\n\nYou previously ran ```get_stock_price(**"
                "{'ticker': 'AMZN'})``` and got ```176.555```\nYou previously ran "
                "```reply_with_answer(**{'answer': 'The current stock price of Amazon"
                " is $176.555.'})``` and got ```The current stock price of Amazon is $176.555.```\n"
            ),
            "response keywords - and": ["80.29"],
            "actions": ["get_stock_price", "reply_with_answer"],
        },
        # correct itself
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        'You are an helpful assistant.\n\n'
                        'You have access to the following functions:\n'
                        'def reply_with_answer(answer):\n   """\n'
                        '   This function lets you reply to the user question when you have '
                        'sufficient information.\n'
                        '   Args:\n      answer (string): Your response '
                        'to the user question.\n   """\n'
                        'def get_stock_price(ticker):\n   """\n'
                        '   This function queries Yahoo Finance to get the current price for a '
                        'provided stock ticker.\n'
                        '   Args:\n      ticker (str): The stock ticker'
                        ' e.g. AAPL for Apple.\n   """\n\n'
                        'Your responses must always be formatted as:\n'
                        '{\n   "thought": "Think about what you need to do and which function'
                        ' you need to call."'
                        '\n   "name": "The name of the function you thought of calling.",'
                        '\n   "arguments": "The function arguments as '
                        '{"argument":"value"}."\n}\n\n'
                        'Important:\n'
                        '1/ You have already memorized a lot of information '
                        'and should be able to answer some questions directly.\n'
                        '2/ Use `reply_with_answer` to reply to the user question when you have '
                        'sufficient information.\n'
                        '3/ Pay special attention to function descriptions. Do not use'
                        ' functions that are not relevant and do not invent new functions.\n'
                        '4/ If asked to summarize text, aim for a 2 or 3 lines summary.\n'
                        '5/ If you need to save files, use "./tmp/" as temporary directory.'
                    ),
                },
                {
                    "role": "user",
                    "content": "Task:\nWhat's the current stock price of Amazon?\n\n",
                },
                {
                    "role": "assistant",
                    "content": (
                        '\n{\n   "thought": "I need to look up the current Amazon stock price by '
                        'using the \'get_stock_price\' function.",\n   "name": "get_stock_price",'
                        '\n   "arguments": {\n       "ticker": "Amazon"\n   }\n}'
                    ),
                },
            ],
            "question": (
                "What's the current stock price of Amazon?\n\nYou previously ran "
                "```get_stock_price(**{'ticker': 'Amazon'})``` and got "
                "```Amazon is not a valid ticker```\n"
            ),
            "response keywords - and": ["80.29"],
            "actions": ["get_stock_price", "reply_with_answer"],
        },
    ]

    _ = evaluate(local_agent, test_dataset)

    # Tools - Advanced
    ## Many tools to choose from & General questions
    print("# Tools - Advanced")
    print("## Many tools to choose from & General questions")
    local_agent = LocalAgent(
        model_path=args.model_path,
        functions=[
            get_stock_price,
            download_youtube_audio_as_mp3,
            convert_from_mp3_to_wav,
            get_wav_transcript,
            get_most_recent_reddit_submissions,
            get_current_temperature,
        ],
        environ=environ,
        logger=logger,
    )
    local_agent.function_dict["get_stock_price"]["function"] = lambda ticker: "70.23"

    test_dataset = [
        {
            "question": "What's the current stock price of Amazon?",
            "actions": ["get_stock_price", "reply_with_answer"],
        },
        {
            "question": "Summarize the most recent topics of discussion on r/LocalLLaMA",
            "actions": ["get_most_recent_reddit_submissions", "reply_with_answer"],
        },
        {
            "question": "Should I take a warm coat before going out?",
            "actions": ["get_current_temperature", "reply_with_answer"],
        },
        {"question": "What is 2+3-1 equals to?", "actions": ["reply_with_answer"]},
    ]

    _ = evaluate(local_agent, test_dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    args = parser.parse_args()
    print(args)

    main(args)
