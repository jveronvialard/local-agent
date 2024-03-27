import json
import logging
import os

from llama_cpp import Llama

from src.utils import parse_docstring, build_prompt


class LocalAgent(object):
    """
    Toy agent
    """

    def __init__(
        self,
        model_path: str,
        functions: list = [],
        environ: dict = {},
        output_dir: str = "./tmp/",
        logger=None,
    ):
        self.model_path = model_path
        self.environ = environ
        self.output_dir = output_dir
        self.logger = logger

        # Model
        if self.model_path.endswith(
            "Nous-Hermes-2-Yi-34B.Q3_K_M.gguf"
        ) or self.model_path.endswith("nous-hermes-2-solar-10.7b.Q4_K_M.gguf"):
            self.llm = Llama(
                model_path=self.model_path, n_gpu_layers=-1, n_ctx=4096, verbose=False
            )
        else:
            raise ValueError(
                # "Unsupported model: {model_path}".format(model_path=model_path)
                f"Unsupported model: {model_path}"
            )

        # Logger
        if os.path.isdir(self.output_dir) is False:
            os.mkdir(self.output_dir)

        if self.logger is None:
            logging.basicConfig(
                filename=self.output_dir + "agent.log",
                format="%(asctime)s %(message)s",
                filemode="w",
            )
            self.logger = logging.getLogger()
            self.logger.setLevel(logging.DEBUG)

        # Functions
        self.function_dict = {
            "reply_with_answer": {  # final function call
                "function": lambda answer: str(answer),
                "description": (
                    "This function lets you reply to the user question "
                    "when you have sufficient information."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "description": "Your response to the user question.",
                        },
                    },
                },
            },
        }

        for function in functions:
            d = parse_docstring(function.__doc__)
            d["function"] = function
            self.function_dict.update({function.__name__: d})

        # System prompt
        self.system_prompt = "You are an helpful assistant.\n"

        if len(self.environ) > 0:
            self.system_prompt += (
                "\nYou have access to the following environment variables:\n"
                + "\n".join([f"{k}={v}" for k, v in self.environ.items()])
                # + "\n".join(["{}={}".format(k, v) for k, v in self.environ.items()])
                + "\n"
            )

        self.system_prompt += "\nYou have access to the following functions:\n"
        for function_name in ["reply_with_answer"] + [f.__name__ for f in functions]:
            self.system_prompt += (
                "def "
                + function_name
                + "("
                + ", ".join(
                    self.function_dict[function_name]["parameters"]["properties"].keys()
                )
                + "):\n"
            )
            self.system_prompt += (
                '   """\n   ' + self.function_dict[function_name]["description"] + "\n"
            )
            self.system_prompt += "   Args:\n"
            for key, value in self.function_dict[function_name]["parameters"][
                "properties"
            ].items():
                self.system_prompt += "      {name} ({type}): {description}\n".format(
                    name=key, type=value["type"], description=value["description"]
                )
            self.system_prompt += '   """\n'

        self.system_prompt += (
            "\nYour responses must always be formatted as:\n"
            "{\n"
            '   "thought": "Think about what you need to do and which function you need to call."\n'
            '   "name": "The name of the function you thought of calling.",\n'
            '   "arguments": "The function arguments as {"argument":"value"}."'
            "\n}\n\n"
        )

        self.system_prompt += (
            "Important:\n"
            "1/ You have already memorized a lot of information "
            "and should be able to answer some questions directly.\n"
            "2/ Use `reply_with_answer` to reply to the user question "
            "when you have sufficient information.\n"
            "3/ Pay special attention to function descriptions. "
            "Do not use functions that are not relevant and do not invent new functions.\n"
            "4/ If asked to summarize text, aim for a 2 or 3 lines summary.\n"
            '5/ If you need to save files, use "./tmp/" as temporary directory.\n'
        )

    def _llm(self, prompt: str) -> str:
        """LLM inference."""
        try:
            output = self.llm(
                prompt,
                max_tokens=300,
                echo=False,
                temperature=0.8,  # To get alternative answers
            )
        except Exception as e:
            t = type(e)
            s = str(e)
            msg = "{}: {}".format(t, s)
            return None, msg

        prediction = output["choices"][0]["text"]
        self.logger.info(prediction)
        return prediction, ""

    def _parse_prediction(self, prediction: str) -> tuple:
        """Parse JSON prediction."""
        try:
            output = json.loads(prediction.strip("\n"))
            name = output["name"]
            arguments = output["arguments"]
            if name.endswith("()"):
                name = name[:-2]
        except Exception as e:
            msg = "JSON can not be parsed"
            self.logger.debug(msg)
            return None, None, msg
        if isinstance(name, str) is False:
            msg = '"name" is not a string'
            self.logger.debug(msg)
            return name, arguments, msg
        if isinstance(arguments, dict) is False:
            msg = '"arguments" must be returned as {"argument":"value"}'
            self.logger.debug(msg)
            return name, arguments, msg
        if name not in self.function_dict.keys():
            msg = 'there isn\'t a function called "{}"'.format(name)
            self.logger.debug(msg)
            return name, arguments, msg
        elif (
            len(
                set(arguments.keys()).difference(
                    set(self.function_dict[name]["parameters"]["properties"].keys())
                )
            )
            > 0
        ):  # TODO: required parameters enhancement
            # msg = 'incorrect arguments for function "{}"'.format(name)
            msg = f'incorrect arguments for function "{name}"'
            self.logger.debug(msg)
            return name, arguments, msg
        else:
            for key, value in arguments.items():
                if (
                    isinstance(value, str)
                    and len(value) > 3
                    and value[:2] == "${"
                    and value[-1] == "}"
                    and value[2:-1] in self.environ.keys()
                ):
                    arguments[key] = self.environ[value[2:-1]]
                if isinstance(value, str) and value in self.environ.keys():
                    arguments[key] = self.environ[value]
            return name, arguments, ""

    def _do_function_call(self, name: str, arguments: str) -> str:
        """Function call."""
        self.logger.debug("\n*** FUNCTION CALL ***\n")
        self.logger.debug(
            # "{name}(**{arguments}):\n".format(name=name, arguments=arguments)
            "%s(**%s):\n", name, arguments
        )
        try:
            result = self.function_dict[name]["function"](**arguments)
        except Exception as e:
            t = type(e)
            s = str(e)
            result = "{}: {}".format(t, s)
        # self.logger.debug(result.strip("\n") + "\n**************")
        self.logger.debug("%s\n**************", result.strip("\n"))
        return result

    def ask(self, question: str, messages: list = []) -> tuple:
        """Entry point for asking a question, potentially with previous messages."""
        history = []

        if len(messages) == 0:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    # "content": "Task:\n{question}\n\n".format(question=question),
                    "content": f"Task:\n{question}\n\n",
                },
            ]
        else:
            assert messages[-1]["role"] == "assistant"
            messages.append(
                {
                    "role": "user",
                    # "content": "Task:\n{question}\n\n".format(question=question),
                    "content": f"Task:\n{question}\n\n",
                }
            )

        while True:
            prompt = build_prompt(messages) + "<|im_start|>assistant"
            # self.logger.debug(
            #     "\n*** PROMPT ***\n" + prompt.strip("\n") + "\n**************"
            # )
            self.logger.debug(
                "\n*** PROMPT ***\n%s\n**************", prompt.strip("\n")
            )

            # llm chooses action in JSON format
            counter = 0
            prediction, llm_msg = self._llm(prompt)
            if prediction is None:
                return llm_msg, history, messages

            name, arguments, msg = self._parse_prediction(prediction)

            while counter < 5 and (
                name is None
                or msg != ""
                or (
                    len(history) > 0
                    and name == history[-1]["name"]
                    and arguments == history[-1]["arguments"]
                )
            ):  # make other attempts
                prediction, llm_msg = self._llm(prompt)
                if prediction is None:
                    return llm_msg, history, messages
                name, arguments, msg = self._parse_prediction(prediction)
                counter += 1

            # result
            if name is None:
                return "LLM is unable to follow JSON format", history, messages
            if msg != "":
                result = msg
            else:
                result = self._do_function_call(name, arguments)

            messages.append({"role": "assistant", "content": prediction})
            history.append(
                {
                    "name": name,
                    "arguments": arguments,
                    "result": result,
                    "attempts": counter + 1,
                }
            )

            if len(history) >= 2 and history[-1] == history[-2]:
                return "Repeated call to the same function.", history, messages

            if len(history) > 5:
                return "Too many function calls.", history, messages

            # return final response
            if name.startswith("reply_with_answer") and msg == "":
                return result, history, messages

            # follow-up action needed
            # content = "Task:\n{question}\n\n".format(question=question)
            content = f"Task:\n{question}\n\n"
            for d in history:
                content += "You previously ran ```{name}(**{arguments})``` ".format(
                    name=d["name"], arguments=d["arguments"]
                )
                content += "and got ```{result}```\n".format(
                    result=d["result"]
                )
            content += (
                "\n\nRemember:\n"
                "1/ You have already memorized a lot of information "
                "and should be able to answer some questions directly.\n"
                "2/ Use `reply_with_answer` to reply to the user question "
                "when you have sufficient information.\n"
                "3/ Pay special attention to function descriptions. "
                "Do not use functions that are not relevant and do not invent new functions.\n"
                "4/ If asked to summarize text, aim for a 2 or 3 lines summary.\n"
                '5/ If you need to save files, use "./tmp/" as temporary directory.\n'
            )
            messages.append({"role": "user", "content": content})
