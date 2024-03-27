# Local Agent 

Minimalistic example with few dependencies for building and testing an agent using open-weight models running on a local machine.

## Introduction
Last year, I quickly became amazed by open-source initiatives for enabling Large-Language-Models (LLMs) to run on consumer laptops. I experimented with these local approaches for some time, and recently observed improvements in the quality of function calling. Thus, I decided to build a simple agent tailored to my personal requirements. While this implementation is not fit for production needs, it can be an interesting resource for understanding how to build and evaluate such agents, and how to run them locally with complete control on the generation process. 

Key takeaways:
- the more tools involved the harder it gets for the agent to make the correct choices (and errors compound within system workflows), so specialized workflows should be preferred;
- getting consistent experiences remains challenging, therefore incorporating tracing and evaluation mechanisms is critical to build effective workflows;
- faster inference directly translate into quicker interations for prototyping.

Note: While sharing my implementation on GitHub, I encountered a similar project called [Hermes-Function-Calling](https://github.com/NousResearch/Hermes-Function-Calling) with a model fine-tuned specifically for function calling. I recommend exploring it as well.

## Agent
The agent uses a local Large-Language-Model (LLM) for reasoning and interacting with the user. I am using [llama.cpp](https://github.com/ggerganov/llama.cpp) for LLM inference, with the default configurations for the high-level API since I have not noticed meaningful speed-up from e.g. `LlamaPromptLookupDecoding`. The agent also uses an Automatic-Speech-Recognition (ASR) model for transcribing audio into text. I am using [whisper.cpp](https://github.com/ggerganov/whisper.cpp) for the ASR inference. For insights on hardware requirements, refer to: https://github.com/ggerganov/llama.cpp/discussions/4167. For details about quantization methods and max RAM required, refer to: https://huggingface.co/TheBloke/Nous-Hermes-2-Yi-34B-GGUF#provided-files.

The agent is defined in `src/agent.py`. The agent prompting is inspired from [ReAct](https://arxiv.org/abs/2210.03629) where questions are answered by interleaving Though, Action, and Observation steps. This implementation uses system/user/assistant blocks and prompts the assistant to choose a tool at each step (writing the final response is one of the tool), with the previous function calls being passed as additional context. The tools are defined in `src/tools.py`, with the function logic extracted from the function definition and [Google-style](https://google.github.io/styleguide/pyguide.html) docstrings.

Note: one could also use the `chat-format` parameter in llama-cpp-python for building prompts.

## Model choice
Typical benchmarks:  [Chatbot Arena](https://chat.lmsys.org/), [Alpaca Eval](https://tatsu-lab.github.io/alpaca_eval/), [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard), or [Wild Bench](https://huggingface.co/spaces/allenai/WildBench). It's worth noting that models might outpace evaluation benchmarks these days, so do take these comparisons with caution. You might find [r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/) useful for further insights and community feedback.

Recently, I have used models fine-tuned by [NousResearch](https://nousresearch.com/) and quantized by [TheBloke](https://huggingface.co/TheBloke). Some examples include: https://huggingface.co/NousResearch/Nous-Hermes-2-Yi-34B, https://huggingface.co/NousResearch/Hermes-2-Pro-Mistral-7B, https://huggingface.co/NousResearch/Nous-Hermes-2-SOLAR-10.7B, and https://huggingface.co/Nexusflow/NexusRaven-V2-13B. NousResearch also provides a [suite](https://huggingface.co/collections/NousResearch/yarn-6510f87837698373cd302ac2) of models with longer context window.

Key takeaways: 
- not all models have the same license;
- smaller models typically have memorized less knowlegde and also make more failed function calling attempts. 

Note: `demo.ipynb` runs a couple examples with `Nous-Hermes-2-Yi-34B` and gives the elapsed time for a Macbook with Apple Silicon.

## Evaluation
This toy evaluation suite focuses on workflow evaluation. Workflows consist of multiple tasks, and the quality of downstream tasks depends on the performance of upstream tasks.

## Running the agent
Environment setup:
```
conda create --name agent
conda activate agent
pip install -r requirements.txt 
```

Environment variables (optional, required for Reddit API and OpenWeatherMap API):
```
export REDDIT_CLIENT_ID=XXX
export REDDIT_CLIENT_SECRET=XXX
export REDDIT_USER_AGENT=XXX
export OPENWEATHERMAP_API_KEY=XXX
```

To download a model:
```
huggingface-cli download NousResearch/Nous-Hermes-2-Yi-34B-GGUF Nous-Hermes-2-Yi-34B.Q3_K_M.gguf --local-dir ./models/ --local-dir-use-symlinks False
```

To run the toy evaluation suite:
```
python evaluation.py --model_path ./models/Nous-Hermes-2-Yi-34B.Q3_K_M.gguf
```

Here are the results for a typical run:
```
# Tools - Basic
## Get Quote
### Fake Service - Fixed price
response keywords - and: 100 / 100
actions: 100 / 100
attempts: 1
### Fake Service - Unexpected Results
response keywords - and: 100 / 100
actions: 100 / 100
attempts: 2
## YouTube
response keywords - and: 67 / 100
actions: 67 / 100
attempts: 2
## Reddit
actions: 0 / 100
attempts: 5
## Weather
actions: 100 / 100
attempts: 2
# Tools - Advanced
## Follow-up & Ability to correct itself
response keywords - and: 100 / 100
actions: 100 / 100
attempts: 2
## Many tools to choose from & General questions
actions: 88 / 100
attempts: 2
```
`response keywords - and` assesses the quality of the final output, `actions` assesses the quality of the sequence of actions taken, `attempts` assesses the number of failed attempts to JSON parse function calls. Refer to `evaluation.py` for more details. 

To run unit tests:
```
pytest -s
```

## Tentative Roadmap
- use newly released models fine-tuned for function calling
- option to add and remove functions, and to call class methods instead of function wrapper
- enhance list of tools
    - PDF / arxiv
- enhance evaluation suite
    - embedding based metrics
- other ASR model
- constrained generation

## Contributing
Feel free to open a PR :)
