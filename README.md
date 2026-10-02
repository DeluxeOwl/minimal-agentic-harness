# minimal-agentic-harness

A minimal Python coding-agent harness, built to teach how agents work.
The core loop calls a model, runs its requested tools, and sends the results back
until the model answers. The harness adds a terminal UI, typed tools, subagents,
`AGENTS.md` instructions, skills, and file-based memory. The code is in
[`src/cmd/`](./src/cmd/).

See the [attached presentation](./index.html) for a step-by-step walkthrough in
English or Romanian.

## Run it

The project requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).
Install the dependencies:

```bash
uv sync --locked
```

`make run` uses DeepSeek through [OpenRouter](https://openrouter.ai/) by default,
so it needs an API key:

```bash
export OPENROUTER_API_KEY="your-key"
```

The memory observer and explorer subagent also use a local MiniCPM server.
Follow the [local model setup](./docs/minicpm5-dspark-llama-cpp/README.md) first.
Then start the server in a separate terminal:

```bash
make serve-llm
```

Leave the server active. In the terminal where you set the API key, run the agent:

```bash
make run
```

Without an OpenRouter key, run the local explorer instead:

```bash
make explore
```

It uses the local model with read-only file tools, without skills or memory.

## Change the model

Model definitions are in [`src/cmd/models.py`](./src/cmd/models.py). Agent model
selections are in [`main.py`](./src/cmd/main.py),
[`subagent.py`](./src/cmd/subagent.py), and [`memory.py`](./src/cmd/memory.py).

For a fully local setup, replace `models.CloudDeepseek` with `models.LocalMiniCPM`
in those three agent files. The worker subagent and memory consolidator otherwise
still use OpenRouter. You can also edit `models.py` to use another OpenAI-compatible
model or provider.

The Bash tool runs commands directly on your machine, without a sandbox or
permission prompts.