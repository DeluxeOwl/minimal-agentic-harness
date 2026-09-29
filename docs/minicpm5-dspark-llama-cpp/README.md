# MiniCPM5-2B + DSpark on llama.cpp

Run the unquantized MiniCPM5-2B with its DSpark draft model on llama.cpp. Decoding gets 1.2–3× faster and the output does not change.

Tested on 2026-09-29 on an M3 Pro (36 GB) with llama.cpp b11146 (Homebrew `llama.cpp` 0.5.0).

## The command

```bash
llama-server \
  -m "$LLAMA_CACHE/MiniCPM5-2B-F16.gguf" \
  -md "$LLAMA_CACHE/MiniCPM5-2.6B-DSpark.gguf" \
  --spec-type draft-dspark --spec-draft-n-max 7 \
  -ngl 99 -ngld 99 -fa on \
  --temp 1.0 --top-p 0.95 --min-p 0.0 \
  -c 8192 --jinja -a MiniCPM5-2B \
  --host 127.0.0.1 --port 8090
```

Then open <http://127.0.0.1:8090> for the web chat UI, or use the OpenAI-compatible API at `http://127.0.0.1:8090/v1`.

The models are in `$LLAMA_CACHE`, which is `/Users/andreisurugiu/LLMs` on this Mac. All commands here use that variable, so it must be set in your shell.

## How DSpark works

MiniCPM5-2B is the *target* model: a 2.5B dense Llama-architecture model with a thinking mode and a 128K context. The F16 GGUF is the unquantized release (5.0 GB).

MiniCPM5-2B-DSpark is the *draft* model: 324M parameters, BF16, 653 MB, trained only for this target. Each decoding step works like this:

1. The draft reads hidden states from 5 layers of the target and proposes a block of 7 tokens in one forward pass. A small "Markov head" conditions each position in the block on the previous token.
2. The target checks the 7 tokens in one batched forward pass. It keeps the drafted tokens up to the first one it disagrees with, plus one token of its own.

On a Mac, the time for one forward pass goes mostly into reading the 5 GB of weights. The target can check 8 positions for about the cost of generating 1 token.

Every emitted token is still sampled from the target, so the output quality does not change. In a test with greedy decoding (T=0), the outputs for three prompts were byte-identical to a run without the draft.

## 1. Install llama.cpp

```bash
brew install llama.cpp                                   # or: brew upgrade llama.cpp
llama-server --help 2>/dev/null | grep -o draft-dspark   # must print "draft-dspark"
```

llama.cpp added DSpark in July 2026 ([#25173](https://github.com/ggml-org/llama.cpp/pull/25173)). If the `grep` prints nothing, your build is too old.

## 2. Download the models (5.7 GB)

```bash
# hf CLI: uv tool install huggingface_hub   (or: pip install -U huggingface_hub)
hf download openbmb/MiniCPM5-2B-GGUF        MiniCPM5-2B-F16.gguf      --local-dir "$LLAMA_CACHE"
hf download openbmb/MiniCPM5-2B-DSpark-GGUF MiniCPM5-2.6B-DSpark.gguf --local-dir "$LLAMA_CACHE"
```

## 3. Start the server

Run [the command](#the-command). These lines in the startup log show that DSpark is on:

```text
common_speculative_impl_draft_dflash: adding speculative implementation 'draft-dspark'
common_speculative_impl_draft_dflash: - n_max=7, n_min=0, p_min=0.00
common_speculative_impl_draft_dflash: - block_size=7, mask_token_id=75982, n_extract=5, sample_from_anchor=true
```

The startup log also shows this error line. It is expected and you can ignore it: `dflash requires ctx_other to be set (this warning is normal during memory fitting)`.

| Flag | Purpose |
| --- | --- |
| `-m …F16.gguf` | The target model, unquantized. |
| `-md …DSpark.gguf` | The draft model. |
| `--spec-type draft-dspark` | Use the DSpark algorithm. |
| `--spec-draft-n-max 7` | Draft the full 7-token block. The default is 3, which discards most of each block. Values above 7 are clamped to 7. |
| `-ngl 99 -ngld 99` | Put all layers of the target and the draft on the GPU (Metal). |
| `-fa on` | Use flash attention. |
| `--temp 1.0 --top-p 0.95 --min-p 0.0` | MiniCPM's recommended sampling, used when a request does not set these values. The llama.cpp default `min_p` of 0.05 can cause repetition loops with this model. |
| `-c 8192` | Context size. The model supports up to 131072. |
| `--jinja` | Use the chat template in the GGUF. Thinking mode and tool calls need it. |
| `-a MiniCPM5-2B` | The model name that the API reports. |
| `--host 127.0.0.1 --port 8090` | Listen on localhost only. Port 8080 is taken by Tilt on this Mac. |

The server uses about 6 GB of RAM.

## 4. Use it

```bash
curl -s http://127.0.0.1:8090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "1+1=?"}], "max_tokens": 512}' \
  | jq '{answer: .choices[0].message.content, thinking: .choices[0].message.reasoning_content, timings}'
```

- The thinking goes to `reasoning_content` and the answer goes to `content`.
- To turn off thinking for one request, add `"chat_template_kwargs": {"enable_thinking": false}` to the request body.
- For OpenAI SDKs, use `base_url="http://127.0.0.1:8090/v1"` and any API key.

To chat in the terminal without a server, give `llama-cli` the same flags:

```bash
llama-cli \
  -m "$LLAMA_CACHE/MiniCPM5-2B-F16.gguf" \
  -md "$LLAMA_CACHE/MiniCPM5-2.6B-DSpark.gguf" \
  --spec-type draft-dspark --spec-draft-n-max 7 \
  -ngl 99 -ngld 99 -fa on -c 8192 \
  --temp 1.0 --top-p 0.95 --min-p 0.0
```

## 5. Check that DSpark is working

For each request, the server log prints a line like this:

```text
draft acceptance = 0.59195 (  824 accepted /  1392 generated), mean len =  5.14
```

`mean len` is the number of tokens produced per target forward pass. Without a draft it is 1. With DSpark it can be at most 8: 7 drafted tokens plus 1 from the target. Each API response also has these counts in `timings.draft_n` and `timings.draft_n_accepted`.

## 6. Benchmark

[`bench.sh`](./bench.sh) sends 3 prompts at T=0 and at T=1.0 to a running server, one at a time. It prints tokens/s and draft acceptance for each:

```bash
./bench.sh 8090        # optional 2nd argument: max_tokens (default 1024)
```

For the baseline, stop the DSpark server, start one without the draft, and run the script again:

```bash
llama-server -m "$LLAMA_CACHE/MiniCPM5-2B-F16.gguf" \
  -ngl 99 -fa on -c 8192 --jinja -a MiniCPM5-2B --host 127.0.0.1 --port 8090
```

Results on the M3 Pro (F16 target, up to 1024 tokens, one run per row):

| Prompt | Temp | No draft (tok/s) | DSpark (tok/s) | Speedup | Mean len |
| --- | --- | --- | --- | --- | --- |
| math | 0 | 26.9 | 75.8 | 2.8× | 5.14 |
| code | 0 | 26.2 | 81.0 | 3.1× | 5.55 |
| general | 0 | 26.2 | 35.9 | 1.4× | 2.44 |
| math | 1.0 | 26.1 | 51.8 | 2.0× | 3.58 |
| code | 1.0 | 27.0 | 47.7 | 1.8× | 3.31 |
| general | 1.0 | 26.9 | 31.2 | 1.2× | 2.15 |

- Math and code get the largest speedup because the draft predicts them well. Free-form prose gets the smallest.
- Greedy decoding (T=0) gets more speedup than sampling (T=1.0).
- The T=1.0 rows are single samples, so the numbers vary between runs.

## Notes

- The draft works only with MiniCPM5-2B, because it reads that model's hidden states and uses its tokenizer. The draft's model card pairs it with the Q4_K_M file.
- If you make the server reachable from other machines, add `--api-key <key>`.
- To stop a server that runs in the background: `pkill -f 'llama(-server| serve).*--port 8090'`. The pattern matches both `llama-server` and `llama serve`, which run the same server.

## Sources

- [openbmb/MiniCPM5-2B-GGUF](https://huggingface.co/openbmb/MiniCPM5-2B-GGUF)
- [openbmb/MiniCPM5-2B-DSpark-GGUF](https://huggingface.co/openbmb/MiniCPM5-2B-DSpark-GGUF)
- [openbmb/MiniCPM5-2B-DSpark](https://huggingface.co/openbmb/MiniCPM5-2B-DSpark): draft specification and reference acceptance lengths
- [MiniCPM llama.cpp deployment skill](https://github.com/OpenBMB/MiniCPM/blob/main/skills/minicpm5-deploy-llama-cpp/SKILL.md)
- [llama.cpp speculative decoding docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)
