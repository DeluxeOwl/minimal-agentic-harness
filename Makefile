.PHONY: run format check serve-llm

# MiniCPM5-2B with its DSpark draft model, served by llama.cpp.
MINICPM_DOCS := docs/minicpm5-dspark-llama-cpp/README.md
MINICPM_TARGET := MiniCPM5-2B-F16.gguf
MINICPM_DRAFT := MiniCPM5-2.6B-DSpark.gguf

run:
	uv run --locked src/cmd/main.py

# ruff format does not sort imports. Import sorting is the I001 lint rule.
format:
	uv run --locked ruff check --select I --fix .
	uv run --locked ruff format .

check: format
	uv run --locked ruff check .
	uv run --locked mypy

# Check steps 1 and 2 of the MiniCPM docs, then run the server command from them.
serve-llm:
	@command -v llama-server >/dev/null || \
		{ echo "llama-server not found. Install llama.cpp: brew install llama.cpp" >&2; exit 1; }
	@llama-server --help 2>/dev/null | grep -q draft-dspark || \
		{ echo "llama-server has no DSpark support. Upgrade: brew upgrade llama.cpp" >&2; exit 1; }
	@test -n "$$LLAMA_CACHE" || \
		{ echo "LLAMA_CACHE is not set. Set it to the directory that has the models. See $(MINICPM_DOCS)." >&2; exit 1; }
	@for model in $(MINICPM_TARGET) $(MINICPM_DRAFT); do \
		test -f "$$LLAMA_CACHE/$$model" || \
			{ echo "$$LLAMA_CACHE/$$model not found. See step 2 of $(MINICPM_DOCS)." >&2; exit 1; }; \
	done
	llama-server \
		-m "$$LLAMA_CACHE/$(MINICPM_TARGET)" \
		-md "$$LLAMA_CACHE/$(MINICPM_DRAFT)" \
		--spec-type draft-dspark --spec-draft-n-max 7 \
		-ngl 99 -ngld 99 -fa on \
		--temp 1.0 --top-p 0.95 --min-p 0.0 \
		-c 131000 --jinja -a MiniCPM5-2B \
		--host 127.0.0.1 --port 8090
