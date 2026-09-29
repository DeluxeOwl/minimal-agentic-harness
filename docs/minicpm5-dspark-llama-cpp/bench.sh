#!/usr/bin/env bash
# Measure decode speed of a running llama-server, with or without DSpark.
#
#   ./bench.sh [port] [max_tokens]
#
# Sends 3 prompts (math, code, general) at temperature 0 and 1.0, one at a time,
# and prints tokens/s plus draft acceptance (only present when a draft model is loaded).
# Needs: curl, jq.
set -euo pipefail

PORT="${1:-8090}"
MAX_TOKENS="${2:-1024}"
URL="http://127.0.0.1:${PORT}/v1/chat/completions"

PROMPTS=(
  "math|Find all real x such that x^3 - 6x^2 + 11x - 6 = 0. Show your work."
  "code|Write a Python function that returns the longest palindromic substring of a string. Include a short explanation."
  "general|Explain how a refrigerator keeps food cold, for a 12-year-old."
)

# Warm-up so the first measured request doesn't pay one-time costs.
curl -s "$URL" -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":8}' > /dev/null

printf '%-8s %5s %7s %8s %s\n' prompt temp tokens tok/s "draft accepted"
for temp in 0 1.0; do
  for entry in "${PROMPTS[@]}"; do
    name="${entry%%|*}"
    prompt="${entry#*|}"
    body=$(jq -n --arg p "$prompt" --argjson t "$temp" --argjson n "$MAX_TOKENS" '{
      messages: [{role: "user", content: $p}],
      temperature: $t, top_p: 0.95, min_p: 0.0, max_tokens: $n, seed: 42
    }')
    curl -s "$URL" -H 'Content-Type: application/json' -d "$body" |
      jq -r --arg name "$name" --arg t "$temp" '
        .timings as $x
        | ($x.predicted_per_second * 10 | round / 10) as $tps
        | (if ($x.draft_n // 0) > 0
             then "\($x.draft_n_accepted)/\($x.draft_n) (\($x.draft_n_accepted * 100 / $x.draft_n | round)%)"
             else "-" end) as $acc
        | "\($name)\t\($t)\t\($x.predicted_n)\t\($tps)\t\($acc)"' |
      awk -F'\t' '{printf "%-8s %5s %7s %8s %s\n", $1, $2, $3, $4, $5}'
  done
done
