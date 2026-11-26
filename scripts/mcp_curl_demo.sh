#!/usr/bin/env bash
# Raw-HTTP walkthrough of the AnimeText Studio MCP server, for programs that do not use the
# MCP SDK. This is what the streamable-http transport actually speaks: JSON-RPC 2.0 POSTs to
# /mcp, an `Mcp-Session-Id` header echoed back on every request, and
# `Accept: application/json, text/event-stream`.
#
#   ./scripts/mcp_curl_demo.sh [PORT]        # default 8765
#
# It starts nothing. Bring the server up first, e.g.
#   python -m animetext_studio --mock --headless --serve --mcp-port 8765
set -uo pipefail

PORT="${1:-8765}"
URL="http://127.0.0.1:${PORT}/mcp"
IMAGE="${IMAGE:-examples/sample_page.png}"
OUT="${OUT:-out/annotated_from_curl.png}"
HDR=(-H 'Content-Type: application/json' -H 'Accept: application/json, text/event-stream')

say() { printf '\n\033[1m$ %s\033[0m\n' "$1"; }

# --- 0. health --------------------------------------------------------------------------
say "curl -s http://127.0.0.1:${PORT}/health"
curl -s "http://127.0.0.1:${PORT}/health"; echo

# --- 1. initialize: the session id comes back in a header, not in the body ---------------
say "curl -si -X POST $URL -d '{...initialize...}'"
INIT=$(curl -si "${HDR[@]}" -X POST "$URL" -d '{
  "jsonrpc": "2.0", "id": 1, "method": "initialize",
  "params": {
    "protocolVersion": "2025-06-18",
    "capabilities": {},
    "clientInfo": {"name": "curl", "version": "1"}
  }}')
echo "$INIT" | grep -iE '^(HTTP|mcp-session-id)'
SESSION=$(printf '%s' "$INIT" | tr -d '\r' | awk -F': ' 'tolower($1)=="mcp-session-id"{print $2}')
if [ -z "${SESSION}" ]; then echo "no session id - is the server up on port ${PORT}?"; exit 1; fi
SID=(-H "Mcp-Session-Id: ${SESSION}")

# --- 2. the handshake is not finished until the client says it is initialized ------------
say "curl ... -d '{\"method\":\"notifications/initialized\"}'"
curl -s -o /dev/null -w '%{http_code}\n' "${HDR[@]}" "${SID[@]}" -X POST "$URL" \
  -d '{"jsonrpc":"2.0","method":"notifications/initialized"}'

# --- 3. what can this server do? (19 tools, 6 resources) ---------------------------------
say "curl ... -d '{\"method\":\"tools/list\"}' | jq -r '.result.tools[].name'"
curl -s "${HDR[@]}" "${SID[@]}" -X POST "$URL" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' \
  | jq -r '.result.tools[].name' | paste -sd' ' -

say "curl ... -d '{\"method\":\"resources/list\"}' | jq -r '.result.resources[].uri'"
curl -s "${HDR[@]}" "${SID[@]}" -X POST "$URL" \
  -d '{"jsonrpc":"2.0","id":3,"method":"resources/list"}' \
  | jq -r '.result.resources[].uri' | paste -sd' ' -

# --- 4. a call that changes nothing ------------------------------------------------------
say "curl ... tools/call get_status"
curl -s "${HDR[@]}" "${SID[@]}" -X POST "$URL" \
  -d '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"get_status","arguments":{}}}' \
  | jq -c '.result | {isError, app: .structuredContent.app, version: .structuredContent.version}'

# --- 5. load a page, then run the pipeline (the slow one) --------------------------------
say "curl ... tools/call load_image {\"path\":\"${IMAGE}\"}"
curl -s "${HDR[@]}" "${SID[@]}" -X POST "$URL" \
  -d "{\"jsonrpc\":\"2.0\",\"id\":5,\"method\":\"tools/call\",
       \"params\":{\"name\":\"load_image\",\"arguments\":{\"path\":\"${IMAGE}\"}}}" \
  | jq -c '.result.structuredContent'

say "curl ... tools/call run_pipeline {}"
curl -s "${HDR[@]}" "${SID[@]}" -X POST "$URL" \
  -d '{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"run_pipeline","arguments":{}}}' \
  | jq -c '.result.structuredContent | {block_count, timing_ms: .result.timing_ms}'

# --- 6. the answer, and a picture of it --------------------------------------------------
say "curl ... tools/call get_result {\"block_ids\":[1,2]}"
curl -s "${HDR[@]}" "${SID[@]}" -X POST "$URL" \
  -d '{"jsonrpc":"2.0","id":7,"method":"tools/call",
       "params":{"name":"get_result","arguments":{"block_ids":[1,2]}}}' \
  | jq -c '.result.structuredContent.result.blocks | map({id, bbox, text})'

say "curl ... tools/call get_annotated_image  (pixels arrive as image content, size in structuredContent)"
curl -s "${HDR[@]}" "${SID[@]}" -X POST "$URL" \
  -d '{"jsonrpc":"2.0","id":8,"method":"tools/call","params":{"name":"get_annotated_image","arguments":{}}}' \
  | jq -c '.result | {meta: .structuredContent, content_type: .content[0].type,
                      mime: .content[0].mimeType, chars: (.content[0].data | length)}'

# --- 7. an error is a result, not a broken transport -------------------------------------
say "curl ... tools/call set_config {\"patch\":{\"detection\":{\"model\":\"nope\"}}}"
curl -s "${HDR[@]}" "${SID[@]}" -X POST "$URL" \
  -d '{"jsonrpc":"2.0","id":9,"method":"tools/call",
       "params":{"name":"set_config","arguments":{"patch":{"detection":{"model":"nope"}}}}}' \
  | jq -c '.result | {isError, error: .structuredContent}'

# --- 8. resources are cheap reads of the same state --------------------------------------
say "curl ... resources/read log://mcp"
curl -s "${HDR[@]}" "${SID[@]}" -X POST "$URL" \
  -d '{"jsonrpc":"2.0","id":10,"method":"resources/read","params":{"uri":"log://mcp"}}' \
  | jq -c '.result.contents[0].text | fromjson | {calls, errors, last: .entries[0].tool}'

printf '\nDone. Session %s against %s\n' "${SESSION}" "${URL}"
