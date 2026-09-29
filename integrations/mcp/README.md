# FindBack MCP server

`findback_mcp.py` exposes a running FindBack server to any Model Context Protocol client —
Claude Desktop, Claude Code, Cursor and others — as three tools: `list_recordings`,
`ask_memory` (the answer from verified memory plus the cited evidence frame as an image) and
`get_evidence_frame`. It uses only the Python standard library and reuses the portable Skill's
HTTP client, so token handling is the same in both.

`ask_memory` uses a recording's newest complete memory. For a live camera capture that is still
recording it asks the run following the camera, and `at_end` then means the newest moment memory
has processed; those answers carry the server's live-recording warning.

MCP is the connection; the [`findback-video-memory` Skill](../../skills/findback-video-memory/SKILL.md)
is the knowledge — when to use FindBack, what to ask first, and what an answer may and may not
claim. Install both for the best behaviour.

```json
{"mcpServers": {"findback": {
  "command": "python3",
  "args": ["/path/to/AgentX/integrations/mcp/findback_mcp.py"],
  "env": {"FINDBACK_URL": "http://127.0.0.1:9000"}}}}
```

Set `FINDBACK_TOKEN` in that `env` block when the server requires one; it is never returned to
the client, and error text is redacted. The server's read-and-ask token (`AGENTX_AGENT_TOKEN`) is
enough for all three tools and cannot change the server's memory. Verified with Claude Code as the client:

```bash
claude -p "Using the findback MCP tools, where was the Red toolkit last seen as of 00:07 in the \
'Controlled desk fixture' recording?" --mcp-config mcp.json --strict-mcp-config \
  --allowedTools "mcp__findback__ask_memory,mcp__findback__list_recordings"
# → As of 00:07, the Red toolkit was last seen at 00:05.8 in the Right area
#   (its position at 00:07 itself is unconfirmed).
```

Protocol tests: `tests/test_mcp_server.py`.
