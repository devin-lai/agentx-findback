# Troubleshooting

`scripts/findback.py` prints a JSON error document on stdout (`error.exit_code`,
`error.message`, `error.http_status` for HTTP failures, sometimes `partial`) and a hint on
stderr. It never prints `FINDBACK_TOKEN`; a token found in any message is replaced by
`[REDACTED]`.

| Exit code | Meaning |
| --- | --- |
| 0 | Success. A "no match" answer is still a success; read `status`. |
| 2 | Usage error: a missing or invalid argument. Nothing was sent, or the script refused locally. |
| 3 | Server unreachable, HTTP error, invalid response, or a run that failed or was cancelled. |
| 4 | Timeout: one request exceeded `--request-timeout`, or `--wait` exceeded `--timeout`. |

## The server is unreachable

`Cannot reach FindBack at http://...: Connection refused`

- Check `FINDBACK_URL` (scheme, host, port; no `/api` suffix needed). Default is
  `http://127.0.0.1:9000`.
- Start the server from the FindBack repository: `uv run --no-sync agentx serve` (loopback port
  9000), then run `health`.
- For a server on another machine, prefer an SSH tunnel to a loopback port, for example
  `ssh -N -L 127.0.0.1:19000:127.0.0.1:9000 <host>` and `FINDBACK_URL=http://127.0.0.1:19000`.
- HTTPS certificates must validate; the script has no switch to skip verification.
- A redirect is refused on purpose, so the token is never forwarded; use the server's own URL.

## HTTP 401 or 503 on /api/v1

- **401** "Sign in with the configured access token.": the server has `AGENTX_API_TOKEN`. Set
  `FINDBACK_TOKEN` to the same value in the environment of the agent process (a shell profile,
  a secret manager or the harness's environment settings). Do not paste it into the chat, a
  command line or a file inside this skill. `health` reports `auth_required` and
  `token_configured` without revealing either value.
- **503** "Configure AGENTX_API_TOKEN before network access.": the server has no token and
  only accepts loopback clients. Connect over loopback (or a tunnel), or have the operator set
  a token.

## HTTP 403 with a token

"The agent token can read memory and ask questions...": the server gave this agent a
read-and-ask token (`AGENTX_AGENT_TOKEN`). Listing, asking, frames and reports work; uploads,
registrations, indexing, reviews and deletions need the operator. Tell the user what you would
have done and why; do not retry or look for another token.

## No perception backend

`health` lists `index_backends` and `suggest_backends`.

- `reference` (OpenCV template matching) is always available and is the default. It suits
  distinct textures, a fixed camera and stable light.
- `rtdetr` needs the server's `vision` extra and its pinned weights (`rtdetr_installed`).
- `cosmos` needs a configured Cosmos-Reason2 service or checkpoint; `sam2` needs an installed
  SAM 2.1 checkpoint. Indexing with one that is not configured fails with 422 and the reason.
- `suggest` returns 503 when its backend is unavailable and 502 when the backend failed; nothing
  is registered either way. RT-DETR proposes only its 80 COCO categories, so a custom object
  may get no proposal: ask the user to point it out instead.

## Planner fallback and citation warnings

- "The planner model was unavailable or returned an invalid plan; the local query workflow
  answered instead." The answer is still rendered from verified memory. Report the warning.
- "The planner agent failed with an internal error; ..." The same fallback, after a server
  defect. Report it.
- "The agent cited evidence it never retrieved or that is unsupported for the selected object;
  it was removed." The citation check in code worked; report the warning with the answer.
- "The agent answered about a different object; the selected object's memory is shown." The
  planner's prose was discarded in favor of the object the request resolved to.
- `planner: local` when the user asked something indirect ("the item that disappeared") means
  no planner is configured. Ask for the object's registered name, or use `--object-id`.
- `--no-planner` forces the deterministic resolver for a comparison.

## Indexing failures

- 422 "An index run is already queued or running for this video.": wait for it with
  `run <run_id> --wait`, or use `index --reuse --wait`, which attaches to it.
- 422 "Register at least one object before building memory.": register first.
- A run with `"status": "failed"` carries `error`. A server restart fails the run that was
  executing; building again creates a new run and keeps the failed one for diagnosis.
- Exit 4 from `--wait`: the run continues on the server. Resume with `run <run_id> --wait`.
- 422 "Wait for the current index run before changing registration.": registration and region
  changes are blocked while a run is active.

## Question errors

- 422 "That time has not been processed.": the cutoff is beyond the indexed range. Use a
  smaller `--at-ms`, or `--at-end`.
- 422 "This run has no queryable memory.": the run is not complete. Wait for it or pick another.
- 404 "Analysis run not found." or "Selected object does not belong to this run.": check the IDs
  with `videos`; objects registered after a run are not part of it.
- `status: unknown` with `evidence_missing` or `evidence_changed`: the original recording was
  deleted or modified on the server. No visual claim is possible until it is restored.

## Frames and reports

- 422 "This observation has no visible object evidence.": only visible observations have frames.
- 404 "Original evidence is missing.": the source video is gone from the server.
- `report` 429: another report is being built; retry after `Retry-After`. 413: the report
  exceeds the server's limit. The script will not overwrite an existing file without `--force`.

## The skill itself fails verification

If `model_signing verify` or `scripts/sign_skills.py verify` reports a mismatch or extra files,
the installed directory is not the signed release. Remove stray files such as `__pycache__` or
`.DS_Store`, or reinstall the skill; do not use a modified copy.
