# OpenClaw integration

The portable [FindBack Skill](../../skills/findback-video-memory/SKILL.md) ran on
OpenClaw with Spark-hosted Nemotron and hosted StepFun outer agents.
See the [competition results](../../docs/benchmarks/INDEX.md) for the limits of agent integrations. The FindBack server's own planner is configured
separately; switching OpenClaw's model does not switch that planner.

## Safe experiment setup

1. Run the FindBack app and local Nemotron endpoint on the assigned Spark as in
   [deployment](../../docs/DEPLOYMENT.md). Install OpenClaw in a project-local directory. Keep configuration and sandbox build receipts private.
2. Give OpenClaw a fresh workspace per independent question. Copy the signed
   `skills/findback-video-memory/` directory into its `skills/` subdirectory and
   keep the directory intact. The Skill saves answer evidence outside its own
   directory; do not let an agent edit signed files.
3. Use OpenClaw's Docker sandbox for `read` and `exec` tools. Mount no app `.env`,
   API key or token into the sandbox. The evaluated image was a minimal ARM64
   Python image with `/bin/sleep`, a read-only root and dropped capabilities.
4. Start the [short-lived host proxy](read_only_proxy.py) with the app's mode-0600
   `.env`. On Spark's default Docker bridge, the evaluated command shape was:

   ```bash
   python integrations/openclaw/read_only_proxy.py --env-file .env \
     --listen-host 172.17.0.1 --listen-port 19001 \
     --upstream http://127.0.0.1:9000
   ```

   The sandbox saw `FINDBACK_URL=http://172.17.0.1:19001` and no
   `FINDBACK_TOKEN`. The proxy allows API reads and questions for existing runs;
   other POST paths are denied. Keep proxy test receipts privately. This is not a multi-tenant authentication service: other local
   containers on that bridge may reach it. Shut it down when the run ends.
5. Keep provider credentials in the host process environment. StepFun's API key
   must be available to OpenClaw's model provider, not to its tool container.

The evaluated OpenClaw configs set `sandbox.mode=all`, `backend=docker`,
`scope=session`, an isolated workspace, and `tools.allow=[read,exec]`. They used
OpenClaw's `agent exec --json` path. In JSON output, retrieve evidence from
`payloads[*].mediaUrl`; the `final` text has the `MEDIA:` line removed.

For explicit live-location requests without a recording, wrap the call with
the narrow [host guard](run_guarded.py):

```bash
python integrations/openclaw/run_guarded.py --message-file request.txt -- \
  openclaw agent exec --config openclaw.json --message-file request.txt --json
```

The guard returns an OpenClaw-shaped JSON refusal before launching the agent
for that one request class. Recording-relative questions are passed through.
This guard does not correct other agent errors or replace the server's source
validation. Use the sandbox and keep credentials outside tool execution.

OpenClaw references: [agent exec](https://docs.openclaw.ai/cli/agent),
[sandboxing](https://docs.openclaw.ai/gateway/sandboxing),
[Docker backend](https://docs.openclaw.ai/gateway/sandboxing/docker-backend),
[vLLM provider](https://docs.openclaw.ai/providers/vllm),
[StepFun provider](https://docs.openclaw.ai/providers/stepfun).
