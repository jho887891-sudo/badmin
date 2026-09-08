# Codex - DeepSeek Harness Relay

Windows-compatible Session bridge for the existing Harness at `http://127.0.0.1:3080`.
It does not start Harness, parse its HTML, control its UI, or expose shell/system operations.

## Run

```powershell
cd tools\codex-dsh-relay
.\relay.cmd sessions --cwd "E:\具身智能\badmin_project" --non-blank
.\relay.cmd attach <deepseek_session_id>
.\relay.cmd send <deepseek_session_id> "message"
.\relay.cmd follow <deepseek_session_id>
.\relay.cmd delegate --session <deepseek_session_id> --message "task"
.\relay.cmd daemon
```

`delegate` requires `CODEX_THREAD_ID`. It refuses to infer a thread from recency.
Jobs persist in `relay/jobs.sqlite`; structured, redacted logs are written to `logs/relay.log`.
The daemon resumes recoverable jobs after restart and defers Codex delivery while the target thread is busy.

Codex delivery defaults to `codex exec resume --json <thread_id> <message>`. To use another supported launcher in tests or deployments, set `CODEX_RELAY_DELIVERY_COMMAND` with `{thread_id}` and `{message}` placeholders.

The MCP server is registered as `deepseek_session_bridge` in the local Codex configuration. Reload Codex or start a new task after installation so Codex discovers the six `deepseek_*` tools.

## API behavior

The adapter probes the running Gateway. The installed Harness currently uses RPC envelopes at `/api/session.list`, `/api/session.history`, `/api/session.prompt`, and `/api/session.cancel`. A legacy slash/args carrier is retained as a runtime fallback.
