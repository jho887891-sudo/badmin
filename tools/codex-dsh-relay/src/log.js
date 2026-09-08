import { appendFileSync, mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';

export class RelayLogger {
  constructor(path = process.env.CODEX_DSH_RELAY_LOG || resolve('logs/relay.log')) {
    this.path = path;
    mkdirSync(dirname(path), { recursive: true });
  }

  write({ jobId = null, codexThreadId = null, deepseekSessionId = null, event, status, detail = null }) {
    const safeDetail = redact(detail);
    appendFileSync(this.path, `${JSON.stringify({
      timestamp: new Date().toISOString(), job_id: jobId, codex_thread_id: codexThreadId,
      deepseek_session_id: deepseekSessionId, event, status, ...(safeDetail ? { detail: safeDetail } : {}),
    })}\n`, 'utf8');
  }
}

function redact(value) {
  if (value === null || value === undefined) return null;
  return String(value)
    .replace(/(?:sk-[A-Za-z0-9_-]{12,}|Bearer\s+[A-Za-z0-9._-]+)/gi, '[REDACTED]')
    .replace(/-----BEGIN [^-]*PRIVATE KEY-----[\s\S]*?-----END [^-]*PRIVATE KEY-----/g, '[REDACTED PRIVATE KEY]');
}
