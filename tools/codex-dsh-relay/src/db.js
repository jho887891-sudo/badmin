import { mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { DatabaseSync } from 'node:sqlite';

export const ACTIVE_STATUSES = ['created', 'running', 'deepseek_completed', 'waiting_codex', 'queued_to_codex', 'codex_processing'];

export class RelayStore {
  constructor(path = process.env.CODEX_DSH_RELAY_DB || resolve('relay/jobs.sqlite')) {
    mkdirSync(dirname(path), { recursive: true });
    this.path = path;
    this.db = new DatabaseSync(path);
    this.closed = false;
    this.db.exec('PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;');
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS jobs (
        job_id TEXT PRIMARY KEY,
        codex_thread_id TEXT NOT NULL,
        deepseek_session_id TEXT NOT NULL,
        goal TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        status TEXT NOT NULL,
        last_event_seq INTEGER NOT NULL DEFAULT -1,
        deepseek_result TEXT,
        relevant_events TEXT,
        codex_delivery_status TEXT NOT NULL DEFAULT 'pending',
        delivery_attempts INTEGER NOT NULL DEFAULT 0,
        next_delivery_at TEXT,
        error TEXT
      );
      CREATE INDEX IF NOT EXISTS jobs_status_idx ON jobs(status, updated_at);
      CREATE INDEX IF NOT EXISTS jobs_session_idx ON jobs(deepseek_session_id, updated_at);
    `);
  }

  create(job) {
    const now = new Date().toISOString();
    this.db.prepare(`INSERT INTO jobs (
      job_id, codex_thread_id, deepseek_session_id, goal, created_at, updated_at, status, last_event_seq
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)`)
      .run(job.jobId, job.codexThreadId, job.deepseekSessionId, job.goal, now, now, job.status || 'created', job.lastEventSeq ?? -1);
    return this.get(job.jobId);
  }

  get(jobId) {
    return mapRow(this.db.prepare('SELECT * FROM jobs WHERE job_id = ?').get(jobId));
  }

  list() {
    return this.db.prepare('SELECT * FROM jobs ORDER BY created_at DESC').all().map(mapRow);
  }

  recoverable() {
    const marks = ACTIVE_STATUSES.map(() => '?').join(',');
    return this.db.prepare(`SELECT * FROM jobs WHERE status IN (${marks}) ORDER BY created_at`).all(...ACTIVE_STATUSES).map(mapRow);
  }

  update(jobId, patch) {
    const fields = {
      status: 'status', lastEventSeq: 'last_event_seq', deepseekResult: 'deepseek_result',
      relevantEvents: 'relevant_events', codexDeliveryStatus: 'codex_delivery_status',
      deliveryAttempts: 'delivery_attempts', nextDeliveryAt: 'next_delivery_at', error: 'error',
    };
    const pairs = Object.entries(patch).filter(([key]) => fields[key]);
    if (!pairs.length) return this.get(jobId);
    const sql = pairs.map(([key]) => `${fields[key]} = ?`).join(', ');
    const values = pairs.map(([, value]) => typeof value === 'object' && value !== null ? JSON.stringify(value) : value);
    this.db.prepare(`UPDATE jobs SET ${sql}, updated_at = ? WHERE job_id = ?`).run(...values, new Date().toISOString(), jobId);
    return this.get(jobId);
  }

  close() {
    if (!this.closed) this.db.close();
    this.closed = true;
  }
}

function mapRow(row) {
  if (!row) return null;
  return {
    jobId: row.job_id,
    codexThreadId: row.codex_thread_id,
    deepseekSessionId: row.deepseek_session_id,
    goal: row.goal,
    createdAt: row.created_at,
    updatedAt: row.updated_at,
    status: row.status,
    lastEventSeq: Number(row.last_event_seq),
    deepseekResult: row.deepseek_result,
    relevantEvents: parseJson(row.relevant_events),
    codexDeliveryStatus: row.codex_delivery_status,
    deliveryAttempts: Number(row.delivery_attempts),
    nextDeliveryAt: row.next_delivery_at,
    error: row.error,
  };
}

function parseJson(value) {
  if (!value) return null;
  try { return JSON.parse(value); } catch { return value; }
}
