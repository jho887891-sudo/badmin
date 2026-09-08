import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { RelayStore } from '../src/db.js';
import { RelayService } from '../src/relay.js';

function fixture() {
  const dir = mkdtempSync(join(tmpdir(), 'codex-dsh-relay-'));
  const store = new RelayStore(join(dir, 'jobs.sqlite'));
  const logs = [];
  const logger = { write: (row) => logs.push(row) };
  return { dir, store, logs, close() { store.close(); try { rmSync(dir, { recursive: true, force: true, maxRetries: 5, retryDelay: 50 }); } catch {} } };
}

test('recovers a running job after database reopen', () => {
  const f = fixture();
  const created = f.store.create({ jobId: 'job-1', codexThreadId: 'thread-1', deepseekSessionId: 'session-1', goal: 'goal', status: 'running' });
  assert.equal(created.status, 'running');
  const path = f.store.path;
  f.store.close();
  f.store = new RelayStore(path);
  assert.equal(f.store.recoverable()[0].deepseekSessionId, 'session-1');
  f.close();
});

test('busy Codex thread remains pending without delivery', async () => {
  const f = fixture();
  const job = f.store.create({ jobId: 'job-2', codexThreadId: 'thread-busy', deepseekSessionId: 'session-2', goal: 'goal', status: 'deepseek_completed' });
  let deliveries = 0;
  const service = new RelayService({
    store: f.store,
    logger: { write() {} },
    harness: {},
    delivery: { isBusy: () => true, deliver: async () => { deliveries++; return { delivered: true }; } },
    pollMs: 10,
  });
  await service.processJob(job);
  assert.equal(deliveries, 0);
  assert.equal(f.store.get(job.jobId).status, 'waiting_codex');
  f.close();
});

test('completed durable turn is delivered to the original Codex thread', async () => {
  const f = fixture();
  const job = f.store.create({ jobId: 'job-3', codexThreadId: 'thread-original', deepseekSessionId: 'session-original', goal: 'goal', status: 'running', lastEventSeq: 1 });
  const delivered = [];
  const events = [
    { seq: 2, type: 'assistant/message', data: { message: { content: [{ type: 'text', text: 'done' }] } } },
    { seq: 3, type: 'turn/end', data: { reason: { kind: 'completed' } } },
  ];
  const service = new RelayService({
    store: f.store,
    logger: { write() {} },
    harness: {
      readSession: async () => ({ events }),
      assertSession: async (sessionId) => ({ sessionId, running: false }),
    },
    delivery: {
      isBusy: () => false,
      deliver: async (threadId, message) => { delivered.push({ threadId, message }); return { delivered: true, method: 'fixture' }; },
    },
  });
  await service.processJob(job);
  assert.equal(delivered[0].threadId, 'thread-original');
  assert.match(delivered[0].message, /\[DEEPSEEK_HARNESS_COMPLETION\]/);
  assert.equal(f.store.get(job.jobId).status, 'completed');
  f.close();
});
