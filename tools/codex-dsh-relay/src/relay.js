import { randomUUID } from 'node:crypto';
import { compactEvent, HarnessClient, messageText } from './harness.js';
import { RelayStore } from './db.js';
import { RelayLogger } from './log.js';
import { CodexDelivery } from './delivery.js';

export class RelayService {
  constructor({ harness = new HarnessClient(), store = new RelayStore(), logger = new RelayLogger(), delivery = new CodexDelivery(), pollMs = Number(process.env.CODEX_DSH_RELAY_POLL_MS || 1500) } = {}) {
    this.harness = harness;
    this.store = store;
    this.logger = logger;
    this.delivery = delivery;
    this.pollMs = pollMs;
  }

  async delegate({ sessionId, message, codexThreadId = process.env.CODEX_THREAD_ID }) {
    if (!codexThreadId) throw new Error('CODEX_THREAD_ID is required; refusing to guess a Codex thread');
    const session = await this.harness.assertSession(sessionId);
    if (session.blank) throw new Error('Cannot delegate to a blank DeepSeek session');
    const history = await this.harness.readSession(sessionId, { maxMessages: 3 });
    const lastSeq = history.events.at(-1)?.seq ?? -1;
    const job = this.store.create({
      jobId: `job-${randomUUID()}`,
      codexThreadId,
      deepseekSessionId: sessionId,
      goal: message,
      status: 'created',
      lastEventSeq: lastSeq,
    });
    this.log(job, 'job-created', 'created');
    try {
      const sent = await this.harness.sendMessage(sessionId, message);
      if (!sent.accepted) throw new Error('Harness did not accept prompt');
      const updated = this.store.update(job.jobId, { status: 'running', codexDeliveryStatus: 'pending' });
      this.log(updated, 'deepseek-prompt-accepted', 'running', sent.requestId);
      return { ...updated, requestId: sent.requestId };
    } catch (error) {
      this.store.update(job.jobId, { status: 'failed', error: error.message });
      throw error;
    }
  }

  async processOnce() {
    const jobs = this.store.recoverable();
    for (const job of jobs) await this.processJob(job);
    return jobs.length;
  }

  async processJob(job) {
    try {
      if (['created', 'running'].includes(job.status)) await this.observeDeepSeek(job);
      const current = this.store.get(job.jobId);
      if (['deepseek_completed', 'waiting_codex', 'queued_to_codex'].includes(current.status)) await this.deliverToCodex(current);
    } catch (error) {
      const current = this.store.update(job.jobId, { status: 'failed', error: error.message });
      this.log(current, 'job-error', 'failed', error.message);
    }
  }

  async observeDeepSeek(job) {
    const snapshot = await this.harness.readSession(job.deepseekSessionId, { maxMessages: 3 });
    const events = snapshot.events.filter((event) => event.seq > job.lastEventSeq).sort((a, b) => a.seq - b.seq);
    if (!events.length) return;
    const lastEventSeq = events.at(-1).seq;
    const assistant = [...events].reverse().find((event) => event.type === 'assistant/message' && messageText(event));
    const turnEnd = [...events].reverse().find((event) => event.type === 'turn/end');
    const session = await this.harness.assertSession(job.deepseekSessionId);
    const relevant = events.filter((event) => ['assistant/message', 'tool/call', 'tool/result', 'turn/end'].includes(event.type)).map(compactEvent);
    const patch = { lastEventSeq, relevantEvents: relevant.slice(-30) };
    if (assistant) patch.deepseekResult = messageText(assistant);
    if (turnEnd && completedReason(turnEnd) && !session.running) {
      patch.status = 'deepseek_completed';
      patch.codexDeliveryStatus = 'pending';
      this.log(job, 'deepseek-turn-completed', 'deepseek_completed', `seq=${turnEnd.seq}`);
    } else if (turnEnd && failedReason(turnEnd) && !session.running) {
      patch.status = 'failed';
      patch.error = `DeepSeek turn ended: ${turnEnd.data?.reason?.kind || 'failed'}`;
    }
    this.store.update(job.jobId, patch);
  }

  async deliverToCodex(job) {
    if (job.nextDeliveryAt && Date.parse(job.nextDeliveryAt) > Date.now()) return;
    if (this.delivery.isBusy(job.codexThreadId)) {
      const queued = this.store.update(job.jobId, { status: 'waiting_codex', codexDeliveryStatus: 'busy', nextDeliveryAt: new Date(Date.now() + this.pollMs).toISOString() });
      this.log(queued, 'codex-busy', 'waiting_codex');
      return;
    }
    this.store.update(job.jobId, { status: 'codex_processing', codexDeliveryStatus: 'delivering', deliveryAttempts: job.deliveryAttempts + 1 });
    const result = await this.delivery.deliver(job.codexThreadId, completionMessage(job));
    if (result.delivered) {
      const done = this.store.update(job.jobId, { status: 'completed', codexDeliveryStatus: 'delivered', error: null, nextDeliveryAt: null });
      this.log(done, 'codex-delivered', 'completed', result.method);
    } else {
      const attempts = job.deliveryAttempts + 1;
      const retryMs = Math.min(60_000, 1000 * 2 ** Math.min(attempts, 6));
      const pending = this.store.update(job.jobId, {
        status: result.busy ? 'waiting_codex' : 'queued_to_codex',
        codexDeliveryStatus: result.busy ? 'busy' : 'retry',
        deliveryAttempts: attempts,
        nextDeliveryAt: new Date(Date.now() + retryMs).toISOString(),
        error: result.error || result.stderr || `delivery exit ${result.code}`,
      });
      this.log(pending, 'codex-delivery-deferred', pending.status, pending.error);
    }
  }

  async daemon({ signal } = {}) {
    this.logger.write({ event: 'daemon-start', status: 'running' });
    while (!signal?.aborted) {
      await this.processOnce();
      await new Promise((resolve) => setTimeout(resolve, this.pollMs));
    }
    this.logger.write({ event: 'daemon-stop', status: 'completed' });
  }

  log(job, event, status, detail = null) {
    this.logger.write({ jobId: job.jobId, codexThreadId: job.codexThreadId, deepseekSessionId: job.deepseekSessionId, event, status, detail });
  }
}

export function completionMessage(job) {
  const events = Array.isArray(job.relevantEvents) ? job.relevantEvents : [];
  const files = extractToolHints(events, /write|edit|patch/i);
  const tests = extractToolHints(events, /test|pytest|npm test|pnpm test/i);
  const errors = events.filter((event) => event.type === 'turn/end' && !completedReason(event));
  return `[DEEPSEEK_HARNESS_COMPLETION]

Job ID: ${job.jobId}
DeepSeek Session: ${job.deepseekSessionId}

Original Goal:
${job.goal}

DeepSeek Final Response:
${job.deepseekResult || '(no durable assistant text captured)'}

Relevant Events:
${JSON.stringify(events.slice(-10), null, 2)}

Files Changed:
${files || '(not reliably available)'}

Tests:
${tests || '(not reliably available)'}

Errors:
${errors.length ? JSON.stringify(errors, null, 2) : 'None reported by durable turn events.'}

请继续之前的代理任务。结合原 Codex Thread 上下文独立审查结果；必要时检查代码、日志和测试。若未解决，继续使用同一个 DeepSeek Session；若已解决，直接向用户总结。`;
}

function completedReason(event) {
  return event.data?.reason?.kind === 'completed' || event.data?.status === 'completed';
}

function failedReason(event) {
  const kind = event.data?.reason?.kind || event.data?.status;
  return ['failed', 'cancelled', 'canceled', 'error'].includes(kind);
}

function extractToolHints(events, pattern) {
  return events.filter((event) => pattern.test(JSON.stringify(event.data || {}))).map((event) => JSON.stringify(event.data)).slice(-5).join('\n');
}
