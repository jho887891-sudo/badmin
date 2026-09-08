import { randomUUID } from 'node:crypto';

const DEFAULT_URL = 'http://127.0.0.1:3080';

export class HarnessError extends Error {
  constructor(message, details = {}) {
    super(message);
    this.name = 'HarnessError';
    this.details = details;
  }
}

export class HarnessClient {
  constructor({ baseUrl = process.env.DEEPSEEK_HARNESS_URL || DEFAULT_URL, fetchImpl = fetch } = {}) {
    this.baseUrl = baseUrl.replace(/\/$/, '');
    this.fetch = fetchImpl;
    this.protocol = null;
  }

  async health() {
    const response = await this.fetch(`${this.baseUrl}/`, { method: 'HEAD' });
    return { ok: response.ok, status: response.status };
  }

  async detectProtocol() {
    if (this.protocol) return this.protocol;
    const rpcId = randomUUID();
    const response = await this.fetch(`${this.baseUrl}/api/session.list`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ type: 'client-request', rpcId, method: 'session.list', payload: {} }),
    });
    if (response.ok) {
      const body = await response.json();
      if (body?.type === 'server-response' && body.rpcId === rpcId) {
        this.protocol = 'dot-rpc-envelope';
        return this.protocol;
      }
    }

    const fallback = await this.fetch(`${this.baseUrl}/api/session/list`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ args: {} }),
    });
    if (fallback.ok) {
      this.protocol = 'slash-args';
      return this.protocol;
    }
    throw new HarnessError('No supported Harness API Gateway protocol detected', {
      dotStatus: response.status,
      slashStatus: fallback.status,
    });
  }

  async call(method, args = {}) {
    const protocol = await this.detectProtocol();
    if (protocol === 'slash-args') {
      const response = await this.fetch(`${this.baseUrl}/api/${method.replace('.', '/')}`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ args }),
      });
      if (!response.ok) throw new HarnessError(`Harness HTTP ${response.status}`, { method });
      const body = await response.json();
      return unwrap(body, method);
    }

    const rpcId = randomUUID();
    const response = await this.fetch(`${this.baseUrl}/api/${method}`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ type: 'client-request', rpcId, method, payload: args }),
    });
    if (!response.ok) throw new HarnessError(`Harness HTTP ${response.status}`, { method, rpcId });
    const body = await response.json();
    if (body?.rpcId !== rpcId) throw new HarnessError('Harness rpcId mismatch', { method, rpcId, received: body?.rpcId });
    return { value: unwrap(body, method), requestId: rpcId };
  }

  async listSessions({ cursor, cwd, includeBlank = true } = {}) {
    const response = await this.call('session.list', cursor ? { cursor } : {});
    let items = response.value?.items || response.value || [];
    if (!includeBlank) items = items.filter((item) => !item.blank);
    if (cwd) items = items.filter((item) => normalizePath(item.cwd) === normalizePath(cwd));
    items.sort((a, b) => b.updatedAt - a.updatedAt);
    return { sessions: items, protocol: this.protocol };
  }

  async readSession(sessionId, { maxMessages = 10, beforeSeq } = {}) {
    const args = { sessionId, maxMessages };
    if (beforeSeq !== undefined) args.beforeSeq = beforeSeq;
    const response = await this.call('session.history', args);
    const value = response.value;
    const listed = await this.listSessions({ includeBlank: true });
    const metadata = listed.sessions.find((item) => item.sessionId === sessionId);
    if (!metadata) throw new HarnessError('Session not found after history read', { sessionId });
    return {
      sessionId,
      metadata,
      events: (value.events || []).map((entry) => entry.event || entry),
      hasMore: Boolean(value.hasMore),
      projections: value.projections,
    };
  }

  async sendMessage(sessionId, message, { mode = 'queue' } = {}) {
    const before = await this.assertSession(sessionId);
    const response = await this.call('session.prompt', {
      sessionId,
      mode,
      content: [{ type: 'text', text: message }],
      clientTimeZone: Intl.DateTimeFormat().resolvedOptions().timeZone,
    });
    const after = await this.assertSession(sessionId);
    if (before.sessionId !== after.sessionId) throw new HarnessError('Session ID changed while prompting', { before, after });
    return { sessionId, accepted: response.value?.accepted === true, requestId: response.requestId };
  }

  async stop(sessionId) {
    await this.assertSession(sessionId);
    const response = await this.call('session.cancel', { sessionId });
    return { sessionId, accepted: response.value?.accepted === true };
  }

  async getStatus(sessionId) {
    const session = await this.assertSession(sessionId);
    const history = await this.readSession(sessionId, { maxMessages: 3 });
    const lastEvent = history.events.at(-1);
    const lastEnd = [...history.events].reverse().find((event) => event.type === 'turn/end');
    let status = session.running ? 'running' : 'idle';
    const endState = lastEnd?.data?.status || lastEnd?.data?.outcome || lastEnd?.data?.reason?.kind;
    if (!session.running && endState === 'completed') status = 'completed';
    if (!session.running && ['failed', 'cancelled', 'canceled'].includes(endState)) status = endState === 'canceled' ? 'cancelled' : endState;
    return { sessionId, status, running: session.running, idle: !session.running, lastEventSeq: lastEvent?.seq ?? -1, lastTurnEnd: lastEnd || null };
  }

  async followSession(sessionId, { afterSeq = -1, pollMs = 1000, timeoutMs = 0, onEvents = () => {} } = {}) {
    const started = Date.now();
    let cursor = afterSeq;
    while (timeoutMs === 0 || Date.now() - started < timeoutMs) {
      const snapshot = await this.readSession(sessionId, { maxMessages: 3 });
      const fresh = snapshot.events.filter((event) => event.seq > cursor).sort((a, b) => a.seq - b.seq);
      if (fresh.length) {
        cursor = fresh.at(-1).seq;
        await onEvents(fresh);
      }
      const status = await this.getStatus(sessionId);
      const completed = fresh.some((event) => event.type === 'turn/end' && (event.data?.status === 'completed' || event.data?.reason?.kind === 'completed')) && status.idle;
      if (completed) return {
        sessionId,
        status: 'completed',
        lastEventSeq: cursor,
        events: fresh.filter((event) => ['user/message', 'assistant/message', 'tool/call', 'tool/result', 'turn/end'].includes(event.type)).map(compactEvent),
      };
      await delay(pollMs);
    }
    return { sessionId, status: 'timeout', lastEventSeq: cursor, events: [] };
  }

  async assertSession(sessionId) {
    const { sessions } = await this.listSessions({ includeBlank: true });
    const session = sessions.find((item) => item.sessionId === sessionId);
    if (!session) throw new HarnessError('Session not found', { sessionId });
    return session;
  }
}

function unwrap(body, method) {
  if (body?.result?.ok === true) return body.result.value;
  if (body?.result?.ok === false) throw new HarnessError(body.result.error?.message || `${method} failed`, body.result.error);
  if (body?.ok === true && 'value' in body) return body.value;
  if (body?.ok === false) throw new HarnessError(body.error?.message || `${method} failed`, body.error);
  return body;
}

export function messageText(event) {
  if (event.type === 'user/message') return blocksText(event.data?.content);
  if (event.type === 'assistant/message') return blocksText(event.data?.message?.content);
  if (event.type === 'tool/result') return blocksText(event.data?.message?.content);
  return '';
}

export function compactEvent(event) {
  const copy = structuredClone(event);
  delete copy.sourceEventSeqs;
  if (copy.type === 'assistant/message' && copy.data?.message?.content) {
    copy.data.message.content = copy.data.message.content.filter((block) => block.type !== 'reasoning');
  }
  return copy;
}

function blocksText(content) {
  if (!Array.isArray(content)) return '';
  return content.filter((block) => block?.type === 'text').map((block) => block.text).join('\n');
}

function normalizePath(value = '') {
  return value.replaceAll('/', '\\').replace(/\\+$/, '').toLowerCase();
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
