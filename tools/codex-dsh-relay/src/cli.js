#!/usr/bin/env node
import { resolve } from 'node:path';
import { compactEvent, HarnessClient, messageText } from './harness.js';
import { RelayStore } from './db.js';
import { RelayService } from './relay.js';

const [command, ...args] = process.argv.slice(2);
const harness = new HarnessClient();
const store = new RelayStore(resolve(import.meta.dirname, '..', 'relay', 'jobs.sqlite'));
const relay = new RelayService({ harness, store });

try {
  switch (command) {
    case 'daemon': await relay.daemon(); break;
    case 'sessions': await sessions(); break;
    case 'attach': await attach(args[0]); break;
    case 'send': await send(args[0], args.slice(1).join(' ')); break;
    case 'follow': await follow(args[0]); break;
    case 'jobs': print(store.list()); break;
    case 'status': await status(args[0]); break;
    case 'cancel': await cancel(args[0]); break;
    case 'delegate': await delegate(args); break;
    case 'once': print({ processed: await relay.processOnce() }); break;
    default: usage(command ? `Unknown command: ${command}` : null);
  }
} catch (error) {
  console.error(JSON.stringify({ error: error.message, details: error.details || null }, null, 2));
  process.exitCode = 1;
} finally {
  if (command !== 'daemon') store.close();
}

async function sessions() {
  const cwdIndex = args.indexOf('--cwd');
  const cwd = cwdIndex >= 0 ? args[cwdIndex + 1] : undefined;
  const result = await harness.listSessions({ cwd, includeBlank: !args.includes('--non-blank') });
  print(result);
}

async function attach(sessionId) {
  required(sessionId, 'session id');
  const result = await harness.readSession(sessionId, { maxMessages: 10 });
  print({
    sessionId: result.sessionId,
    metadata: result.metadata,
    hasMore: result.hasMore,
    projections: result.projections,
    messages: result.events.filter((event) => ['user/message', 'assistant/message'].includes(event.type)).map((event) => ({ seq: event.seq, role: event.type.split('/')[0], text: messageText(event) })).filter((item) => item.text),
    toolEvents: result.events.filter((event) => ['tool/call', 'tool/result', 'turn/end'].includes(event.type)).map(compactEvent),
  });
}

async function send(sessionId, message) {
  required(sessionId, 'session id'); required(message, 'message');
  print(await harness.sendMessage(sessionId, message));
}

async function follow(sessionId) {
  required(sessionId, 'session id');
  const afterIndex = args.indexOf('--after');
  const afterSeq = afterIndex >= 0 ? Number(args[afterIndex + 1]) : -1;
  const timeoutIndex = args.indexOf('--timeout');
  const timeoutMs = timeoutIndex >= 0 ? Number(args[timeoutIndex + 1]) : 0;
  const result = await harness.followSession(sessionId, { afterSeq, timeoutMs, onEvents: (events) => {
    for (const event of events.filter((item) => ['user/message', 'assistant/message', 'tool/call', 'tool/result', 'turn/end'].includes(item.type))) print(compactEvent(event));
  } });
  print(result);
}

async function status(id) {
  required(id, 'job id or session id');
  const job = store.get(id);
  print(job || await harness.getStatus(id));
}

async function cancel(id) {
  required(id, 'job id or session id');
  const job = store.get(id);
  if (job) {
    await harness.stop(job.deepseekSessionId);
    print(store.update(id, { status: 'cancelled', codexDeliveryStatus: 'cancelled' }));
  } else print(await harness.stop(id));
}

async function delegate(values) {
  const sessionIndex = values.indexOf('--session');
  const messageIndex = values.indexOf('--message');
  const sessionId = sessionIndex >= 0 ? values[sessionIndex + 1] : null;
  const message = messageIndex >= 0 ? values[messageIndex + 1] : null;
  required(sessionId, '--session'); required(message, '--message');
  print(await relay.delegate({ sessionId, message }));
}

function required(value, label) { if (!value) throw new Error(`Missing ${label}`); }
function print(value) { console.log(JSON.stringify(value, null, 2)); }
function usage(error) {
  if (error) console.error(error);
  console.log(`Usage:
  relay daemon
  relay sessions [--cwd PATH] [--non-blank]
  relay attach <deepseek_session_id>
  relay send <deepseek_session_id> "message"
  relay follow <deepseek_session_id> [--after SEQ] [--timeout MS]
  relay jobs
  relay status <job_id|session_id>
  relay cancel <job_id|session_id>
  relay delegate --session <deepseek_session_id> --message "..."`);
  if (error) process.exitCode = 1;
}
