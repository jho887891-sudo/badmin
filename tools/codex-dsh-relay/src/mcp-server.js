#!/usr/bin/env node
import readline from 'node:readline';
import { createSessionBridge } from './bridge.js';

const bridge = createSessionBridge();
const schemas = {
  deepseek_list_sessions: { type: 'object', properties: { cwd: { type: 'string' }, include_blank: { type: 'boolean' }, cursor: { type: 'string' } }, additionalProperties: false },
  deepseek_read_session: { type: 'object', required: ['session_id'], properties: { session_id: { type: 'string' }, max_messages: { type: 'integer', minimum: 1 }, before_seq: { type: 'integer', minimum: 0 } }, additionalProperties: false },
  deepseek_send_message: { type: 'object', required: ['session_id', 'message'], properties: { session_id: { type: 'string' }, message: { type: 'string' }, mode: { type: 'string', enum: ['queue', 'steer'] } }, additionalProperties: false },
  deepseek_follow_session: { type: 'object', required: ['session_id'], properties: { session_id: { type: 'string' }, after_seq: { type: 'integer' }, timeout_ms: { type: 'integer', minimum: 1 }, poll_ms: { type: 'integer', minimum: 100 } }, additionalProperties: false },
  deepseek_get_status: { type: 'object', required: ['session_id'], properties: { session_id: { type: 'string' } }, additionalProperties: false },
  deepseek_stop: { type: 'object', required: ['session_id'], properties: { session_id: { type: 'string' } }, additionalProperties: false },
};

const descriptions = {
  deepseek_list_sessions: 'List existing sessions from the running DeepSeek Harness. Never creates a session.',
  deepseek_read_session: 'Read durable history from an existing DeepSeek Harness session.',
  deepseek_send_message: 'Send a prompt to the same existing DeepSeek Harness session.',
  deepseek_follow_session: 'Follow durable assistant, tool, and turn completion events for one session.',
  deepseek_get_status: 'Get running/idle/completion status and the latest durable event sequence.',
  deepseek_stop: 'Cancel only the selected DeepSeek Harness session current run.',
};

const input = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
input.on('line', async (line) => {
  let request;
  try { request = JSON.parse(line); } catch { return; }
  if (!request.id) return;
  try {
    if (request.method === 'initialize') return reply(request.id, { protocolVersion: '2025-03-26', capabilities: { tools: {} }, serverInfo: { name: 'codex-dsh-session-bridge', version: '0.1.0' } });
    if (request.method === 'ping') return reply(request.id, {});
    if (request.method === 'tools/list') return reply(request.id, { tools: Object.keys(schemas).map((name) => ({ name, description: descriptions[name], inputSchema: schemas[name] })) });
    if (request.method === 'tools/call') {
      const handler = bridge[request.params?.name];
      if (!handler) throw new Error(`Unknown tool: ${request.params?.name}`);
      const value = await handler(request.params.arguments || {});
      return reply(request.id, { content: [{ type: 'text', text: JSON.stringify(value, null, 2) }], structuredContent: value });
    }
    replyError(request.id, -32601, `Method not found: ${request.method}`);
  } catch (error) {
    reply(request.id, { content: [{ type: 'text', text: JSON.stringify({ error: error.message, details: error.details || null }) }], isError: true });
  }
});

function reply(id, result) { process.stdout.write(`${JSON.stringify({ jsonrpc: '2.0', id, result })}\n`); }
function replyError(id, code, message) { process.stdout.write(`${JSON.stringify({ jsonrpc: '2.0', id, error: { code, message } })}\n`); }
