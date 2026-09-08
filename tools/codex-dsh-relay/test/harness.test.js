import test from 'node:test';
import assert from 'node:assert/strict';
import { HarnessClient } from '../src/harness.js';

test('detects dot RPC envelope and preserves rpc correlation', async () => {
  const seen = [];
  const client = new HarnessClient({ fetchImpl: async (url, init) => {
    seen.push({ url, body: JSON.parse(init.body) });
    const request = seen.at(-1).body;
    return new Response(JSON.stringify({ type: 'server-response', rpcId: request.rpcId, result: { ok: true, value: { items: [] } } }), { status: 200 });
  }});
  const result = await client.listSessions();
  assert.equal(result.protocol, 'dot-rpc-envelope');
  assert.equal(seen[0].url, 'http://127.0.0.1:3080/api/session.list');
  assert.equal(seen[0].body.method, 'session.list');
});

test('sendMessage never creates or forks a session', async () => {
  const methods = [];
  const client = new HarnessClient();
  client.protocol = 'dot-rpc-envelope';
  client.assertSession = async (sessionId) => ({ sessionId, blank: false });
  client.call = async (method) => {
    methods.push(method);
    return { value: { accepted: true }, requestId: 'rpc-1' };
  };
  const result = await client.sendMessage('session-existing', 'hello');
  assert.deepEqual(result, { sessionId: 'session-existing', accepted: true, requestId: 'rpc-1' });
  assert.deepEqual(methods, ['session.prompt']);
});
