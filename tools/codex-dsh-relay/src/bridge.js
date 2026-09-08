import { HarnessClient, compactEvent, messageText } from './harness.js';

export function createSessionBridge(client = new HarnessClient()) {
  return {
    async deepseek_list_sessions(input = {}) {
      return client.listSessions({ cwd: input.cwd, includeBlank: input.include_blank ?? true, cursor: input.cursor });
    },
    async deepseek_read_session(input) {
      const result = await client.readSession(input.session_id, { maxMessages: input.max_messages || 10, beforeSeq: input.before_seq });
      return {
        ...result,
        messages: result.events
          .filter((event) => ['user/message', 'assistant/message'].includes(event.type))
          .map((event) => ({ seq: event.seq, role: event.type.split('/')[0], text: messageText(event) }))
          .filter((message) => message.text),
        tool_events: result.events.filter((event) => ['tool/call', 'tool/result'].includes(event.type)).map(compactEvent),
      };
    },
    async deepseek_send_message(input) {
      return client.sendMessage(input.session_id, input.message, { mode: input.mode || 'queue' });
    },
    async deepseek_follow_session(input) {
      const events = [];
      const result = await client.followSession(input.session_id, {
        afterSeq: input.after_seq ?? -1,
        timeoutMs: input.timeout_ms ?? 30_000,
        pollMs: input.poll_ms ?? 1000,
        onEvents: (batch) => events.push(...batch.filter((event) => ['assistant/message', 'tool/call', 'tool/result', 'turn/end'].includes(event.type))),
      });
      return { ...result, events: events.map(compactEvent) };
    },
    async deepseek_get_status(input) {
      return client.getStatus(input.session_id);
    },
    async deepseek_stop(input) {
      return client.stop(input.session_id);
    },
  };
}
