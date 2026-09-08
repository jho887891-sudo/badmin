import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import { spawn } from 'node:child_process';
import { join } from 'node:path';

export class CodexDelivery {
  constructor({ command = process.env.CODEX_RELAY_DELIVERY_COMMAND || resolveCodexCommand(), codexHome = process.env.CODEX_HOME || join(process.env.USERPROFILE || '', '.codex'), spawnImpl = spawn, timeoutMs = Number(process.env.CODEX_RELAY_DELIVERY_TIMEOUT_MS || 10_000) } = {}) {
    this.command = command;
    this.codexHome = codexHome;
    this.spawn = spawnImpl;
    this.timeoutMs = timeoutMs;
  }

  isBusy(threadId) {
    const processFile = join(this.codexHome, 'process_manager', 'chat_processes.json');
    try {
      const rows = JSON.parse(readFileSync(processFile, 'utf8'));
      if (rows.some((row) => row.conversationId === threadId && row.osPid)) return true;
    } catch {}
    return existsSync(join(this.codexHome, 'thread-writer-locks', `${threadId}.lock`));
  }

  async deliver(threadId, message) {
    if (this.isBusy(threadId)) return { delivered: false, busy: true, method: 'pending-queue' };
    const configured = process.env.CODEX_RELAY_DELIVERY_COMMAND;
    const args = configured
      ? expandTemplate(configured, threadId, message)
      : ['exec', 'resume', '--json', threadId, message];
    const command = configured ? args.shift() : this.command;
    return new Promise((resolve) => {
      const child = this.spawn(command, args, { windowsHide: true, shell: false, stdio: ['ignore', 'pipe', 'pipe'] });
      let settled = false;
      let output = '';
      let error = '';
      const finish = (result) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        resolve(result);
      };
      const timer = setTimeout(() => {
        child.kill();
        finish({ delivered: false, busy: true, method: 'codex-exec-resume', error: `delivery timed out after ${this.timeoutMs}ms` });
      }, this.timeoutMs);
      child.stdout?.on('data', (chunk) => { output += chunk; });
      child.stderr?.on('data', (chunk) => { error += chunk; });
      child.on('error', (cause) => finish({ delivered: false, busy: false, method: 'codex-exec-resume', error: cause.message }));
      child.on('close', (code) => finish({ delivered: code === 0, busy: false, method: 'codex-exec-resume', code, output, error }));
    });
  }
}

function expandTemplate(template, threadId, message) {
  const parts = template.match(/(?:[^\s"]+|"[^"]*")+/g) || [];
  return parts.map((part) => part.replace(/^"|"$/g, '').replaceAll('{thread_id}', threadId).replaceAll('{message}', message));
}

function resolveCodexCommand() {
  const root = join(process.env.LOCALAPPDATA || '', 'OpenAI', 'Codex', 'bin');
  try {
    const candidates = readdirSync(root, { withFileTypes: true })
      .filter((entry) => entry.isDirectory())
      .map((entry) => join(root, entry.name, 'codex.exe'))
      .filter(existsSync)
      .sort((a, b) => statSync(b).mtimeMs - statSync(a).mtimeMs);
    if (candidates[0]) return candidates[0];
  } catch {}
  return 'codex';
}
