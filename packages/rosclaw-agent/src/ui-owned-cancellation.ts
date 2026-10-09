// Extracted from main.ts so the actual UI adapter can be exercised without
// starting the CLI. main.ts imports and re-exports this same binding.
export function attachOwnedUIAbort(genuineMode: unknown, options: {
 rosclawHome: string;
 current(): import("./extension/index.js").UIReceiptOwner | undefined;
 turn(): string | undefined;
}) {
 const mode = genuineMode as {
  defaultEditor: { handleInput(data: string): void };
  session: { abort(): Promise<void>; isStreaming: boolean };
 };
 const pending = new Set<Promise<void>>();
 const outcomes = new Map<string, { ok: boolean; code?: string }>();
 const inFlight = new Set<string>();
 const confirmed = new Set<string>();
 let inputAbort = false;
 const originalInput = mode.defaultEditor.handleInput;
 const session = mode.session;
 const originalAbort = session.abort;
 // Recheck the instance's prototype at abort time: the observed installed
 // SDK path can replace that hook after this adapter attaches. This does not
 // establish timing for other SDK versions or Ctrl-C input delivery.
 const abortBeforeInit = Object.getPrototypeOf(session)?.abort as typeof originalAbort | undefined;
 mode.defaultEditor.handleInput = function(data) {
  const prior = inputAbort;
  inputAbort = data === "\u001b" || data === "\u0003";
  try { return originalInput.call(this, data); }
  finally { inputAbort = prior; }
 };
 session.abort = function() {
  const owner = options.current();
  if (inputAbort && owner && owner.turn_id === options.turn()) {
   const captured = { ...owner };
   const key = JSON.stringify(captured);
   if (!inFlight.has(key) && !confirmed.has(key)) {
    inFlight.add(key);
    const work = (async () => {
     try {
      const { bridgeCall } = await import("./bridge/bridge-client.js");
      if (options.current() !== owner || options.turn() !== captured.turn_id) {
       outcomes.set(key, { ok: false, code: "OWNERSHIP_CHANGED" });
       return;
      }
      const result = await bridgeCall(options.rosclawHome, "pi.op.cancel_owned", captured);
      const unchanged = options.current() === owner && options.turn() === captured.turn_id;
      const ok = unchanged && result.ok === true;
      outcomes.set(key, { ok, code: unchanged ? String(result.code ?? "") : "OWNERSHIP_CHANGED" });
      if (ok) confirmed.add(key);
     } catch {
      outcomes.set(key, { ok: false, code: "CANCEL_RPC_FAILED" });
     } finally {
      inFlight.delete(key);
     }
    })();
    pending.add(work);
    void work.then(() => pending.delete(work));
   }
  }
  const liveAbort = Object.getPrototypeOf(session)?.abort as typeof originalAbort | undefined;
  return (typeof liveAbort === "function" && liveAbort !== abortBeforeInit
   ? liveAbort : originalAbort).call(this);
 };
 return {
  async drain() {
   while (pending.size) await Promise.all([...pending]);
   const result = [...outcomes.values()];
   outcomes.clear();
   return result;
  },
 };
}
