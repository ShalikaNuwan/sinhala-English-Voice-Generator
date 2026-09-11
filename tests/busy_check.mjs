import { readFileSync } from "node:fs";
import vm from "node:vm";

const noop = () => {};
const stubEl = () => ({ addEventListener: noop, classList:{add:noop,remove:noop,toggle:noop,contains:()=>false},
  querySelector:()=>null, querySelectorAll:()=>[], style:{}, dataset:{}, textContent:"", innerHTML:"", value:"", files:[] });
const sandbox = { document:{ getElementById: stubEl, querySelectorAll:()=>[], querySelector:()=>null },
  window:{}, setTimeout:(fn,ms)=>globalThis.setTimeout(fn,ms), clearTimeout:noop, console,
  fetch: async () => ({ ok:true, json: async () => [] }) };
sandbox.globalThis = sandbox;
vm.createContext(sandbox);
vm.runInContext(readFileSync("app/static/app.js", "utf8"), sandbox);
const { run } = sandbox;

let failures = 0;
const check = (name, ok) => { if (!ok) { failures++; console.log(`  FAIL  ${name}`); } else console.log(`  ok    ${name}`); };

/** A card holding three buttons, close enough to the real DOM for run(). */
function makeCard() {
  const errorSlot = { textContent:"", hidden:true };
  const buttons = [];
  const card = {
    classList:{ contains:(c)=>c==="segment" },
    querySelector:(sel)=> sel===".segment-error" ? errorSlot : null,
    querySelectorAll:()=>buttons,
  };
  const make = (label) => {
    const b = { textContent:label, disabled:false, dataset:{}, isConnected:true, closest:()=>card };
    buttons.push(b); return b;
  };
  return { card, errorSlot, buttons, make };
}

console.log("double click");
{
  const { make, buttons } = makeCard();
  const button = make("Approve");
  let calls = 0;
  const slow = async () => { calls += 1; await new Promise(r => setTimeout(r, 60)); };
  const first = run(button, slow, {label:"Approving…"});
  const second = run(button, slow, {label:"Approving…"});      // fired before the first finished
  check("busy flag set while in flight", button.dataset.busy === "1");
  check("label swapped to the busy text", button.textContent === "Approving…");
  check("sibling buttons disabled during the action", buttons.every(b => b.disabled));
  await Promise.all([first, second]);
  check("the action ran exactly once", calls === 1);
  check("label restored afterwards", button.textContent === "Approve");
  check("buttons re-enabled afterwards", buttons.every(b => !b.disabled));
  check("busy flag cleared", button.dataset.busy === "");
}

console.log("\nfailure handling");
{
  const { make, buttons, errorSlot } = makeCard();
  const button = make("Approve");
  make("Regenerate this segment");
  await run(button, async () => { throw new Error("This segment has no voiced audio to approve"); });
  check("message shown in the card, not an alert", errorSlot.textContent.includes("no voiced audio"));
  check("error slot made visible", errorSlot.hidden === false);
  check("buttons usable again after a failure", buttons.every(b => !b.disabled));
  check("can be retried after a failure", button.dataset.busy === "");
}

console.log("\nsequential clicks");
{
  const { make } = makeCard();
  const button = make("Approve");
  let calls = 0;
  await run(button, async () => { calls += 1; });
  await run(button, async () => { calls += 1; });
  check("a second click after completion does run", calls === 2);
}

console.log(failures ? `\n${failures} FAILED` : "\nall checks passed");
process.exit(failures ? 1 : 0);
