import { readFileSync } from "node:fs";
import vm from "node:vm";

// Minimal DOM so app.js can finish loading; we only exercise the pure render helpers.
const noop = () => {};
const stubEl = () => ({ addEventListener: noop, classList:{add:noop, remove:noop, toggle:noop},
  querySelector:()=>null, querySelectorAll:()=>[], style:{}, dataset:{}, textContent:"", innerHTML:"", value:"", files:[] });
const sandbox = {
  document: { getElementById: stubEl, querySelectorAll: () => [], querySelector: () => null },
  window: {}, setTimeout: noop, clearTimeout: noop, console,
  fetch: async () => ({ ok:true, json: async () => [] }),
};
sandbox.globalThis = sandbox;
vm.createContext(sandbox);
vm.runInContext(readFileSync("app/static/app.js", "utf8"), sandbox);

const { segmentCard } = sandbox;
let failures = 0;
const check = (name, condition) => {
  if (!condition) { failures += 1; console.log(`  FAIL  ${name}`); } else { console.log(`  ok    ${name}`); }
};

const flagged = { id:"s1", segment_index:5, kind:"narration", qa_status:"needs_review",
  tts_audio_url:"/media/a.wav", transcript_si:"සිංහල", narration_en:"English",
  qa:{ passed:false, issues:["Duration ratio 0.42 is outside 0.55-1.45"] } };
const passed = { ...flagged, qa_status:"passed", qa:{passed:true, issues:[]} };
const approved = { ...flagged, qa_status:"passed",
  qa:{ passed:true, approved:true, issues:[], overridden_issues:["Duration ratio 0.42 is outside 0.55-1.45"] } };
const noAudio = { ...flagged, tts_audio_url:null };
const recording = { id:"s2", segment_index:6, kind:"original", qa_status:"needs_review",
  tts_audio_url:"/media/b.wav", transcript_si:"", qa:{passed:false, issues:["Recorded audio kept as is. Confirm."]} };

console.log("approve button visibility");
check("shown on a flagged narration segment", segmentCard(flagged).includes('class="approve'));
check("hidden once the segment has passed", !segmentCard(passed).includes('class="approve'));
check("hidden when there is no audio to judge", !segmentCard(noAudio).includes('class="approve'));
check("never offered on a kept recording", !segmentCard(recording).includes('class="approve'));

console.log("\nstate shown to the reviewer");
check("flagged card shows the QA issue", segmentCard(flagged).includes("Duration ratio 0.42"));
check("approved card reads 'approved'", segmentCard(approved).includes(">approved<"));
check("approved card still shows what was overridden", segmentCard(approved).includes("Approved over: Duration ratio 0.42"));
check("recording keeps its confirm button", segmentCard(recording).includes('class="confirm"'));
check("confirmed recording has confirm disabled", segmentCard({...recording, qa_status:"passed"}).includes("disabled"));

console.log("\nsafety and plumbing");
check("every card carries an error slot", segmentCard(flagged).includes('class="segment-error error"'));
check("textareas remember their loaded value", segmentCard(flagged).includes("data-initial="));
check("script injection in narration is escaped",
  !segmentCard({...flagged, narration_en:'</textarea><img src=x onerror=alert(1)>'}).includes("<img src=x"));

console.log(failures ? `\n${failures} FAILED` : "\nall checks passed");
process.exit(failures ? 1 : 0);
