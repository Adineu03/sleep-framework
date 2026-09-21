# SLEEP — Presenter Script for the Scroll Experience

*Open `experience/index.html` fullscreen (F11). Scroll slowly and evenly — the page does the
work. Say only what's below; let the animations breathe. Total: ~12–14 minutes.*

---

**[Landing — the title]**
"This is SLEEP — five months of research in one scroll. Stay with me."

**[Scroll → the problem]**
"Every LLM you've used has amnesia. Close the chat, and everything it learned is gone."
*(let the three cards land)* "RAG is a workaround — a lookup, not memory."
*(the research question appears)* "So: can a frozen model learn permanently, from one exposure,
without destroying itself? That's the whole project."

**[Scroll → the brain]**
"Biology solved this. Watch." *(say nothing while the brain tours: tagging → competition)*
*(at replay, when the brain turns glassy)* "During sleep, deep inside, the hippocampus replays
the day — and trains the cortex."
*(result line)* "That output spec — that's exactly what an LLM is missing."

**[Scroll → SLEEP, the proposal]**
"So we built it. This is a transformer." *(dismantle)* "Every layer: attention — the librarian —
and an MLP — the filing cabinets. Remember that."
*(as the five organs dock, name them only)* "Tagging. Memory bank. The budget office.
The night shift. And the wafers — the only part that ever learns."
*(wafer card)* "Note the address: top-third attention. Where the biology analogy pointed.
Keep an eye on that."
*(reassembly)* "Same frozen model — now with a hippocampus, a budget office, and a night shift."

**[Scroll → one fact's journey]**
"Let's follow one fact through — this is a real fact from our test set."
*(let the packet travel; read nothing aloud — the cards do it)*
*(at the morning Q&A)* "That's the design intent. Now we test it."

**[Scroll → the exam]**
"Three numbers. Recognition — the easy mode. Recall — the real test. And a damage meter.
The memory bank is wiped before every question: only the weights may answer."

**[Scroll → first results]**
*(silence while the first two bars draw)*
*(when the recall bar stays empty)* "Zero. Not low — zero."
*(the giant 0.00)* "We named it: recognition without recall."

**[Scroll → the feedbacks]**
"Two reviews said the same thing: prove it. So we rebuilt everything around proof —
five seeds, four models, ten cycles, real baselines, and the system never grades its own
homework again."

**[Scroll → replication]**
*(let the four rows draw silently)*
*(the giant ≈0.006)* "Same floor. Every model. This isn't a bug in one model —
it's the mechanism."

**[Scroll → the low point]**
*(read the lines with the scroll, slowly)* "…and honestly, this is where the paper almost
ended. An honest negative result. 'Empirical Limits.'"
*(the green line)* "…unless we were asking the wrong question."

**[Scroll → the turn]**
"We ran a pre-mortem: assume we're wrong — where's the bug?"
*(wrong address highlights)* "We'd been handing facts to the librarian."
*(MLP highlights)* "The editing literature had the map all along: facts live in the
mid-stack MLPs."
*(THE MOVE — say nothing. Let the wafers travel. This is the money shot.)*
*(the clamp)* "And a third culprit: our own safety clip was erasing the learning.
We didn't remove it — we right-sized it."

**[Scroll → the proof]**
*(bars draw; speak only at the finale)* "0.75. On Mistral — a family the recipe never met
during development. The floor was 0.006."

**[Scroll → the long run]**
"Ten cycles, line by line. Solid is one seed, dotted is its twin."
*(as baselines draw)* "The standard baseline preserves but doesn't learn. SLEEP holds
three times above both — the whole horizon."

**[Scroll → before/after]**
*(read nothing — let the table build. Pause at the last row.)*
"The title of the paper itself changed: from 'Empirical Limits' to
'Diagnosing and Repairing'."

**[Scroll → the gaps]**
"Two things we're honest about." *(shelf dims)* "Early memories fade — each night rehearses
only its own facts."
*(the sweep)* "The fix is what real sleep does: replay old memories alongside new.
Not built yet. It's next."

**[Scroll → where it could go]**
"Bengaluru, Monday. Documents go in once." *(night falls)* "The model sleeps on it."
*(Tuesday Q&A)* "Next morning — answers from its own weights."
*(the pan)* "One month later, 2,700 kilometres away, nobody re-uploads anything…"
*(Dubai Q&A)* "…and it still knows."
*(the final line — read it exactly as written on screen, then stop)*
"This is the vision — not yet production-optimised, and a long way from there.
But it is the first step. Thank you."

---

*Q&A crib: floor 0.006 → 0.750 (Mistral, 5 seeds ±0.033) · beats naive 4/4 models ·
3× vs EWC · naive detonates 1.5B ×24, SLEEP holds ×2.4 · 35 pre-registered runs, zero
failures · gaps: early-batch fade, ~+0.12 damage/cycle, Llama tuning ·
repo: github.com/Adineu03/sleep-framework*
