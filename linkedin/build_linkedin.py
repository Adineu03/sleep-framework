"""Render the 6-image SLEEP LinkedIn series (1080x1350, 4:5) via Edge headless.

Win-first arc: hook -> how it works (architecture) -> the proof -> beats the
baselines -> the breakthrough insight -> what's next.

Run:  python build_linkedin.py
Out:  linkedin/01.png ... 06.png  (rendered at 2x: 2160x2700)
"""

from __future__ import annotations

import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
HTML_DIR = os.path.join(HERE, "html")
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
os.makedirs(HTML_DIR, exist_ok=True)

CSS = """
* { margin:0; padding:0; box-sizing:border-box; }
body { background:#F4F6FA; }
.card {
  position:relative; width:1080px; height:1350px; overflow:hidden;
  background:#F4F6FA; color:#1A1F2E;
  font-family:'Segoe UI', system-ui, sans-serif;
  display:flex; flex-direction:column; padding:72px 72px 60px;
}
.card.dark { background:#141927; color:#E6EAF4; }
.eyebrow { font-family:Consolas, monospace; font-size:22px; letter-spacing:0.18em;
  text-transform:uppercase; color:#3D52C9; margin-bottom:34px; }
.dark .eyebrow { color:#8FA0FF; }
h1 { font-family:Georgia, serif; font-weight:normal; font-size:64px; line-height:1.18;
  letter-spacing:-0.01em; }
h1 b { font-weight:650; }
.sub { font-size:27px; color:#5C6577; margin-top:16px; }
.dark .sub { color:#97A0B5; }
.footer { margin-top:auto; border-top:1px solid #D5DBE7; padding-top:22px;
  font-family:Consolas, monospace; font-size:19px; color:#5C6577;
  display:flex; justify-content:space-between; }
.dark .footer { border-top-color:#2C3349; color:#97A0B5; }
.mono { font-family:Consolas, monospace; }
.amber { color:#A96D12; } .green { color:#2C7D57; } .indigo { color:#3D52C9; }
.dark .green { color:#4FB98A; } .dark .amber { color:#E0A44A; }

.panel { background:#FFFFFF; border:1px solid #D5DBE7; border-radius:6px; padding:40px 44px; }
.dark .panel { background:#1B2133; border-color:#2C3349; }

/* vertical bar chart */
.vchart { display:flex; align-items:flex-end; justify-content:space-around; height:480px;
  border-bottom:3px solid #1A1F2E; padding:0 10px; }
.vg { display:flex; flex-direction:column; align-items:center; }
.vbars { display:flex; align-items:flex-end; gap:10px; }
.vb { width:110px; border-radius:4px 4px 0 0; position:relative; }
.vb .val { position:absolute; top:-40px; left:50%; transform:translateX(-50%);
  font-family:Consolas, monospace; font-size:26px; white-space:nowrap; color:#1A1F2E; }
.vlab { font-size:23px; color:#3D4658; text-align:center; line-height:1.25; }

/* architecture */
.arch-stage { background:#FFFFFF; border:1px solid #D5DBE7; border-radius:8px;
  padding:26px 30px; display:flex; flex-direction:column; gap:16px; }
.arch-stage .tag { font-family:Consolas, monospace; font-size:20px; letter-spacing:0.14em;
  text-transform:uppercase; }
.arch-step { display:flex; align-items:center; gap:18px; }
.arch-step svg { flex:0 0 44px; }
.arch-step p { font-size:24px; line-height:1.35; color:#3D4658; }
.arch-step p b { color:#1A1F2E; }
.arch-arrow { text-align:center; font-size:40px; color:#8093C9; line-height:1; }

.row3 { border:1px solid #2C3349; border-radius:8px; padding:30px 34px; display:flex;
  align-items:center; gap:28px; }
.row3 svg { flex:0 0 64px; }
.row3 p { font-size:29px; line-height:1.45; color:#C3CAD9; }
.row3 p b { color:#E6EAF4; font-weight:650; }

.chip { display:inline-flex; align-items:center; gap:12px; background:#E2F1EA;
  color:#2C7D57; border-radius:6px; padding:14px 22px; font-size:25px; font-weight:600; }
"""


def wrap(body: str, dark: bool = False) -> str:
    cls = "card dark" if dark else "card"
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>SLEEP LinkedIn</title>
<style>{CSS}</style></head>
<body><div class="{cls}">{body}</div></body></html>"""


IMAGES: list[tuple[bool, str]] = []

# ---- 01: The hook ----------------------------------------------------------
IMAGES.append((False, """
<p class="eyebrow">SLEEP &middot; A research update</p>
<h1 style="font-size:72px;">In April, I posted about a memory system that <b>couldn't remember</b>.<br><br>Today, <b>it can</b>.</h1>
<div style="flex:1; display:flex; flex-direction:column; justify-content:center; align-items:center; gap:18px;">
  <div style="display:flex; align-items:baseline; gap:44px;">
    <span class="mono amber" style="font-size:84px;">0.006</span>
    <span style="font-size:72px; color:#5C6577;">&rarr;</span>
    <span class="mono green" style="font-size:168px; font-weight:600;">0.75</span>
  </div>
  <p class="sub mono" style="font-size:24px; letter-spacing:0.08em;">FREE-FORM RECALL &middot; BEFORE &rarr; AFTER</p>
</div>
<div class="footer"><span>SLEEP &middot; SP Jain School of Global Management</span><span>1 / 6</span></div>
"""))

# ---- 02: How it works (architecture) ---------------------------------------
IMAGES.append((False, """
<p class="eyebrow">How it works</p>
<h1 style="font-size:56px;">A model that learns the way <b>you</b> do:<br>it sleeps on it.</h1>
<div style="flex:1; display:flex; flex-direction:column; justify-content:center; gap:14px; margin-top:26px;">
  <div class="arch-stage" style="border-left:6px solid #3D52C9;">
    <span class="tag indigo">&#9728; Wake &mdash; while working</span>
    <div class="arch-step">
      <svg viewBox="0 0 44 44" width="44" height="44" fill="none" stroke="#3D52C9" stroke-width="2.6"><rect x="8" y="4" width="28" height="36" rx="3"/><line x1="14" y1="14" x2="30" y2="14"/><line x1="14" y1="21" x2="30" y2="21"/><line x1="14" y1="28" x2="24" y2="28"/></svg>
      <p><b>Reads new information once</b> &mdash; a document, a fact, a conversation.</p>
    </div>
    <div class="arch-step">
      <svg viewBox="0 0 44 44" width="44" height="44" fill="none" stroke="#3D52C9" stroke-width="2.6"><path d="M22 4 l5 11 12 1-9 8 3 12-11-7-11 7 3-12-9-8 12-1z"/></svg>
      <p><b>Flags what's surprising</b> &mdash; and files it into short-term memory, exactly once.</p>
    </div>
  </div>
  <div class="arch-arrow">&darr;</div>
  <div class="arch-stage" style="border-left:6px solid #2C7D57;">
    <span class="tag green">&#9789; Sleep &mdash; offline</span>
    <div class="arch-step">
      <svg viewBox="0 0 44 44" width="44" height="44" fill="none" stroke="#2C7D57" stroke-width="2.6"><path d="M6 22 a16 16 0 1 0 6-12" /><path d="M6 6 v10 h10"/></svg>
      <p><b>Replays each memory in 20+ wordings</b> &mdash; teaching itself the fact, not the sentence.</p>
    </div>
    <div class="arch-step">
      <svg viewBox="0 0 44 44" width="44" height="44" fill="none" stroke="#2C7D57" stroke-width="2.6"><rect x="6" y="8" width="32" height="28" rx="3"/><line x1="6" y1="18" x2="38" y2="18"/><line x1="6" y1="27" x2="38" y2="27"/><line x1="15" y1="12" x2="22" y2="12"/><line x1="15" y1="22" x2="29" y2="22"/></svg>
      <p><b>Writes it where facts actually live</b> &mdash; the mid-stack layers &mdash; under safety rails that protect old knowledge.</p>
    </div>
  </div>
  <div class="arch-arrow">&darr;</div>
  <div class="chip" style="align-self:center;">&#10003;&nbsp; Tomorrow: answers from its own weights &mdash; no prompt stuffing, no retrieval</div>
</div>
<div class="footer"><span>SLEEP &middot; SP Jain School of Global Management</span><span>2 / 6</span></div>
"""))

# ---- 03: The proof ---------------------------------------------------------
IMAGES.append((False, """
<p class="eyebrow">The proof</p>
<h1 style="font-size:56px;">Read once. Sleep.<br><b>Answer from its own weights.</b></h1>
<p class="sub">pre-registered gates &middot; 5 seeds per model &middot; 35 runs, zero failures</p>
<div class="panel" style="margin-top:44px;">
  <div class="vchart" style="position:relative;">
    <div style="position:absolute; left:10px; right:10px; bottom:4px; border-top:3px dashed #A96D12;"></div>
    <div class="vg"><div class="vbars"><div class="vb" style="height:435px; background:#2C7D57;"><span class="val" style="font-size:30px; font-weight:600;">0.75</span></div></div></div>
    <div class="vg"><div class="vbars"><div class="vb" style="height:296px; background:#3D52C9;"><span class="val">0.51</span></div></div></div>
    <div class="vg"><div class="vbars"><div class="vb" style="height:110px; background:#3D52C9;"><span class="val">0.19</span></div></div></div>
    <div class="vg"><div class="vbars"><div class="vb" style="height:61px; background:#3D52C9;"><span class="val">0.11</span></div></div></div>
  </div>
  <div style="display:flex; justify-content:space-around; padding:0 10px; margin-top:14px;">
    <span class="vlab" style="width:200px;">Mistral-7B</span>
    <span class="vlab" style="width:200px;">Llama-8B &#9873;</span>
    <span class="vlab" style="width:200px;">Qwen-7B</span>
    <span class="vlab" style="width:200px;">Qwen-1.5B</span>
  </div>
  <div style="display:flex; justify-content:flex-end; align-items:center; gap:14px; margin-top:24px;">
    <span style="display:inline-block; width:56px; border-top:3px dashed #A96D12;"></span>
    <span class="mono amber" style="font-size:23px;">where we started: 0.006</span>
  </div>
</div>
<p style="font-family:Georgia, serif; font-size:33px; margin-top:36px; line-height:1.4;">Recall rose <b>17&ndash;125&times;</b> on every model &mdash; strongest on a family the recipe never saw in development.</p>
<p class="sub" style="font-style:italic; margin-top:8px;">&#9873; Llama: recall &#10003;, damage control still being tuned &mdash; flagged openly in the paper.</p>
<div class="footer"><span>SLEEP &middot; SP Jain School of Global Management</span><span>3 / 6</span></div>
"""))

# ---- 04: Beats the baselines ------------------------------------------------
IMAGES.append((False, """
<p class="eyebrow">Against the field</p>
<h1 style="font-size:56px;">Over ten learning cycles, SLEEP <b>beats both standard approaches</b>.</h1>
<p class="sub">cumulative recall after 10 rounds of continuous learning &middot; Qwen-7B, both seeds agree</p>
<div class="panel" style="margin-top:40px;">
  <div class="vchart" style="height:380px;">
    <div class="vg"><div class="vbars"><div class="vb" style="height:330px; background:#2C7D57;"><span class="val" style="font-size:30px; font-weight:600;">0.174</span></div></div></div>
    <div class="vg"><div class="vbars"><div class="vb" style="height:106px; background:#8093C9;"><span class="val">0.056</span></div></div></div>
    <div class="vg"><div class="vbars"><div class="vb" style="height:108px; background:#8093C9;"><span class="val">0.057</span></div></div></div>
  </div>
  <div style="display:flex; justify-content:space-around; padding:0 10px; margin-top:14px;">
    <span class="vlab" style="width:260px;"><b>SLEEP</b></span>
    <span class="vlab" style="width:260px;">EWC (standard baseline)</span>
    <span class="vlab" style="width:260px;">naive fine-tuning</span>
  </div>
</div>
<div style="display:flex; flex-direction:column; gap:18px; margin-top:36px;">
  <div class="chip">&#10003;&nbsp; More retained knowledge than naive fine-tuning on 4 of 4 models</div>
  <div class="chip">&#10003;&nbsp; 3&times; the standard continual-learning baseline (EWC)</div>
  <div class="chip">&#10003;&nbsp; Saves small models: damage &times;24 without SLEEP &rarr; &times;2.4 with it</div>
</div>
<div class="footer"><span>SLEEP &middot; SP Jain School of Global Management</span><span>4 / 6</span></div>
"""))

# ---- 05: The breakthrough (dark) --------------------------------------------
IMAGES.append((True, """
<p class="eyebrow">The breakthrough</p>
<h1 style="font-size:62px;">Three discoveries made it work &mdash; <b>each one is reusable</b>.</h1>
<div style="flex:1; display:flex; flex-direction:column; justify-content:center; gap:30px;">
  <div class="row3">
    <svg viewBox="0 0 64 64" width="64" height="64" fill="none" stroke="#8FA0FF" stroke-width="3">
      <rect x="10" y="8" width="44" height="48" rx="3"/><line x1="10" y1="24" x2="54" y2="24"/><line x1="10" y1="40" x2="54" y2="40"/>
      <line x1="26" y1="16" x2="38" y2="16"/><line x1="26" y1="32" x2="38" y2="32"/><line x1="26" y1="48" x2="38" y2="48"/>
    </svg>
    <p><b>Write where facts actually live.</b> Not where the brain-analogy points &mdash; the mid-stack MLP layers, mapped by the model-editing literature.</p>
  </div>
  <div class="row3">
    <svg viewBox="0 0 64 64" width="64" height="64" fill="none" stroke="#8FA0FF" stroke-width="3">
      <path d="M8 14 h32 v20 h-20 l-8 8 v-8 h-4 z"/><path d="M28 34 h28 v18 h-6 v8 l-8-8 h-14 z"/>
    </svg>
    <p><b>Teach the fact, not the sentence.</b> 20+ wordings turn memorization into knowledge the model can actually use.</p>
  </div>
  <div class="row3">
    <svg viewBox="0 0 64 64" width="64" height="64" fill="none" stroke="#8FA0FF" stroke-width="3">
      <path d="M32 6 L54 14 v16 c0 14-9 24-22 28 C19 54 10 44 10 30 V14 z"/><path d="M22 32 l7 7 l13-14"/>
    </svg>
    <p><b>Right-size the safety rails.</b> Calibrated well, they're not a tax &mdash; they're the advantage that keeps learning from destroying the model.</p>
  </div>
</div>
<p style="font-family:Georgia, serif; font-style:italic; font-size:31px; color:#C3CAD9; line-height:1.45;">The meta-lesson for biologically-inspired AI: <b style="font-style:normal; color:#E6EAF4;">mechanisms transfer &mdash; anatomy doesn't.</b></p>
<div class="footer"><span>SLEEP &middot; the localisation recipe</span><span>5 / 6</span></div>
"""))

# ---- 06: What's next -------------------------------------------------------
IMAGES.append((False, """
<p class="eyebrow">What's next</p>
<h1 style="font-size:58px;">Toward AI that <b>genuinely remembers</b> &mdash; assistants that know you, models that learn overnight.</h1>
<div style="flex:1; display:flex; flex-direction:column; justify-content:center; gap:30px;">
  <div class="panel" style="display:flex; align-items:center; gap:36px;">
    <svg viewBox="0 0 64 64" width="88" height="88" fill="none" stroke="#3D52C9" stroke-width="2.6">
      <path d="M16 6 h24 l12 12 v40 h-36 z"/><path d="M40 6 v12 h12"/>
      <line x1="24" y1="30" x2="44" y2="30"/><line x1="24" y1="38" x2="44" y2="38"/><line x1="24" y1="46" x2="36" y2="46"/>
    </svg>
    <div>
      <p style="font-size:34px; font-weight:650;">Updated paper coming soon</p>
      <p style="font-family:Georgia, serif; font-style:italic; font-size:26px; color:#5C6577; margin-top:10px; line-height:1.4;">"Recognition Without Recall: Diagnosing and Repairing Biologically-Inspired Memory Consolidation in Transformers"</p>
    </div>
  </div>
  <div class="panel">
    <p style="font-size:28px; line-height:1.5; color:#3D4658;">Everything is open &mdash; code, pre-registrations, and every result file:<br><b style="color:#1A1F2E;">4 models &middot; 3 families &middot; 95+ seeded runs &middot; all public.</b></p>
  </div>
  <p class="mono indigo" style="font-size:30px; text-align:center;">github.com/Adineu03/sleep-framework</p>
</div>
<div class="footer"><span>Aditya Tripathi &middot; SP Jain School of Global Management</span><span>6 / 6</span></div>
"""))


def main() -> None:
    for i, (dark, body) in enumerate(IMAGES, 1):
        html_path = os.path.join(HTML_DIR, f"{i:02d}.html")
        png_path = os.path.join(HERE, f"{i:02d}.png")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(wrap(body, dark=dark))
        url = "file:///" + html_path.replace("\\", "/")
        subprocess.run([
            EDGE, "--headless", "--disable-gpu", "--hide-scrollbars",
            "--force-device-scale-factor=2",
            f"--screenshot={png_path}", "--window-size=1080,1350", url,
        ], check=True, capture_output=True, timeout=60)
        print(f"rendered {i:02d}.png")


if __name__ == "__main__":
    main()
