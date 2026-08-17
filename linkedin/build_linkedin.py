"""Render the 5-image SLEEP LinkedIn series (1080x1350, 4:5) via Edge headless.

Run:  python build_linkedin.py
Out:  linkedin/01.png ... 05.png  (rendered at 2x: 2160x2700)
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
.vchart { display:flex; align-items:flex-end; justify-content:space-around; height:520px;
  border-bottom:3px solid #1A1F2E; padding:0 10px; }
.vg { display:flex; flex-direction:column; align-items:center; gap:0; width:200px; }
.vbars { display:flex; align-items:flex-end; gap:10px; height:460px; }
.vb { width:64px; border-radius:4px 4px 0 0; position:relative; }
.vb .val { position:absolute; top:-40px; left:50%; transform:translateX(-50%);
  font-family:Consolas, monospace; font-size:24px; white-space:nowrap; color:#1A1F2E; }
.vlab { font-size:23px; color:#3D4658; margin-top:14px; text-align:center; line-height:1.25; }
.legend { display:flex; gap:36px; justify-content:center; margin-top:30px; }
.legend span { font-size:23px; color:#5C6577; display:inline-flex; align-items:center; gap:10px; }
.legend i { width:20px; height:20px; border-radius:4px; display:inline-block; }

.row3 { border:1px solid #2C3349; border-radius:8px; padding:30px 34px; display:flex;
  align-items:center; gap:28px; }
.row3 svg { flex:0 0 64px; }
.row3 p { font-size:29px; line-height:1.45; color:#C3CAD9; }
.row3 p b { color:#E6EAF4; font-weight:650; }
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
<div class="footer"><span>SLEEP &middot; SP Jain School of Global Management</span><span>1 / 5</span></div>
"""))

# ---- 02: What was broken ---------------------------------------------------
IMAGES.append((False, """
<p class="eyebrow">The diagnosis</p>
<h1 style="font-size:56px;">Every model could <b>read</b> the facts.<br>None could <b>recall</b> them.</h1>
<p class="sub">4 models &middot; 3 families &middot; 5 seeds each &middot; the same floor everywhere</p>
<div class="panel" style="margin-top:44px;">
  <div class="vchart">
    <div class="vg"><div class="vbars">
      <div class="vb" style="height:427px; background:#8093C9;"></div>
      <div class="vb" style="height:4px; background:#A96D12;"><span class="val amber">0.007</span></div>
    </div><span class="vlab">Mistral-7B</span></div>
    <div class="vg"><div class="vbars">
      <div class="vb" style="height:454px; background:#8093C9;"></div>
      <div class="vb" style="height:4px; background:#A96D12;"><span class="val amber">0.006</span></div>
    </div><span class="vlab">Llama-8B</span></div>
    <div class="vg"><div class="vbars">
      <div class="vb" style="height:350px; background:#8093C9;"></div>
      <div class="vb" style="height:4px; background:#A96D12;"><span class="val amber">0.006</span></div>
    </div><span class="vlab">Qwen-7B</span></div>
    <div class="vg"><div class="vbars">
      <div class="vb" style="height:322px; background:#8093C9;"></div>
      <div class="vb" style="height:4px; background:#A96D12;"><span class="val amber">0.007</span></div>
    </div><span class="vlab">Qwen-1.5B</span></div>
  </div>
  <div class="legend">
    <span><i style="background:#8093C9;"></i>fact visible in context</span>
    <span><i style="background:#A96D12;"></i>from trained weights</span>
  </div>
</div>
<p style="font-family:Georgia, serif; font-size:33px; margin-top:40px; line-height:1.4;">The dissociation was universal. We called it <b>recognition without recall</b>.</p>
<div class="footer"><span>SLEEP &middot; SP Jain School of Global Management</span><span>2 / 5</span></div>
"""))

# ---- 03: The twist (dark) --------------------------------------------------
IMAGES.append((True, """
<p class="eyebrow">The turn</p>
<h1 style="font-size:66px;">The failure wasn't a law of nature.<br><b>It was a wrong address.</b></h1>
<div style="flex:1; display:flex; flex-direction:column; justify-content:center; gap:30px;">
  <div class="row3">
    <svg viewBox="0 0 64 64" width="64" height="64" fill="none" stroke="#8FA0FF" stroke-width="3">
      <rect x="10" y="8" width="44" height="48" rx="3"/><line x1="10" y1="24" x2="54" y2="24"/><line x1="10" y1="40" x2="54" y2="40"/>
      <line x1="26" y1="16" x2="38" y2="16"/><line x1="26" y1="32" x2="38" y2="32"/><line x1="26" y1="48" x2="38" y2="48"/>
    </svg>
    <p>Facts don't live where the brain-analogy pointed. They live in <b>mid-stack MLP layers</b> &mdash; the model-editing literature had the map.</p>
  </div>
  <div class="row3">
    <svg viewBox="0 0 64 64" width="64" height="64" fill="none" stroke="#8FA0FF" stroke-width="3">
      <path d="M8 14 h32 v20 h-20 l-8 8 v-8 h-4 z"/><path d="M28 34 h28 v18 h-6 v8 l-8-8 h-14 z"/>
    </svg>
    <p>Teach the <b>fact in 20+ wordings</b> &mdash; not the sentence. One wording trains a parrot; many train knowledge.</p>
  </div>
  <div class="row3">
    <svg viewBox="0 0 64 64" width="64" height="64" fill="none" stroke="#8FA0FF" stroke-width="3">
      <path d="M32 6 L54 14 v16 c0 14-9 24-22 28 C19 54 10 44 10 30 V14 z"/><path d="M22 32 l7 7 l13-14"/>
    </svg>
    <p>Our own safety rails were <b>erasing the writes</b>. Right-sized, they became the advantage.</p>
  </div>
</div>
<div class="footer"><span>SLEEP &middot; the localisation repair</span><span>3 / 5</span></div>
"""))

# ---- 04: The proof ---------------------------------------------------------
IMAGES.append((False, """
<p class="eyebrow">The proof</p>
<h1 style="font-size:56px;">Read once. Sleep.<br><b>Answer from its own weights.</b></h1>
<p class="sub">pre-registered gates &middot; 5 seeds per model &middot; 35 runs, zero failures</p>
<div class="panel" style="margin-top:44px;">
  <div class="vchart" style="position:relative; height:480px;">
    <div style="position:absolute; left:10px; right:10px; bottom:4px; border-top:3px dashed #A96D12;"></div>
    <div class="vg"><div class="vbars" style="height:auto;">
      <div class="vb" style="height:435px; width:110px; background:#2C7D57;"><span class="val" style="font-size:30px; font-weight:600;">0.75</span></div>
    </div></div>
    <div class="vg"><div class="vbars" style="height:auto;">
      <div class="vb" style="height:296px; width:110px; background:#3D52C9;"><span class="val">0.51</span></div>
    </div></div>
    <div class="vg"><div class="vbars" style="height:auto;">
      <div class="vb" style="height:110px; width:110px; background:#3D52C9;"><span class="val">0.19</span></div>
    </div></div>
    <div class="vg"><div class="vbars" style="height:auto;">
      <div class="vb" style="height:61px; width:110px; background:#3D52C9;"><span class="val">0.11</span></div>
    </div></div>
  </div>
  <div style="display:flex; justify-content:space-around; padding:0 10px; margin-top:14px;">
    <span class="vlab" style="width:200px;">Mistral-7B</span>
    <span class="vlab" style="width:200px;">Llama-8B &#9873;</span>
    <span class="vlab" style="width:200px;">Qwen-7B</span>
    <span class="vlab" style="width:200px;">Qwen-1.5B</span>
  </div>
  <div style="display:flex; justify-content:flex-end; align-items:center; gap:14px; margin-top:24px;">
    <span style="display:inline-block; width:56px; border-top:3px dashed #A96D12;"></span>
    <span class="mono amber" style="font-size:23px;">the old floor: 0.006</span>
  </div>
</div>
<p style="font-family:Georgia, serif; font-size:33px; margin-top:36px; line-height:1.4;">Recall rose <b>17&ndash;125&times;</b> on every model we tested.</p>
<p class="sub" style="font-style:italic; margin-top:8px;">&#9873; Llama flagged honestly: recall &#10003;, background damage still above our bar.</p>
<div class="footer"><span>SLEEP &middot; SP Jain School of Global Management</span><span>4 / 5</span></div>
"""))

# ---- 05: What's next -------------------------------------------------------
IMAGES.append((False, """
<p class="eyebrow">What's next</p>
<h1 style="font-size:58px;">It also beats the standard continual-learning baseline &mdash; <b>3&times; vs EWC</b> &mdash; over ten learning cycles.</h1>
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
    <p style="font-size:28px; line-height:1.5; color:#3D4658;">The bigger lesson for biologically-inspired ML:<br><b style="color:#1A1F2E;">mechanisms transfer &mdash; anatomy doesn't.</b><br>Copy the brain's algorithms; let the substrate's own structure decide the wiring.</p>
  </div>
  <p class="mono indigo" style="font-size:30px; text-align:center;">github.com/Adineu03/sleep-framework</p>
</div>
<div class="footer"><span>Aditya Tripathi &middot; SP Jain School of Global Management</span><span>5 / 5</span></div>
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
