"""
╔══════════════════════════════════════════════════════════════╗
║        NeuroBee — Heart Disease Prediction AI                ║
║        Flask Web App  →  opens at http://localhost:5000      ║
║                                                              ║
║  pip install flask scikit-learn numpy requests               ║
║                                                              ║
║  1. Set MODEL_PATH below to your .pkl file                   ║
║  2. python neurobee_web.py                                   ║
║  3. Open  http://localhost:5000  in your browser             ║
╚══════════════════════════════════════════════════════════════╝
"""

# ─── CHANGE THIS TO YOUR MODEL .pkl PATH ───────────────────────────────────
MODEL_PATH  = r"C:\Users\penum\PycharmProjects\PythonProject\heart_model.pkl"  # <── PUT YOUR TRAINED MODEL PATH HERE
SCALER_PATH = r"C:\Users\penum\PycharmProjects\PythonProject\heart_scaler.pkl" # <── PUT YOUR SCALER PATH HERE (or "" to skip)
# ───────────────────────────────────────────────────────────────────────────

import os, math, threading, webbrowser, pickle, json
import numpy as np
import requests as req
from flask import Flask, request, jsonify, render_template_string

OPENROUTER_KEY = "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"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
BOT_MODEL      = "anthropic/claude-3-haiku"

# ── Load model ──────────────────────────────────────────────────────────────
MODEL  = None
SCALER = None
try:
    with open(MODEL_PATH, "rb") as f:
        MODEL = pickle.load(f)
    print(f"✅ Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"⚠️  Could not load model ({e}) — using built-in approximation")

try:
    if SCALER_PATH:
        with open(SCALER_PATH, "rb") as f:
            SCALER = pickle.load(f)
        print(f"✅ Scaler loaded from {SCALER_PATH}")
except Exception as e:
    print(f"⚠️  Could not load scaler ({e})")

def builtin_predict(v):
    logit = (-5.5 + 0.028*v[0] + 0.6*v[1] + 0.55*v[2] + 0.018*v[3]
             + 0.003*v[4] + 0.5*v[5] + 0.35*v[6] - 0.022*v[7]
             + 0.75*v[8] + 0.42*v[9] - 0.55*v[10] + 0.72*v[11] + 0.7*v[12])
    return 1.0 / (1.0 + math.exp(-logit))

def run_prediction(values):
    arr = np.array(values).reshape(1, -1)
    if MODEL:
        if SCALER:
            arr = SCALER.transform(arr)
        prob = float(MODEL.predict_proba(arr)[0][1]) if hasattr(MODEL,"predict_proba") else float(MODEL.predict(arr)[0])
        return prob
    return builtin_predict(values)

# ────────────────────────────────────────────────────────────────────────────
#  HTML — The entire frontend (welcome + app)
# ────────────────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>NeuroBee — Heart Disease AI</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet"/>
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg:#04080f;--bg2:#080f1d;--bg3:#0c1628;--card:#0a1422;
  --border:#152540;--b2:#1e3a60;
  --red:#f03e5a;--red2:#ff7088;
  --blue:#2979ff;--blue2:#5c9eff;--blue3:#8db8ff;
  --teal:#00d4b8;--gold:#ffc947;
  --green:#22d97a;--orange:#ff8c42;
  --text:#dde8f8;--text2:#6b90b8;--text3:#3d5a7a;
  --glow-r:rgba(240,62,90,0.4);--glow-b:rgba(41,121,255,0.35);
}
html,body{height:100%;font-family:'DM Sans',sans-serif;background:var(--bg);color:var(--text);overflow-x:hidden}

/* ── NOISE TEXTURE OVERLAY ── */
body::before{content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
  background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.035'/%3E%3C/svg%3E");
  opacity:.4}

/* ═══════════════════════════ WELCOME ═══════════════════════════ */
#welcome{position:fixed;inset:0;z-index:1000;display:flex;flex-direction:column;align-items:center;justify-content:center;
  background:radial-gradient(ellipse 80% 60% at 50% 40%,#0d1f4a 0%,var(--bg) 70%);
  transition:opacity .7s,transform .7s}
#welcome.out{opacity:0;transform:scale(1.04);pointer-events:none}

.grid-canvas{position:absolute;inset:0;opacity:.5}
.pulse-ring{position:absolute;border-radius:50%;border:1px solid rgba(240,62,90,.18);animation:pring 4s ease-out infinite}
.pulse-ring:nth-child(1){width:280px;height:280px;animation-delay:0s}
.pulse-ring:nth-child(2){width:480px;height:480px;animation-delay:1.3s}
.pulse-ring:nth-child(3){width:700px;height:700px;animation-delay:2.6s}
@keyframes pring{0%{transform:scale(.7);opacity:.7}100%{transform:scale(1.5);opacity:0}}

.w-heart{font-size:5rem;line-height:1;animation:hb 1.3s ease-in-out infinite;filter:drop-shadow(0 0 28px var(--red))}
@keyframes hb{0%,100%{transform:scale(1)}15%{transform:scale(1.28)}30%{transform:scale(1)}45%{transform:scale(1.16)}70%{transform:scale(1)}}

.w-logo{font-family:'Syne',sans-serif;font-weight:800;font-size:clamp(2.8rem,8vw,5.5rem);letter-spacing:-.02em;
  background:linear-gradient(135deg,#fff 30%,var(--blue2));-webkit-background-clip:text;-webkit-text-fill-color:transparent;
  background-clip:text;line-height:1;margin:.3em 0 .15em}
.w-sub{font-family:'JetBrains Mono',monospace;font-size:clamp(.75rem,1.5vw,.95rem);letter-spacing:.25em;
  color:var(--text2);text-transform:uppercase;margin-bottom:.5em}
.w-tag{font-size:.9rem;color:var(--text2);max-width:500px;text-align:center;line-height:1.8;margin-bottom:2.5em}

/* ECG SVG */
.ecg-wrap{width:380px;height:56px;margin-bottom:2rem;overflow:hidden;position:relative}
.ecg-wrap svg{position:absolute;left:0;top:0;width:200%;height:100%;animation:ecgScroll 2s linear infinite}
@keyframes ecgScroll{from{transform:translateX(0)}to{transform:translateX(-50%)}}
.ecg-path{fill:none;stroke:var(--red);stroke-width:2;stroke-linecap:round;stroke-linejoin:round}

.w-btn{font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;letter-spacing:.15em;text-transform:uppercase;
  color:var(--blue2);border:1.5px solid var(--blue);background:transparent;padding:1em 3em;cursor:pointer;
  position:relative;overflow:hidden;transition:all .3s;animation:wblink 2s ease-in-out infinite}
.w-btn::before{content:'';position:absolute;inset:0;background:linear-gradient(90deg,transparent,rgba(41,121,255,.15),transparent);
  transform:translateX(-100%);transition:.5s}
.w-btn:hover::before{transform:translateX(100%)}
.w-btn:hover{background:rgba(41,121,255,.1);box-shadow:0 0 40px rgba(41,121,255,.4);transform:scale(1.04)}
@keyframes wblink{0%,100%{opacity:1}50%{opacity:.55}}

.w-footer{position:absolute;bottom:1.5rem;font-family:'JetBrains Mono',monospace;font-size:.7rem;
  letter-spacing:.12em;color:var(--text3);text-transform:uppercase}

/* ═══════════════════════════ APP SHELL ═══════════════════════════ */
#app{display:none;flex-direction:column;min-height:100vh;position:relative;z-index:1}

/* HEADER */
header{display:flex;align-items:center;justify-content:space-between;padding:.9rem 2rem;
  background:rgba(8,15,29,.92);border-bottom:1px solid var(--border);
  backdrop-filter:blur(16px);position:sticky;top:0;z-index:100}
.hdr-logo{font-family:'Syne',sans-serif;font-weight:800;font-size:1.3rem;letter-spacing:-.01em}
.hdr-logo span:first-child{color:#fff}
.hdr-logo span:last-child{color:var(--red)}
.hdr-badge{font-family:'JetBrains Mono',monospace;font-size:.68rem;letter-spacing:.12em;
  color:var(--text2);background:rgba(41,121,255,.08);border:1px solid var(--border);
  padding:.35em 1em;text-transform:uppercase}
.hdr-model{font-family:'JetBrains Mono',monospace;font-size:.7rem;
  color:var(--green);padding:.3em .8em;background:rgba(34,217,122,.08);border:1px solid rgba(34,217,122,.25)}

/* LAYOUT */
.app-body{flex:1;display:grid;grid-template-columns:1fr 420px;gap:1.2rem;
  padding:1.4rem 1.8rem;max-width:1600px;width:100%;margin:0 auto}

/* SECTION TITLE */
.stitle{font-family:'JetBrains Mono',monospace;font-size:.75rem;font-weight:600;
  letter-spacing:.2em;text-transform:uppercase;color:var(--blue2);
  display:flex;align-items:center;gap:.7em;margin-bottom:1rem}
.stitle::after{content:'';flex:1;height:1px;background:linear-gradient(90deg,var(--border),transparent)}

/* CARD */
.card{background:var(--card);border:1px solid var(--border);position:relative;overflow:hidden;padding:1.4rem}
.card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,var(--blue),var(--red),var(--blue));background-size:200%;
  animation:cshimmer 4s linear infinite}
@keyframes cshimmer{to{background-position:200%}}

/* FORM GRID */
.form-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:.9rem}
.fg label{display:block;font-family:'JetBrains Mono',monospace;font-size:.68rem;font-weight:600;
  letter-spacing:.1em;color:var(--text2);text-transform:uppercase;margin-bottom:.35rem}
.fg label em{color:var(--teal);font-style:normal;font-size:.6rem;margin-left:.4em}
.fg input,.fg select{width:100%;background:rgba(255,255,255,.04);border:1px solid var(--border);
  padding:.6rem .8rem;color:var(--text);font-family:'DM Sans',sans-serif;font-size:.95rem;font-weight:500;
  outline:none;transition:border-color .2s,box-shadow .2s}
.fg input:focus,.fg select:focus{border-color:var(--blue);box-shadow:0 0 0 3px rgba(41,121,255,.12)}
.fg select option{background:var(--bg2)}

/* PREDICT BTN */
.predict-btn{width:100%;padding:1.1rem;font-family:'Syne',sans-serif;font-weight:800;font-size:1.05rem;
  letter-spacing:.12em;text-transform:uppercase;color:#fff;
  background:linear-gradient(135deg,#b8102a,var(--red));border:none;cursor:pointer;
  position:relative;overflow:hidden;transition:transform .2s,box-shadow .2s;
  box-shadow:0 6px 28px rgba(240,62,90,.45);margin-top:1.1rem}
.predict-btn:hover{transform:translateY(-2px);box-shadow:0 10px 40px rgba(240,62,90,.6)}
.predict-btn .shine{position:absolute;top:0;left:-120%;width:60%;height:100%;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.25),transparent);
  animation:bshine 3s infinite}
@keyframes bshine{to{left:120%}}

/* ── RESULT ── */
.result-area{margin-top:1.4rem}

.gauge-row{display:flex;align-items:center;gap:2rem;flex-wrap:wrap}
.gauge-wrap{position:relative;width:172px;height:172px;flex-shrink:0}
.gauge-svg{width:100%;height:100%;transform:rotate(-90deg)}
.gauge-bg-c{fill:none;stroke:var(--bg3);stroke-width:14}
.gauge-fill-c{fill:none;stroke-width:14;stroke-linecap:round;
  transition:stroke-dasharray 1.6s cubic-bezier(.4,0,.2,1),stroke .4s}
.gauge-center{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center}
.gauge-pct{font-family:'Syne',sans-serif;font-weight:800;font-size:2.1rem;line-height:1}
.gauge-sub{font-family:'JetBrains Mono',monospace;font-size:.6rem;letter-spacing:.15em;color:var(--text2);margin-top:.25em}

.result-info{flex:1}
.risk-chip{display:inline-block;font-family:'Syne',sans-serif;font-weight:800;font-size:1rem;
  letter-spacing:.08em;padding:.45em 1.4em;margin-bottom:.9rem}
.risk-low {background:rgba(34,217,122,.12);color:var(--green); border:1px solid rgba(34,217,122,.4)}
.risk-mod {background:rgba(255,140,66,.12); color:var(--orange);border:1px solid rgba(255,140,66,.4)}
.risk-high{background:rgba(240,62,90,.12);  color:var(--red);   border:1px solid rgba(240,62,90,.45)}

.result-msg{font-size:.95rem;line-height:1.75;color:var(--text2);margin-bottom:1rem}

/* Vital bars */
.vitals{display:flex;flex-direction:column;gap:.6rem;margin-top:1rem}
.v-row{display:flex;align-items:center;gap:.8rem;font-size:.8rem}
.v-name{width:140px;color:var(--text2);font-weight:500;letter-spacing:.04em;font-family:'JetBrains Mono',monospace;font-size:.7rem}
.v-track{flex:1;height:5px;background:var(--bg3);overflow:hidden}
.v-fill{height:100%;transition:width 1.4s ease}
.v-val{width:80px;text-align:right;font-weight:600;font-size:.8rem;font-family:'JetBrains Mono',monospace}

/* Precaution tags */
.prec-tags{display:flex;flex-wrap:wrap;gap:.45rem;margin-top:.9rem}
.ptag{font-size:.73rem;padding:.3em .85em;background:rgba(255,201,71,.08);
  border:1px solid rgba(255,201,71,.28);color:var(--gold);letter-spacing:.04em}

/* ── RIGHT PANEL — CHAT ── */
.chat-panel{display:flex;flex-direction:column;height:calc(100vh - 78px);position:sticky;top:78px}
.chat-card{display:flex;flex-direction:column;height:100%;background:var(--card);
  border:1px solid var(--border);overflow:hidden}

.chat-hdr{padding:.9rem 1.1rem;background:rgba(8,15,29,.9);border-bottom:1px solid var(--border);
  display:flex;align-items:center;gap:.8rem;flex-shrink:0}
.chat-avatar{width:38px;height:38px;background:linear-gradient(135deg,var(--blue),var(--red));
  display:flex;align-items:center;justify-content:center;font-size:1.15rem;flex-shrink:0}
.chat-meta h3{font-family:'Syne',sans-serif;font-size:.82rem;font-weight:700;letter-spacing:.06em;color:var(--blue2)}
.chat-meta p{font-size:.7rem;color:var(--text2);margin-top:.1em;font-family:'JetBrains Mono',monospace}
.chat-dot{margin-left:auto;width:8px;height:8px;background:var(--green);animation:cdot 2s infinite}
@keyframes cdot{50%{opacity:.3}}

/* Quick chips */
.quick-bar{padding:.6rem .8rem;border-bottom:1px solid rgba(21,37,64,.6);display:flex;flex-wrap:wrap;gap:.35rem;flex-shrink:0}
.qchip{font-size:.68rem;letter-spacing:.04em;padding:.3em .75em;border:1px solid var(--border);
  color:var(--text2);cursor:pointer;background:transparent;font-family:'JetBrains Mono',monospace;
  transition:all .2s}
.qchip:hover{border-color:var(--blue2);color:var(--blue2);background:rgba(41,121,255,.08)}

/* Messages */
.chat-msgs{flex:1;overflow-y:auto;padding:.9rem;display:flex;flex-direction:column;gap:.7rem;scroll-behavior:smooth}
.chat-msgs::-webkit-scrollbar{width:3px}
.chat-msgs::-webkit-scrollbar-thumb{background:var(--border)}

.msg{max-width:90%;font-size:.85rem;line-height:1.65;padding:.7rem .95rem;animation:msgin .25s ease}
@keyframes msgin{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
.msg.bot{background:rgba(41,121,255,.07);border:1px solid rgba(41,121,255,.18);align-self:flex-start;border-radius:2px 10px 10px 10px}
.msg.user{background:rgba(240,62,90,.09);border:1px solid rgba(240,62,90,.2);align-self:flex-end;border-radius:10px 2px 10px 10px}
.typing-dot{display:inline-block;width:7px;height:7px;background:var(--blue2);border-radius:50%;margin:0 2px;
  animation:tdot 1.1s infinite}
.typing-dot:nth-child(2){animation-delay:.18s}
.typing-dot:nth-child(3){animation-delay:.36s}
@keyframes tdot{0%,80%,100%{transform:translateY(0)}40%{transform:translateY(-7px)}}

/* Chat input */
.chat-input-row{padding:.7rem;border-top:1px solid var(--border);display:flex;gap:.5rem;
  background:rgba(8,15,29,.8);flex-shrink:0}
#chat-in{flex:1;padding:.6rem .85rem;background:rgba(255,255,255,.04);border:1px solid var(--border);
  color:var(--text);font-family:'DM Sans',sans-serif;font-size:.9rem;outline:none;resize:none;
  transition:border-color .2s}
#chat-in:focus{border-color:var(--blue)}
#chat-in::placeholder{color:var(--text3)}
.send-btn{width:40px;height:40px;background:linear-gradient(135deg,var(--blue),#1050b8);
  border:none;cursor:pointer;color:#fff;font-size:1rem;transition:transform .2s,box-shadow .2s;flex-shrink:0}
.send-btn:hover{transform:scale(1.1);box-shadow:0 4px 18px rgba(41,121,255,.55)}

/* Disclaimer bar */
.disclaimer{padding:.5rem 2rem;font-family:'JetBrains Mono',monospace;font-size:.62rem;
  letter-spacing:.1em;color:var(--text3);background:var(--bg2);border-top:1px solid var(--border);
  text-align:center;text-transform:uppercase;position:sticky;bottom:0}

/* Scrollbar global */
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--border)}

@media(max-width:980px){
  .app-body{grid-template-columns:1fr}
  .chat-panel{height:520px;position:relative;top:0}
}
</style>
</head>
<body>

<!-- ══════════════════════════════ WELCOME ══════════════════════════════ -->
<div id="welcome">
  <canvas class="grid-canvas" id="gridCanvas"></canvas>
  <div class="pulse-ring"></div>
  <div class="pulse-ring"></div>
  <div class="pulse-ring"></div>

  <div class="w-heart">❤️</div>

  <div class="w-logo">NeuroBee</div>
  <div class="w-sub">Heart Disease Prediction AI</div>

  <div class="ecg-wrap">
    <svg viewBox="0 0 760 56" preserveAspectRatio="none">
      <!-- two copies for seamless loop -->
      <polyline class="ecg-path"
        points="0,28 40,28 55,28 68,6 80,52 92,28 120,28 134,18 148,28 190,28
                210,28 225,28 238,6 250,52 262,28 290,28 304,18 318,28 360,28
                380,28 395,28 408,6 420,52 432,28 460,28 474,18 488,28 530,28
                550,28 565,28 578,6 590,52 602,28 630,28 644,18 658,28 700,28 760,28"/>
    </svg>
  </div>

  <div class="w-tag">Advanced AI-powered cardiovascular risk assessment.<br>Personalised cardiac guidance from our heart health AI assistant.</div>

  <button class="w-btn" onclick="enterApp()">▶ &nbsp;Press Enter to Begin</button>
  <div class="w-footer">NeuroBee · Cardiac Intelligence System · 2025</div>
</div>

<!-- ══════════════════════════════ APP ══════════════════════════════ -->
<div id="app">
  <header>
    <div class="hdr-logo"><span>Neuro</span><span>Bee</span>&nbsp;<span style="font-size:.65rem;color:var(--text2);font-family:'JetBrains Mono',monospace;font-weight:400;letter-spacing:.15em">HEART AI</span></div>
    <div style="display:flex;gap:.8rem;align-items:center">
      <div class="hdr-model" id="model-badge">● LOADING</div>
      <div class="hdr-badge">🔒 Cardiac Intelligence System</div>
    </div>
  </header>

  <div class="app-body">

    <!-- ── LEFT COLUMN ── -->
    <div style="display:flex;flex-direction:column;gap:1.2rem">

      <!-- INPUT FORM -->
      <div class="card">
        <div class="stitle">📋 Patient Data Input</div>
        <div class="form-grid" id="formGrid">

          <div class="fg">
            <label>Age <em>years</em></label>
            <input type="number" id="f_age" min="1" max="120" placeholder="e.g. 54"/>
          </div>
          <div class="fg">
            <label>Sex</label>
            <select id="f_sex">
              <option value="1">Male</option>
              <option value="0">Female</option>
            </select>
          </div>
          <div class="fg">
            <label>Chest Pain Type</label>
            <select id="f_cp">
              <option value="0">Typical Angina</option>
              <option value="1">Atypical Angina</option>
              <option value="2">Non-anginal Pain</option>
              <option value="3">Asymptomatic</option>
            </select>
          </div>
          <div class="fg">
            <label>Resting BP <em>mmHg</em></label>
            <input type="number" id="f_trestbps" min="50" max="250" placeholder="e.g. 130"/>
          </div>
          <div class="fg">
            <label>Cholesterol <em>mg/dl</em></label>
            <input type="number" id="f_chol" min="50" max="700" placeholder="e.g. 245"/>
          </div>
          <div class="fg">
            <label>Fasting Blood Sugar</label>
            <select id="f_fbs">
              <option value="0">≤ 120 mg/dl (Normal)</option>
              <option value="1">&gt; 120 mg/dl (High)</option>
            </select>
          </div>
          <div class="fg">
            <label>Resting ECG</label>
            <select id="f_restecg">
              <option value="0">Normal</option>
              <option value="1">ST-T Abnormality</option>
              <option value="2">LV Hypertrophy</option>
            </select>
          </div>
          <div class="fg">
            <label>Max Heart Rate <em>bpm</em></label>
            <input type="number" id="f_thalach" min="50" max="250" placeholder="e.g. 150"/>
          </div>
          <div class="fg">
            <label>Exercise Angina</label>
            <select id="f_exang">
              <option value="0">No</option>
              <option value="1">Yes</option>
            </select>
          </div>
          <div class="fg">
            <label>ST Depression <em>mm</em></label>
            <input type="number" id="f_oldpeak" step="0.1" min="0" max="10" placeholder="e.g. 1.5"/>
          </div>
          <div class="fg">
            <label>ST Slope</label>
            <select id="f_slope">
              <option value="0">Upsloping</option>
              <option value="1">Flat</option>
              <option value="2">Downsloping</option>
            </select>
          </div>
          <div class="fg">
            <label>Major Vessels <em>0–3</em></label>
            <select id="f_ca">
              <option value="0">0</option>
              <option value="1">1</option>
              <option value="2">2</option>
              <option value="3">3</option>
            </select>
          </div>
          <div class="fg" style="grid-column:1/-1;max-width:33%">
            <label>Thalassemia</label>
            <select id="f_thal">
              <option value="1">Normal</option>
              <option value="2">Fixed Defect</option>
              <option value="3">Reversible Defect</option>
            </select>
          </div>

        </div><!-- /form-grid -->

        <button class="predict-btn" onclick="runPrediction()">
          <span class="shine"></span>
          ⚡ &nbsp;Analyse &amp; Predict Risk
        </button>
      </div><!-- /card -->

      <!-- RESULT -->
      <div class="card" id="resultCard" style="display:none">
        <div class="stitle">📊 Risk Analysis Result</div>
        <div class="result-area">
          <div class="gauge-row">
            <!-- Gauge -->
            <div class="gauge-wrap">
              <svg class="gauge-svg" viewBox="0 0 160 160">
                <circle class="gauge-bg-c" cx="80" cy="80" r="62"/>
                <circle class="gauge-fill-c" id="gaugeFill" cx="80" cy="80" r="62"
                        stroke-dasharray="0 389.6"/>
              </svg>
              <div class="gauge-center">
                <div class="gauge-pct" id="gaugePct" style="color:var(--green)">0%</div>
                <div class="gauge-sub">RISK SCORE</div>
              </div>
            </div>
            <!-- Info -->
            <div class="result-info">
              <div class="risk-chip" id="riskChip">—</div>
              <div class="result-msg" id="resultMsg"></div>
              <div class="prec-tags" id="precTags"></div>
            </div>
          </div>
          <!-- Vital bars -->
          <div class="vitals" id="vitals"></div>
        </div>
      </div>

    </div><!-- /left -->

    <!-- ── RIGHT COLUMN — CHAT ── -->
    <div class="chat-panel">
      <div class="chat-card">

        <div class="chat-hdr">
          <div class="chat-avatar">🤖</div>
          <div class="chat-meta">
            <h3>CardioAI Assistant</h3>
            <p>Heart Health Specialist · Online</p>
          </div>
          <div class="chat-dot"></div>
        </div>

        <div class="quick-bar">
          <button class="qchip" onclick="qsend('What are the main symptoms of heart disease?')">Symptoms</button>
          <button class="qchip" onclick="qsend('What should I eat for a healthy heart?')">Heart Diet</button>
          <button class="qchip" onclick="qsend('What exercises are safe for my heart?')">Exercise</button>
          <button class="qchip" onclick="qsend('Explain my risk result in detail')">My Result</button>
          <button class="qchip" onclick="qsend('What is cholesterol and why does it matter?')">Cholesterol</button>
          <button class="qchip" onclick="qsend('How to lower blood pressure naturally?')">Lower BP</button>
          <button class="qchip" onclick="qsend('What are warning signs of a heart attack?')">Warning Signs</button>
          <button class="qchip" onclick="qsend('How does stress affect the heart?')">Stress &amp; Heart</button>
        </div>

        <div class="chat-msgs" id="chatMsgs">
          <div class="msg bot">
            👋 Hello! I'm <strong>CardioAI</strong> — your cardiac health assistant.<br><br>
            Fill in the patient form on the left and click <strong>Analyse &amp; Predict Risk</strong>.
            I'll analyse your results and provide personalised advice.<br><br>
            <em>I only answer heart-related questions. Ask me anything!</em> 🫀
          </div>
        </div>

        <div class="chat-input-row">
          <textarea id="chat-in" rows="1"
            placeholder="Ask about your heart health or results…"
            onkeydown="chatKey(event)"></textarea>
          <button class="send-btn" onclick="sendChat()">➤</button>
        </div>

      </div>
    </div>

  </div><!-- /app-body -->

  <div class="disclaimer">⚕ For informational purposes only — not a clinical diagnosis · Always consult a qualified cardiologist</div>
</div><!-- /app -->

<script>
// ───────────────────────────────────────────────────
//  STATE
// ───────────────────────────────────────────────────
let latestResult  = null;
let chatHistory   = [];
let gaugeAnimId   = null;
let gaugeCurrent  = 0;

// ───────────────────────────────────────────────────
//  WELCOME GRID CANVAS
// ───────────────────────────────────────────────────
(function(){
  const c = document.getElementById('gridCanvas');
  const ctx = c.getContext('2d');
  function resize(){ c.width=innerWidth; c.height=innerHeight; draw(); }
  function draw(){
    ctx.clearRect(0,0,c.width,c.height);
    ctx.strokeStyle='rgba(30,58,96,0.5)'; ctx.lineWidth=1;
    for(let x=0;x<c.width;x+=44){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,c.height);ctx.stroke();}
    for(let y=0;y<c.height;y+=44){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(c.width,y);ctx.stroke();}
  }
  window.addEventListener('resize',resize); resize();
})();

// ───────────────────────────────────────────────────
//  WELCOME → APP
// ───────────────────────────────────────────────────
function enterApp(){
  document.getElementById('welcome').classList.add('out');
  setTimeout(()=>{
    document.getElementById('welcome').style.display='none';
    document.getElementById('app').style.display='flex';
    checkModelStatus();
  },700);
}
document.addEventListener('keydown', e=>{
  if(e.key==='Enter' && !document.getElementById('welcome').classList.contains('out'))
    enterApp();
});

async function checkModelStatus(){
  try{
    const r = await fetch('/api/status');
    const d = await r.json();
    const badge = document.getElementById('model-badge');
    if(d.model_loaded){
      badge.textContent='● REAL MODEL'; badge.style.color='var(--green)';
      badge.style.background='rgba(34,217,122,.08)'; badge.style.borderColor='rgba(34,217,122,.25)';
    } else {
      badge.textContent='● APPROX MODE'; badge.style.color='var(--orange)';
    }
  }catch(e){}
}

// ───────────────────────────────────────────────────
//  PREDICTION
// ───────────────────────────────────────────────────
async function runPrediction(){
  const ids=['f_age','f_sex','f_cp','f_trestbps','f_chol','f_fbs',
             'f_restecg','f_thalach','f_exang','f_oldpeak','f_slope','f_ca','f_thal'];
  const values = [];
  for(const id of ids){
    const v = document.getElementById(id).value;
    if(v===''||v===null){alert('⚠️  Please fill in all fields.');return;}
    values.push(parseFloat(v));
  }

  const btn = document.querySelector('.predict-btn');
  btn.textContent = '⏳ Analysing…';
  btn.disabled = true;

  try{
    const resp = await fetch('/api/predict',{
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({values})
    });
    const data = await resp.json();
    showResult(data, values);
    botAutoMessage(data);
  }catch(e){
    alert('Prediction failed: '+e);
  } finally {
    btn.innerHTML='<span class="shine"></span>⚡ &nbsp;Analyse &amp; Predict Risk';
    btn.disabled=false;
  }
}

function showResult(data, values){
  const {pct, risk, color} = data;
  document.getElementById('resultCard').style.display='block';
  document.getElementById('resultCard').scrollIntoView({behavior:'smooth',block:'nearest'});

  // Gauge animation
  const circ = 2*Math.PI*62;
  const fill  = document.getElementById('gaugeFill');
  const pctEl = document.getElementById('gaugePct');
  gaugeCurrent = 0;
  clearInterval(gaugeAnimId);
  gaugeAnimId = setInterval(()=>{
    const step = (pct - gaugeCurrent)*0.12;
    gaugeCurrent += Math.abs(step)<0.3 ? (pct-gaugeCurrent) : step;
    const arc = (gaugeCurrent/100)*circ;
    fill.setAttribute('stroke-dasharray',`${arc} ${circ}`);
    fill.setAttribute('stroke', color);
    pctEl.textContent = Math.round(gaugeCurrent)+'%';
    pctEl.style.color = color;
    if(Math.abs(gaugeCurrent-pct)<0.3){ gaugeCurrent=pct; clearInterval(gaugeAnimId); }
  },16);

  // Badge
  const chip = document.getElementById('riskChip');
  chip.textContent = risk+' RISK';
  chip.className = 'risk-chip risk-'+risk.toLowerCase();

  // Message
  const msgs = {
    LOW: `Your cardiovascular risk score is <strong>${pct}%</strong>. Your heart health indicators look stable. Keep maintaining your healthy lifestyle!`,
    MODERATE: `Your cardiovascular risk score is <strong>${pct}%</strong>. Some indicators suggest moderate cardiac risk. Consider lifestyle changes and consult your doctor.`,
    HIGH: `Your cardiovascular risk score is <strong>${pct}%</strong>. ⚠️ Significant cardiac risk detected. Please seek immediate medical evaluation from a cardiologist.`
  };
  document.getElementById('resultMsg').innerHTML = msgs[risk];

  // Precaution tags
  const precs={
    LOW:      ['✅ Regular Exercise','🥗 Heart Diet','🚭 No Smoking','😴 Quality Sleep','💧 Stay Hydrated'],
    MODERATE: ['💊 Consult Doctor','🏃 Moderate Exercise','🧂 Reduce Sodium','📉 Manage Stress','🩺 Annual Check-up'],
    HIGH:     ['🚨 See Cardiologist ASAP','💊 Medication Review','🏥 ECG & Tests','🚫 Avoid Exertion','🩸 Full Lipid Panel','📵 Quit Smoking Now']
  };
  document.getElementById('precTags').innerHTML =
    precs[risk].map(p=>`<span class="ptag">${p}</span>`).join('');

  // Vital bars
  const vitals=[
    {n:'Blood Pressure', v:values[3], min:80, max:200, u:'mmHg', warn:130},
    {n:'Cholesterol',    v:values[4], min:100,max:600, u:'mg/dl',warn:200},
    {n:'Max Heart Rate', v:values[7], min:60, max:220, u:'bpm',  warn:170},
    {n:'ST Depression',  v:values[9], min:0,  max:6,   u:'mm',   warn:2},
  ];
  document.getElementById('vitals').innerHTML = vitals.map(vt=>{
    const norm = Math.min(100,Math.max(0,(vt.v-vt.min)/(vt.max-vt.min)*100));
    const vc   = vt.v>vt.warn?'var(--orange)':'var(--green)';
    return `<div class="v-row">
      <div class="v-name">${vt.n}</div>
      <div class="v-track"><div class="v-fill" style="width:${norm}%;background:${vc}"></div></div>
      <div class="v-val" style="color:${vc}">${vt.v}<small style="color:var(--text2);font-size:.65rem"> ${vt.u}</small></div>
    </div>`;
  }).join('');

  latestResult = {pct, risk, values, color};
}

function botAutoMessage(data){
  const {pct, risk} = data;
  const msgs={
    LOW:      `🔬 Analysis complete! Your risk score is <strong>${pct}%</strong> — <strong>LOW</strong>.\n\nGreat news! Your indicators look healthy. I can explain any factor in detail or answer any heart health question.`,
    MODERATE: `🔬 Analysis complete! Your risk score is <strong>${pct}%</strong> — <strong>MODERATE</strong>.\n\nThere are some areas to monitor. Ask me about any of your specific indicators and I'll explain what they mean and what you can do.`,
    HIGH:     `🔬 Analysis complete! Your risk score is <strong>${pct}%</strong> — <strong>HIGH RISK</strong>.\n\n⚠️ Please consult a cardiologist as soon as possible. I can explain each risk factor and help you understand what steps to take next. Just ask.`
  };
  appendBot(msgs[risk]);
}

// ───────────────────────────────────────────────────
//  CHAT
// ───────────────────────────────────────────────────
function qsend(text){
  document.getElementById('chat-in').value=text;
  sendChat();
}

function chatKey(e){
  if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendChat();}
}

async function sendChat(){
  const inp = document.getElementById('chat-in');
  const text = inp.value.trim();
  if(!text) return;
  inp.value='';
  appendUser(text);
  chatHistory.push({role:'user',content:text});
  const tid = appendTyping();
  try{
    const resp = await fetch('/api/chat',{
      method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({messages:chatHistory, result:latestResult})
    });
    const data = await resp.json();
    removeTyping(tid);
    const reply = data.reply || '⚠️ No response.';
    chatHistory.push({role:'assistant',content:reply});
    appendBot(reply);
  }catch(e){
    removeTyping(tid);
    appendBot('⚠️ Connection error. Please try again.');
  }
}

let typingN=0;
function appendUser(t){
  const d=document.createElement('div');
  d.className='msg user'; d.textContent=t;
  document.getElementById('chatMsgs').appendChild(d);
  scrollChat();
}
function appendBot(html){
  const d=document.createElement('div');
  d.className='msg bot'; d.innerHTML=html.replace(/\n/g,'<br>');
  document.getElementById('chatMsgs').appendChild(d);
  scrollChat();
}
function appendTyping(){
  const id='t'+(++typingN);
  const d=document.createElement('div');
  d.className='msg bot'; d.id=id;
  d.innerHTML='<span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>';
  document.getElementById('chatMsgs').appendChild(d);
  scrollChat(); return id;
}
function removeTyping(id){const e=document.getElementById(id);if(e)e.remove();}
function scrollChat(){const c=document.getElementById('chatMsgs');c.scrollTop=c.scrollHeight;}
</script>
</body>
</html>"""

# ────────────────────────────────────────────────────────────────────────────
#  FLASK ROUTES
# ────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/api/status")
def status():
    return jsonify({"model_loaded": MODEL is not None, "scaler_loaded": SCALER is not None})

@app.route("/api/predict", methods=["POST"])
def predict():
    data   = request.json
    values = data.get("values", [])
    if len(values) != 13:
        return jsonify({"error": "Expected 13 feature values"}), 400
    try:
        prob  = run_prediction(values)
        pct   = round(prob * 100)
        risk  = "LOW" if pct < 35 else ("MODERATE" if pct < 65 else "HIGH")
        color = {"LOW":"#22d97a","MODERATE":"#ff8c42","HIGH":"#f03e5a"}[risk]
        return jsonify({"prob": prob, "pct": pct, "risk": risk, "color": color})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/chat", methods=["POST"])
def chat():
    body     = request.json
    messages = body.get("messages", [])[-16:]
    result   = body.get("result")

    # Build system prompt
    system = (
        "You are CardioAI, an expert cardiac health assistant inside the "
        "NeuroBee Heart Disease Prediction platform. "
        "STRICT RULE: You ONLY answer questions about heart disease, cardiovascular health, "
        "cardiology, ECG, blood pressure, cholesterol, cardiac medications, heart-healthy diet, "
        "and cardiac exercise. If the user asks about anything unrelated to heart health, "
        "politely decline and redirect them. Be warm, medically accurate, and use simple language. "
        "Always advise consulting a real cardiologist for serious concerns. "
        "Write in short paragraphs without heavy markdown."
    )
    if result:
        v   = result.get("values", [])
        cp  = {0:"Typical Angina",1:"Atypical Angina",2:"Non-anginal Pain",3:"Asymptomatic"}
        system += (
            f"\n\n--- PATIENT PREDICTION ---\n"
            f"Risk Level: {result.get('risk')} | Score: {result.get('pct')}%\n"
            f"Age: {int(v[0]) if v else '?'} | Sex: {'Male' if (v[1] if v else 0)==1 else 'Female'}\n"
            f"Chest Pain: {cp.get(int(v[2]),v[2]) if v else '?'}\n"
            f"Resting BP: {v[3] if v else '?'} mmHg | Cholesterol: {v[4] if v else '?'} mg/dl\n"
            f"Max HR: {v[7] if v else '?'} bpm | ST Depression: {v[9] if v else '?'} mm\n"
            f"Exercise Angina: {'Yes' if (v[8] if v else 0)==1 else 'No'}\n"
            "Refer to these values when discussing the patient's results."
        )

    try:
        resp = req.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_KEY}",
                "Content-Type":  "application/json",
                "HTTP-Referer":  "https://neurobee.app",
                "X-Title":       "NeuroBee Heart AI",
            },
            json={
                "model":       BOT_MODEL,
                "messages":    [{"role":"system","content":system}] + messages,
                "max_tokens":  700,
                "temperature": 0.65,
            },
            timeout=35,
        )
        data  = resp.json()
        reply = data["choices"][0]["message"]["content"]
    except Exception as e:
        reply = f"⚠️ Error contacting AI: {str(e)[:120]}"

    return jsonify({"reply": reply})

# ────────────────────────────────────────────────────────────────────────────
#  LAUNCH
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    PORT = 5000
    print("\n" + "═"*58)
    print("  🫀  NeuroBee Heart Disease Prediction AI")
    print("═"*58)
    print(f"  🌐  Opening at  →  http://localhost:{PORT}")
    print(f"  🧠  Model       →  {'LOADED ✅' if MODEL else 'Not found (approximation mode)'}")
    print(f"  ⚖️   Scaler      →  {'LOADED ✅' if SCALER else 'Not found'}")
    print("═"*58)
    print("  Press Ctrl+C to stop the server\n")
    # Auto-open browser after 1.2 s
    threading.Timer(1.2, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()
    app.run(host="0.0.0.0", port=PORT, debug=False)