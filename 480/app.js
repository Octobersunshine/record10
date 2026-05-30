const express = require("express");
const { Worker } = require("worker_threads");
const path = require("path");
const { generateRegexFromSamples } = require("./regex-generator");

const app = express();
app.use(express.json());

const REGEX_TIMEOUT_MS = 2000;

function runRegexWithTimeout(pattern, testString, flagStr) {
  return new Promise((resolve, reject) => {
    const worker = new Worker(path.join(__dirname, "regex-worker.js"), {
      workerData: { pattern, testString, flagStr },
    });

    const timeoutId = setTimeout(() => {
      worker.terminate();
      reject(new Error("TIMEOUT"));
    }, REGEX_TIMEOUT_MS);

    worker.on("message", (result) => {
      clearTimeout(timeoutId);
      if (result.success) {
        resolve(result.matches);
      } else {
        reject(new Error(result.error));
      }
    });

    worker.on("error", (err) => {
      clearTimeout(timeoutId);
      reject(err);
    });

    worker.on("exit", (code) => {
      clearTimeout(timeoutId);
      if (code !== 0) {
        reject(new Error(`Worker exited with code ${code}`));
      }
    });
  });
}

const FLAG_MAP = {
  IGNORECASE: "i",
  MULTILINE: "m",
  DOTALL: "s",
  VERBOSE: "x",
  UNICODE: "u",
  STICKY: "y",
  GLOBAL: "g",
};

function buildFlags(flagsList) {
  let flagStr = "";
  for (const f of flagsList) {
    const upper = f.toUpperCase();
    if (FLAG_MAP[upper]) {
      flagStr += FLAG_MAP[upper];
    }
  }
  if (!flagStr.includes("g")) {
    flagStr += "g";
  }
  return flagStr;
}

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

app.post("/api/regex/test", async (req, res) => {
  const { pattern, test_string, flags = [] } = req.body || {};

  if (pattern === undefined) {
    return res.status(400).json({ error: "缺少必填字段: pattern" });
  }
  if (test_string === undefined) {
    return res.status(400).json({ error: "缺少必填字段: test_string" });
  }

  const flagStr = buildFlags(flags);

  try {
    new RegExp(pattern, flagStr);
  } catch (e) {
    return res.status(400).json({ error: `正则表达式编译失败: ${e.message}` });
  }

  try {
    const matches = await runRegexWithTimeout(pattern, test_string, flagStr);
    res.json({
      pattern,
      test_string,
      flags,
      match_count: matches.length,
      matches,
    });
  } catch (e) {
    if (e.message === "TIMEOUT") {
      return res.status(400).json({
        error: "正则过于复杂，执行超时（超过 2 秒），可能存在灾难性回溯",
        timeout: true,
      });
    }
    return res.status(400).json({ error: `正则表达式执行失败: ${e.message}` });
  }
});

app.post("/api/regex/validate", (req, res) => {
  const { pattern, flags = [] } = req.body || {};

  if (pattern === undefined) {
    return res.status(400).json({ error: "缺少必填字段: pattern" });
  }

  const flagStr = buildFlags(flags);

  try {
    new RegExp(pattern, flagStr);
    res.json({ valid: true, pattern });
  } catch (e) {
    res.json({ valid: false, pattern, error: e.message });
  }
});

app.post("/api/regex/generate", (req, res) => {
  const { positives = [], negatives = [], max_candidates = 10 } = req.body || {};

  if (!Array.isArray(positives) || positives.length === 0) {
    return res.status(400).json({ error: "请至少提供一个正例样本（应该匹配的字符串）" });
  }

  if (!Array.isArray(negatives)) {
    return res.status(400).json({ error: "反例必须是字符串数组" });
  }

  for (const p of positives) {
    if (typeof p !== "string") {
      return res.status(400).json({ error: "所有样本必须是字符串" });
    }
  }
  for (const n of negatives) {
    if (typeof n !== "string") {
      return res.status(400).json({ error: "所有样本必须是字符串" });
    }
  }

  try {
    const result = generateRegexFromSamples({
      positives,
      negatives,
      maxCandidates: max_candidates,
    });
    res.json(result);
  } catch (e) {
    res.status(500).json({ error: `生成失败: ${e.message}` });
  }
});

app.get("/api/health", (_req, res) => {
  res.json({ status: "ok" });
});

app.get("/", (_req, res) => {
  res.send(`<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>正则表达式测试工具</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #0f172a; color: #e2e8f0; min-height: 100vh; padding: 32px;
  }
  .container { max-width: 1000px; margin: 0 auto; }
  h1 {
    font-size: 28px; font-weight: 700; margin-bottom: 8px;
    background: linear-gradient(135deg, #38bdf8, #818cf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .subtitle { color: #94a3b8; font-size: 14px; margin-bottom: 24px; }
  .tabs { display: flex; gap: 4px; margin-bottom: 20px; }
  .tab {
    padding: 10px 20px; background: #1e293b; border: 1px solid #334155; border-bottom: none;
    border-radius: 10px 10px 0 0; cursor: pointer; font-size: 14px; font-weight: 500;
    color: #94a3b8; transition: all .2s;
  }
  .tab:hover { color: #e2e8f0; }
  .tab.active {
    background: #1e293b; color: #38bdf8; border-color: #38bdf8;
    box-shadow: 0 -2px 10px rgba(56, 189, 248, 0.15);
  }
  .tab-content { display: none; }
  .tab-content.active { display: block; }
  .card {
    background: #1e293b; border: 1px solid #334155; border-radius: 0 12px 12px 12px;
    padding: 24px; margin-bottom: 20px;
  }
  label { display: block; font-size: 13px; color: #94a3b8; margin-bottom: 6px; font-weight: 500; }
  input[type="text"], textarea {
    width: 100%; background: #0f172a; border: 1px solid #334155; border-radius: 8px;
    color: #e2e8f0; padding: 10px 14px; font-size: 15px; font-family: 'Cascadia Code', 'Fira Code', monospace;
    outline: none; transition: border-color .2s;
  }
  input:focus, textarea:focus { border-color: #38bdf8; }
  textarea { min-height: 100px; resize: vertical; }
  .row { display: flex; gap: 16px; }
  .row > * { flex: 1; }
  .flags { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 4px; }
  .flags label {
    display: flex; align-items: center; gap: 6px; cursor: pointer;
    font-size: 13px; color: #cbd5e1; font-weight: 400; margin-bottom: 0;
  }
  .flags input[type="checkbox"] { accent-color: #38bdf8; }
  .btn {
    background: linear-gradient(135deg, #38bdf8, #818cf8);
    color: #fff; border: none; border-radius: 8px; padding: 10px 28px;
    font-size: 15px; font-weight: 600; cursor: pointer; margin-top: 16px;
    transition: opacity .2s;
  }
  .btn:hover { opacity: 0.9; }
  .btn:active { opacity: 0.8; }
  .btn-secondary {
    background: transparent; border: 1px solid #38bdf8; color: #38bdf8;
  }
  .btn-secondary:hover { background: #38bdf810; }
  .btn-small { padding: 6px 12px; font-size: 12px; margin-top: 0; }
  .result-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
  .result-header h3 { font-size: 16px; color: #e2e8f0; }
  .match-count {
    background: #38bdf820; color: #38bdf8; padding: 4px 12px;
    border-radius: 20px; font-size: 13px; font-weight: 600;
  }
  .match-item {
    background: #0f172a; border: 1px solid #334155; border-radius: 8px;
    padding: 14px 16px; margin-bottom: 10px;
  }
  .match-item-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
  .match-value {
    font-family: 'Cascadia Code', 'Fira Code', monospace;
    color: #34d399; font-size: 14px; word-break: break-all;
  }
  .match-pos { color: #64748b; font-size: 12px; font-family: monospace; }
  .groups-section { margin-top: 8px; }
  .groups-title { font-size: 12px; color: #64748b; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.5px; }
  .group-row {
    display: flex; gap: 12px; align-items: baseline; padding: 3px 0;
    font-size: 13px; font-family: monospace;
  }
  .group-index { color: #818cf8; min-width: 60px; }
  .group-value { color: #fbbf24; word-break: break-all; }
  .group-pos { color: #64748b; font-size: 12px; }
  .error {
    background: #7f1d1d40; border: 1px solid #dc2626; border-radius: 8px;
    padding: 14px 16px; color: #fca5a5; font-size: 14px;
  }
  .highlight-container {
    background: #0f172a; border: 1px solid #334155; border-radius: 8px;
    padding: 14px 16px; font-family: 'Cascadia Code', 'Fira Code', monospace;
    font-size: 14px; line-height: 1.8; word-break: break-all; white-space: pre-wrap;
  }
  .highlight-container mark {
    background: #38bdf830; color: #38bdf8; border-bottom: 2px solid #38bdf8;
    padding: 1px 0; border-radius: 2px;
  }
  .no-match { color: #64748b; font-style: italic; }
  .api-hint {
    margin-top: 24px; padding: 16px 20px; background: #1e293b; border: 1px solid #334155;
    border-radius: 12px; font-size: 13px; color: #94a3b8; line-height: 1.8;
  }
  .api-hint code {
    background: #0f172a; padding: 2px 6px; border-radius: 4px; color: #38bdf8;
    font-family: 'Cascadia Code', 'Fira Code', monospace; font-size: 12px;
  }
  .sample-input-row { display: flex; gap: 10px; margin-bottom: 8px; }
  .sample-input-row input { flex: 1; }
  .remove-btn {
    background: #dc262620; color: #dc2626; border: 1px solid #dc2626;
    padding: 6px 12px; border-radius: 6px; cursor: pointer; font-size: 12px;
  }
  .remove-btn:hover { background: #dc262630; }
  .candidate-card {
    background: #0f172a; border: 1px solid #334155; border-radius: 8px;
    padding: 14px 16px; margin-bottom: 10px;
  }
  .candidate-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
  .candidate-pattern {
    font-family: 'Cascadia Code', 'Fira Code', monospace;
    color: #34d399; font-size: 14px; word-break: break-all;
  }
  .candidate-score {
    background: #fbbf2420; color: #fbbf24; padding: 4px 10px;
    border-radius: 12px; font-size: 12px; font-weight: 600;
  }
  .candidate-meta { display: flex; gap: 16px; font-size: 12px; color: #64748b; margin-bottom: 8px; }
  .meta-item { display: flex; align-items: center; gap: 4px; }
  .status-badge {
    padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600;
  }
  .status-good { background: #34d39920; color: #34d399; }
  .status-warn { background: #fbbf2420; color: #fbbf24; }
  .status-bad { background: #dc262620; color: #dc2626; }
  .tip {
    padding: 10px 14px; border-radius: 8px; margin-bottom: 8px; font-size: 13px;
  }
  .tip-success { background: #34d39915; color: #34d399; border-left: 3px solid #34d399; }
  .tip-warning { background: #fbbf2415; color: #fbbf24; border-left: 3px solid #fbbf24; }
  .tip-info { background: #38bdf815; color: #38bdf8; border-left: 3px solid #38bdf8; }
  .loading { text-align: center; padding: 40px; color: #64748b; }
  .spinner {
    border: 3px solid #334155; border-top: 3px solid #38bdf8;
    border-radius: 50%; width: 32px; height: 32px;
    animation: spin 1s linear infinite; margin: 0 auto 12px;
  }
  @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
  .example-list { margin-top: 8px; }
  .example-btn {
    display: inline-block; background: #0f172a; border: 1px solid #334155;
    color: #94a3b8; padding: 6px 12px; border-radius: 6px; cursor: pointer;
    font-size: 12px; margin-right: 6px; margin-bottom: 6px;
  }
  .example-btn:hover { border-color: #38bdf8; color: #38bdf8; }
</style>
</head>
<body>
<div class="container">
  <h1>正则表达式工具</h1>
  <p class="subtitle">测试正则表达式，或根据样本自动生成匹配规则</p>

  <div class="tabs">
    <div class="tab active" onclick="switchTab('test')">手动测试</div>
    <div class="tab" onclick="switchTab('generate')">智能生成</div>
  </div>

  <div id="tab-test" class="tab-content active">
    <div class="card">
      <div class="row">
        <div>
          <label>正则表达式 Pattern</label>
          <input type="text" id="pattern" placeholder="例如: (\\d+)-(\\w+)" />
        </div>
        <div>
          <label>标志 Flags</label>
          <div class="flags">
            <label><input type="checkbox" value="GLOBAL" checked disabled /> g 全局</label>
            <label><input type="checkbox" value="IGNORECASE" /> i 忽略大小写</label>
            <label><input type="checkbox" value="MULTILINE" /> m 多行</label>
            <label><input type="checkbox" value="DOTALL" /> s 点匹配换行</label>
            <label><input type="checkbox" value="UNICODE" /> u Unicode</label>
          </div>
        </div>
      </div>
      <div style="margin-top:16px">
        <label>测试字符串 Test String</label>
        <textarea id="test_string" placeholder="输入要匹配的文本..."></textarea>
      </div>
      <button class="btn" onclick="testRegex()">测试匹配</button>
    </div>

    <div class="card" id="highlight-card" style="display:none">
      <label>匹配高亮预览</label>
      <div class="highlight-container" id="highlight"></div>
    </div>

    <div class="card" id="results-card" style="display:none">
      <div class="result-header">
        <h3>匹配结果</h3>
        <span class="match-count" id="match-count">0 个匹配</span>
      </div>
      <div id="results"></div>
    </div>
  </div>

  <div id="tab-generate" class="tab-content">
    <div class="card">
      <p style="color:#94a3b8; font-size:14px; margin-bottom:16px">
        输入几个<strong style="color:#34d399">应该匹配</strong>的正例样本和<strong style="color:#f87171">不应该匹配</strong>的反例样本，系统将自动生成可能的正则表达式。
      </p>

      <div style="margin-bottom:16px">
        <label>快速示例</label>
        <div class="example-list">
          <button class="example-btn" onclick="loadExample('phone')">手机号</button>
          <button class="example-btn" onclick="loadExample('email')">邮箱</button>
          <button class="example-btn" onclick="loadExample('date')">日期</button>
          <button class="example-btn" onclick="loadExample('idcard')">身份证</button>
        </div>
      </div>

      <div style="margin-bottom:16px">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
          <label style="margin-bottom:0; color:#34d399">✓ 正例样本（应该匹配）</label>
          <button class="btn btn-secondary btn-small" onclick="addSample('positive')">+ 添加</button>
        </div>
        <div id="positive-samples"></div>
      </div>

      <div style="margin-bottom:16px">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
          <label style="margin-bottom:0; color:#f87171">✗ 反例样本（不应该匹配）</label>
          <button class="btn btn-secondary btn-small" onclick="addSample('negative')">+ 添加</button>
        </div>
        <div id="negative-samples"></div>
      </div>

      <button class="btn" onclick="generateRegex()">生成正则表达式</button>
    </div>

    <div class="card" id="generate-results-card" style="display:none">
      <div class="result-header">
        <h3>生成结果</h3>
        <span class="match-count" id="candidate-count">0 个候选</span>
      </div>
      <div id="generate-tips"></div>
      <div id="generate-results"></div>
    </div>
  </div>

  <div class="api-hint">
    <strong>API 调用方式：</strong><br>
    <code>POST /api/regex/test</code> — 测试正则表达式<br>
    请求体：<code>{ "pattern": "(\\\\d+)", "test_string": "abc123", "flags": ["IGNORECASE"] }</code><br><br>
    <code>POST /api/regex/validate</code> — 验证正则表达式是否合法<br>
    请求体：<code>{ "pattern": "[a-z+" }</code><br><br>
    <code>POST /api/regex/generate</code> — 根据样本生成正则表达式<br>
    请求体：<code>{ "positives": ["13812345678", "13987654321"], "negatives": ["abc"] }</code>
  </div>
</div>

<script>
function switchTab(tab) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  document.querySelector('.tab:nth-child(' + (tab === 'test' ? '1' : '2') + ')').classList.add('active');
  document.getElementById('tab-' + tab).classList.add('active');
}

async function testRegex() {
  const pattern = document.getElementById('pattern').value;
  const test_string = document.getElementById('test_string').value;
  const flags = [];
  document.querySelectorAll('.flags input[type="checkbox"]:checked').forEach(cb => {
    if (cb.value !== 'GLOBAL') flags.push(cb.value);
  });

  try {
    const resp = await fetch('/api/regex/test', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ pattern, test_string, flags })
    });
    const data = await resp.json();

    if (data.error) {
      document.getElementById('results-card').style.display = '';
      document.getElementById('highlight-card').style.display = 'none';
      document.getElementById('match-count').textContent = '错误';
      document.getElementById('results').innerHTML =
        '<div class="error">' + escapeHtml(data.error) + '</div>';
      return;
    }

    document.getElementById('match-count').textContent = data.match_count + ' 个匹配';

    if (data.match_count === 0) {
      document.getElementById('results-card').style.display = '';
      document.getElementById('highlight-card').style.display = 'none';
      document.getElementById('results').innerHTML =
        '<p class="no-match">未找到匹配项</p>';
      return;
    }

    renderHighlight(data);
    renderResults(data);
  } catch (e) {
    document.getElementById('results-card').style.display = '';
    document.getElementById('highlight-card').style.display = 'none';
    document.getElementById('match-count').textContent = '错误';
    document.getElementById('results').innerHTML =
      '<div class="error">请求失败: ' + escapeHtml(e.message) + '</div>';
  }
}

function renderHighlight(data) {
  const el = document.getElementById('highlight');
  const card = document.getElementById('highlight-card');
  card.style.display = '';
  let html = '';
  let last = 0;
  for (const m of data.matches) {
    html += escapeHtml(data.test_string.slice(last, m.start));
    html += '<mark>' + escapeHtml(m.match) + '</mark>';
    last = m.end;
  }
  html += escapeHtml(data.test_string.slice(last));
  el.innerHTML = html;
}

function renderResults(data) {
  const el = document.getElementById('results');
  const card = document.getElementById('results-card');
  card.style.display = '';
  let html = '';
  data.matches.forEach((m, idx) => {
    html += '<div class="match-item">';
    html += '<div class="match-item-header">';
    html += '<span class="match-value">' + escapeHtml(m.match) + '</span>';
    html += '<span class="match-pos">位置 ' + m.start + '-' + m.end + '</span>';
    html += '</div>';

    if (m.groups && m.groups.length > 0) {
      html += '<div class="groups-section"><div class="groups-title">捕获组 Groups</div>';
      m.groups.forEach(g => {
        html += '<div class="group-row">';
        html += '<span class="group-index">组 ' + g.index + '</span>';
        html += '<span class="group-value">' + (g.value !== null ? escapeHtml(g.value) : '(未匹配)') + '</span>';
        if (g.start !== null) html += '<span class="group-pos">[' + g.start + ',' + g.end + ']</span>';
        html += '</div>';
      });
      html += '</div>';
    }

    const namedKeys = Object.keys(m.named_groups || {});
    if (namedKeys.length > 0) {
      html += '<div class="groups-section" style="margin-top:6px"><div class="groups-title">命名捕获组 Named Groups</div>';
      namedKeys.forEach(name => {
        const ng = m.named_groups[name];
        html += '<div class="group-row">';
        html += '<span class="group-index">&lt;' + escapeHtml(name) + '&gt;</span>';
        html += '<span class="group-value">' + (ng.value !== null ? escapeHtml(ng.value) : '(未匹配)') + '</span>';
        if (ng.start !== null) html += '<span class="group-pos">[' + ng.start + ',' + ng.end + ']</span>';
        html += '</div>';
      });
      html += '</div>';
    }

    html += '</div>';
  });
  el.innerHTML = html;
}

function escapeHtml(str) {
  if (str === null || str === undefined) return '';
  return String(str).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

document.getElementById('pattern').addEventListener('keydown', e => {
  if (e.key === 'Enter') testRegex();
});

let positiveSamples = [];
let negativeSamples = [];
let sampleCounter = 0;

function addSample(type, value = '') {
  const id = ++sampleCounter;
  const container = document.getElementById(type + '-samples');
  const isPositive = type === 'positive';
  const html = \`
    <div class="sample-input-row" id="sample-\${type}-\${id}">
      <input type="text" placeholder="\${isPositive ? '例如: 13812345678' : '例如: abc123'}" value="\${escapeHtml(value)}" />
      <button class="remove-btn" onclick="removeSample('\${type}', \${id})">删除</button>
    </div>
  \`;
  container.insertAdjacentHTML('beforeend', html);
}

function removeSample(type, id) {
  document.getElementById('sample-' + type + '-' + id).remove();
}

function getSamples(type) {
  const inputs = document.querySelectorAll('#' + type + '-samples input');
  const values = [];
  inputs.forEach(input => {
    const val = input.value.trim();
    if (val) values.push(val);
  });
  return values;
}

async function generateRegex() {
  const positives = getSamples('positive');
  const negatives = getSamples('negative');

  if (positives.length === 0) {
    alert('请至少添加一个正例样本');
    return;
  }

  const resultsCard = document.getElementById('generate-results-card');
  const resultsEl = document.getElementById('generate-results');
  const tipsEl = document.getElementById('generate-tips');
  resultsCard.style.display = '';
  resultsEl.innerHTML = '<div class="loading"><div class="spinner"></div>正在生成正则表达式...</div>';
  tipsEl.innerHTML = '';
  document.getElementById('candidate-count').textContent = '生成中...';

  try {
    const resp = await fetch('/api/regex/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ positives, negatives })
    });
    const data = await resp.json();

    if (data.error) {
      resultsEl.innerHTML = '<div class="error">' + escapeHtml(data.error) + '</div>';
      document.getElementById('candidate-count').textContent = '错误';
      return;
    }

    document.getElementById('candidate-count').textContent = data.candidates.length + ' 个候选';

    let tipsHtml = '';
    (data.tips || []).forEach(tip => {
      const cls = tip.type === 'success' ? 'tip-success' : tip.type === 'warning' ? 'tip-warning' : 'tip-info';
      tipsHtml += '<div class="tip ' + cls + '">' + escapeHtml(tip.message) + '</div>';
    });
    tipsEl.innerHTML = tipsHtml;

    if (data.candidates.length === 0) {
      resultsEl.innerHTML = '<p class="no-match">未生成合适的正则表达式，请添加更多样本</p>';
      return;
    }

    let html = '';
    data.candidates.forEach((c, idx) => {
      html += '<div class="candidate-card">';
      html += '<div class="candidate-header">';
      html += '<span class="candidate-pattern">' + escapeHtml(c.pattern) + '</span>';
      html += '<span class="candidate-score">' + (c.score|0) + '分</span>';
      html += '</div>';

      html += '<div class="candidate-meta">';
      html += '<span class="meta-item">召回率: ' + ((c.recall||0)*100).toFixed(0) + '%</span>';
      html += '<span class="meta-item">精确率: ' + ((c.precision||0)*100).toFixed(0) + '%</span>';
      html += '<span class="meta-item">F1: ' + ((c.f1||0)*100).toFixed(0) + '%</span>';
      html += c.matches_all_positives ? '<span class="status-badge status-good">匹配全部正例</span>' : '<span class="status-badge status-bad">遗漏正例</span>';
      html += c.rejects_all_negatives ? '<span class="status-badge status-good">拒绝全部反例</span>' : '<span class="status-badge status-warn">匹配反例</span>';
      html += '</div>';

      html += '<div style="display:flex;gap:10px">';
      html += '<button class="btn btn-secondary btn-small" onclick="usePattern(\\'' + c.pattern.replace(/'/g, "\\\\'") + '\\')">使用此正则</button>';
      html += '</div>';
      html += '</div>';
    });
    resultsEl.innerHTML = html;
  } catch (e) {
    resultsEl.innerHTML = '<div class="error">生成失败: ' + escapeHtml(e.message) + '</div>';
    document.getElementById('candidate-count').textContent = '错误';
  }
}

function usePattern(pattern) {
  document.getElementById('pattern').value = pattern;
  switchTab('test');
}

const examples = {
  phone: {
    positives: ['13812345678', '13987654321', '15011112222'],
    negatives: ['123456', 'abc', '1380013800']
  },
  email: {
    positives: ['user@example.com', 'test.email@company.org', 'a1_b2@xyz.cn'],
    negatives: ['user@', '@example.com', 'user.example.com']
  },
  date: {
    positives: ['2024-01-15', '2025-12-31', '2023-06-01'],
    negatives: ['2024/01/15', '01-15-2024', '2024-13-40']
  },
  idcard: {
    positives: ['110101199001011234', '31010119850615123X'],
    negatives: ['11010119900101', 'abc123', '000000000000000000']
  }
};

function loadExample(name) {
  const ex = examples[name];
  document.getElementById('positive-samples').innerHTML = '';
  document.getElementById('negative-samples').innerHTML = '';
  ex.positives.forEach(v => addSample('positive', v));
  ex.negatives.forEach(v => addSample('negative', v));
}

addSample('positive');
addSample('positive');
addSample('negative');
</script>
</body>
</html>`);
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`正则表达式测试 API 已启动: http://localhost:${PORT}`);
});
