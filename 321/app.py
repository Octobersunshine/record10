from flask import Flask, request, jsonify, render_template_string
from qrcode_utils import generate_qrcode_base64, decode_qrcode_image, MAX_QR_CONTENT_LENGTHS, MAX_QR_CONTENT_LENGTH

app = Flask(__name__)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>二维码工具</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            padding: 40px;
        }
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 30px;
            font-size: 28px;
        }
        .tabs {
            display: flex;
            margin-bottom: 30px;
            border-bottom: 2px solid #eee;
        }
        .tab {
            flex: 1;
            padding: 15px;
            text-align: center;
            cursor: pointer;
            font-size: 16px;
            font-weight: 600;
            color: #666;
            transition: all 0.3s;
            border-bottom: 3px solid transparent;
            margin-bottom: -2px;
        }
        .tab.active {
            color: #667eea;
            border-bottom-color: #667eea;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            color: #333;
            font-weight: 500;
        }
        textarea, input[type="text"], select {
            width: 100%;
            padding: 12px 16px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 14px;
            transition: border-color 0.3s;
        }
        textarea:focus, input[type="text"], select:focus {
            outline: none;
            border-color: #667eea;
        }
        textarea {
            resize: vertical;
            min-height: 100px;
        }
        textarea.over-limit {
            border-color: #dc3545;
            background: #fff5f5;
        }
        textarea.near-limit {
            border-color: #ffc107;
            background: #fffdf5;
        }
        .row {
            display: flex;
            gap: 15px;
        }
        .row .form-group {
            flex: 1;
        }
        button {
            width: 100%;
            padding: 14px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        }
        button:active {
            transform: translateY(0);
        }
        .result {
            margin-top: 25px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
            display: none;
        }
        .result.show {
            display: block;
        }
        .result h3 {
            color: #333;
            margin-bottom: 15px;
        }
        .qrcode-display {
            text-align: center;
            margin-bottom: 15px;
        }
        .qrcode-display img {
            max-width: 250px;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 10px;
            background: white;
        }
        .base64-output {
            background: #f1f3f5;
            padding: 12px;
            border-radius: 6px;
            font-family: monospace;
            font-size: 12px;
            word-break: break-all;
            max-height: 150px;
            overflow-y: auto;
            margin-bottom: 10px;
        }
        .decoded-text {
            background: #f1f3f5;
            padding: 15px;
            border-radius: 6px;
            font-size: 14px;
            line-height: 1.6;
            word-break: break-word;
        }
        .file-upload {
            border: 2px dashed #ccc;
            border-radius: 8px;
            padding: 30px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
        }
        .file-upload:hover, .file-upload.dragover {
            border-color: #667eea;
            background: #f8f9ff;
        }
        .file-upload input {
            display: none;
        }
        .file-upload p {
            color: #666;
            margin-top: 10px;
        }
        .file-preview {
            margin-top: 15px;
            text-align: center;
        }
        .file-preview img {
            max-width: 200px;
            border-radius: 8px;
        }
        .error {
            color: #dc3545;
            background: #f8d7da;
            padding: 12px;
            border-radius: 6px;
            margin-top: 15px;
            display: none;
        }
        .error.show {
            display: block;
        }
        .success {
            color: #155724;
            background: #d4edda;
            padding: 12px;
            border-radius: 6px;
            margin-top: 15px;
            display: none;
        }
        .success.show {
            display: block;
        }
        .copy-btn {
            padding: 8px 16px;
            font-size: 13px;
            background: #6c757d;
            width: auto;
        }
        .copy-btn:hover {
            background: #5a6268;
        }
        .char-counter {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 6px;
            font-size: 13px;
            color: #6c757d;
        }
        .char-counter .count {
            font-weight: 600;
        }
        .char-counter.warning .count {
            color: #ffc107;
        }
        .char-counter.danger .count {
            color: #dc3545;
        }
        .char-hint {
            font-size: 12px;
            color: #dc3545;
            margin-top: 4px;
            display: none;
        }
        .char-hint.show {
            display: block;
        }
        .color-picker-group {
            display: flex;
            gap: 15px;
            margin-bottom: 15px;
        }
        .color-picker-item {
            flex: 1;
        }
        .color-picker-item label {
            display: block;
            margin-bottom: 6px;
        }
        .color-picker-wrapper {
            display: flex;
            gap: 8px;
            align-items: center;
        }
        .color-picker-wrapper input[type="color"] {
            width: 48px;
            height: 40px;
            border: 2px solid #e0e0e0;
            border-radius: 6px;
            padding: 2px;
            cursor: pointer;
            background: none;
        }
        .color-picker-wrapper input[type="text"] {
            flex: 1;
            font-family: monospace;
        }
        .preset-colors {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin-top: 8px;
        }
        .preset-color-btn {
            width: 32px;
            height: 32px;
            border: 2px solid #e0e0e0;
            border-radius: 50%;
            cursor: pointer;
            padding: 0;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .preset-color-btn:hover {
            transform: scale(1.15);
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        }
        .preset-color-btn.active {
            border-color: #667eea;
            box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.3);
        }
        .logo-upload {
            border: 2px dashed #ccc;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
        }
        .logo-upload:hover, .logo-upload.dragover {
            border-color: #667eea;
            background: #f8f9ff;
        }
        .logo-upload input {
            display: none;
        }
        .logo-upload p {
            color: #666;
            font-size: 13px;
            margin-top: 6px;
        }
        .logo-info {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-top: 10px;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 6px;
            display: none;
        }
        .logo-info.show {
            display: flex;
        }
        .logo-info img {
            max-width: 48px;
            max-height: 48px;
            border-radius: 4px;
        }
        .logo-info .remove-logo {
            background: #dc3545;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 4px 10px;
            cursor: pointer;
            font-size: 12px;
            width: auto;
        }
        .logo-info .remove-logo:hover {
            background: #c82333;
        }
        .section-title {
            font-size: 14px;
            font-weight: 600;
            color: #333;
            margin-bottom: 10px;
            padding-bottom: 6px;
            border-bottom: 2px solid #eee;
        }
        .settings-section {
            padding: 15px;
            background: #fafbfc;
            border-radius: 8px;
            margin-bottom: 15px;
        }
        .api-info {
            margin-top: 30px;
            padding: 20px;
            background: #e9ecef;
            border-radius: 8px;
        }
        .api-info h4 {
            margin-bottom: 10px;
            color: #333;
        }
        .api-info code {
            background: #f8f9fa;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 13px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>二维码生成与解码工具</h1>
        
        <div class="tabs">
            <div class="tab active" onclick="switchTab('generate')">生成二维码</div>
            <div class="tab" onclick="switchTab('decode')">解析二维码</div>
        </div>

        <div id="generate" class="tab-content active">
            <div class="form-group">
                <label for="textInput">输入文本或URL</label>
                <textarea id="textInput" placeholder="请输入要生成二维码的文本内容或网址..." oninput="updateCharCount()"></textarea>
                <div class="char-counter" id="charCounter">
                    <span>当前纠错级别最大容量 <strong id="maxLenLabel">2331</strong> 字符</span>
                    <span class="count"><span id="charCount">0</span> / <span id="maxLenCount">2331</span></span>
                </div>
                <div class="char-hint" id="charHint">内容过长，二维码可能无法生成。请精简文本或降低纠错级别。</div>
            </div>

            <div class="settings-section">
                <div class="section-title">基础设置</div>
                <div class="row">
                    <div class="form-group">
                        <label for="errorCorrection">纠错级别</label>
                        <select id="errorCorrection" onchange="updateCharCount()">
                            <option value="L">L - 低纠错 (7%) - 最大2953字符</option>
                            <option value="M" selected>M - 中纠错 (15%) - 最大2331字符</option>
                            <option value="Q">Q - 较高纠错 (25%) - 最大1663字符</option>
                            <option value="H">H - 高纠错 (30%) - 最大1273字符</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="boxSize">方块大小</label>
                        <select id="boxSize">
                            <option value="8">8</option>
                            <option value="10" selected>10</option>
                            <option value="12">12</option>
                            <option value="15">15</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="border">边框宽度</label>
                        <select id="border">
                            <option value="2">2</option>
                            <option value="4" selected>4</option>
                            <option value="6">6</option>
                            <option value="8">8</option>
                        </select>
                    </div>
                </div>
            </div>

            <div class="settings-section">
                <div class="section-title">颜色设置</div>
                <div class="color-picker-group">
                    <div class="color-picker-item">
                        <label>前景色（码点）</label>
                        <div class="color-picker-wrapper">
                            <input type="color" id="fillColor" value="#000000" onchange="syncColor('fill')">
                            <input type="text" id="fillColorText" value="#000000" onchange="syncColorFromText('fill')" maxlength="7">
                        </div>
                    </div>
                    <div class="color-picker-item">
                        <label>背景色</label>
                        <div class="color-picker-wrapper">
                            <input type="color" id="backColor" value="#FFFFFF" onchange="syncColor('back')">
                            <input type="text" id="backColorText" value="#FFFFFF" onchange="syncColorFromText('back')" maxlength="7">
                        </div>
                    </div>
                </div>
                <label style="margin-top: 10px;">预设配色：</label>
                <div class="preset-colors">
                    <button type="button" class="preset-color-btn active" style="background: linear-gradient(135deg, #000000 50%, #FFFFFF 50%);" onclick="applyPreset('#000000', '#FFFFFF', this)" title="经典黑白"></button>
                    <button type="button" class="preset-color-btn" style="background: linear-gradient(135deg, #1a1a2e 50%, #eaeaea 50%);" onclick="applyPreset('#1a1a2e', '#eaeaea', this)" title="深蓝浅灰"></button>
                    <button type="button" class="preset-color-btn" style="background: linear-gradient(135deg, #1e3a8a 50%, #f0f9ff 50%);" onclick="applyPreset('#1e3a8a', '#f0f9ff', this)" title="蓝白"></button>
                    <button type="button" class="preset-color-btn" style="background: linear-gradient(135deg, #7c3aed 50%, #faf5ff 50%);" onclick="applyPreset('#7c3aed', '#faf5ff', this)" title="紫白"></button>
                    <button type="button" class="preset-color-btn" style="background: linear-gradient(135deg, #dc2626 50%, #fef2f2 50%);" onclick="applyPreset('#dc2626', '#fef2f2', this)" title="红白"></button>
                    <button type="button" class="preset-color-btn" style="background: linear-gradient(135deg, #059669 50%, #ecfdf5 50%);" onclick="applyPreset('#059669', '#ecfdf5', this)" title="绿白"></button>
                    <button type="button" class="preset-color-btn" style="background: linear-gradient(135deg, #f59e0b 50%, #fffbeb 50%);" onclick="applyPreset('#f59e0b', '#fffbeb', this)" title="橙白"></button>
                    <button type="button" class="preset-color-btn" style="background: linear-gradient(135deg, #0f172a 50%, #38bdf8 50%);" onclick="applyPreset('#0f172a', '#38bdf8', this)" title="深蓝蓝"></button>
                    <button type="button" class="preset-color-btn" style="background: linear-gradient(135deg, #7c2d12 50%, #fcd34d 50%);" onclick="applyPreset('#7c2d12', '#fcd34d', this)" title="棕黄"></button>
                    <button type="button" class="preset-color-btn" style="background: linear-gradient(135deg, #be185d 50%, #fbcfe8 50%);" onclick="applyPreset('#be185d', '#fbcfe8', this)" title="粉白"></button>
                </div>
            </div>

            <div class="settings-section">
                <div class="section-title">Logo 设置 <span style="font-size: 12px; color: #6c757d; font-weight: normal;">（建议使用透明背景 PNG，将自动启用高纠错）</span></div>
                <div class="logo-upload" id="logoUpload">
                    <input type="file" id="logoInput" accept="image/*" onchange="handleLogoSelect(event)">
                    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#999" stroke-width="2">
                        <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                        <circle cx="8.5" cy="8.5" r="1.5"></circle>
                        <polyline points="21 15 16 10 5 21"></polyline>
                    </svg>
                    <p>点击或拖拽 Logo 图片到此处上传</p>
                </div>
                <div class="logo-info" id="logoInfo">
                    <span style="display: flex; align-items: center; gap: 10px;">
                        <img id="logoPreview" alt="Logo预览">
                        <span id="logoName" style="font-size: 13px; color: #333;"></span>
                    </span>
                    <button type="button" class="remove-logo" onclick="removeLogo()">移除</button>
                </div>
                <div class="form-group" style="margin-top: 12px;">
                    <label for="logoRatio">Logo 大小：<span id="logoRatioValue">22%</span></label>
                    <input type="range" id="logoRatio" min="10" max="35" value="22" style="width: 100%;" oninput="document.getElementById('logoRatioValue').textContent = this.value + '%'">
                </div>
            </div>

            <button onclick="generateQRCode()">生成二维码</button>
            
            <div id="generateResult" class="result">
                <h3>生成结果</h3>
                <div class="qrcode-display">
                    <img id="qrcodeImg" alt="二维码图片">
                </div>
                <label>Base64 编码：</label>
                <div class="base64-output" id="base64Output"></div>
                <button class="copy-btn" onclick="copyBase64()">复制 Base64</button>
            </div>
            <div id="generateSuccess" class="success"></div>
            <div id="generateError" class="error"></div>
        </div>

        <div id="decode" class="tab-content">
            <div class="form-group">
                <label>上传二维码图片</label>
                <div class="file-upload" id="fileUpload">
                    <input type="file" id="fileInput" accept="image/*" onchange="handleFileSelect(event)">
                    <svg width="50" height="50" viewBox="0 0 24 24" fill="none" stroke="#999" stroke-width="2">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                        <polyline points="17 8 12 3 7 8"></polyline>
                        <line x1="12" y1="3" x2="12" y2="15"></line>
                    </svg>
                    <p>点击或拖拽图片到此处上传</p>
                </div>
                <div class="file-preview" id="filePreview"></div>
            </div>
            <button onclick="decodeQRCode()">解析二维码</button>
            
            <div id="decodeResult" class="result">
                <h3>解析结果</h3>
                <div class="decoded-text" id="decodedText"></div>
            </div>
            <div id="decodeError" class="error"></div>
        </div>

        <div class="api-info">
            <h4>API 接口说明</h4>
            <p><strong>生成二维码：</strong> <code>POST /api/generate</code></p>
            <p><strong>解析二维码：</strong> <code>POST /api/decode</code></p>
        </div>
    </div>

    <script>
        const MAX_LENGTHS = { 'L': 2953, 'M': 2331, 'Q': 1663, 'H': 1273 };
        let selectedLogoFile = null;

        function getCurrentMaxLength() {
            const ec = document.getElementById('errorCorrection').value;
            return MAX_LENGTHS[ec] || 2331;
        }

        function updateCharCount() {
            const textarea = document.getElementById('textInput');
            const ecSelect = document.getElementById('errorCorrection');
            let maxLen = getCurrentMaxLength();
            const count = textarea.value.length;
            const countEl = document.getElementById('charCount');
            const counterEl = document.getElementById('charCounter');
            const hintEl = document.getElementById('charHint');
            const maxLenLabel = document.getElementById('maxLenLabel');
            const maxLenCount = document.getElementById('maxLenCount');

            if (selectedLogoFile && !['H', 'Q'].includes(ecSelect.value)) {
                maxLen = MAX_LENGTHS['H'];
            }

            countEl.textContent = count;
            maxLenLabel.textContent = maxLen;
            maxLenCount.textContent = maxLen;
            textarea.classList.remove('over-limit', 'near-limit');
            counterEl.classList.remove('warning', 'danger');
            hintEl.classList.remove('show');

            if (count > maxLen) {
                textarea.classList.add('over-limit');
                counterEl.classList.add('danger');
                hintEl.classList.add('show');
            } else if (count > maxLen * 0.8) {
                textarea.classList.add('near-limit');
                counterEl.classList.add('warning');
            }
        }

        function syncColor(type) {
            if (type === 'fill') {
                const color = document.getElementById('fillColor').value;
                document.getElementById('fillColorText').value = color;
            } else {
                const color = document.getElementById('backColor').value;
                document.getElementById('backColorText').value = color;
            }
            clearPresetActive();
        }

        function syncColorFromText(type) {
            if (type === 'fill') {
                let color = document.getElementById('fillColorText').value.trim();
                if (!color.startsWith('#')) color = '#' + color;
                if (/^#[0-9A-Fa-f]{6}$/.test(color)) {
                    document.getElementById('fillColor').value = color;
                }
            } else {
                let color = document.getElementById('backColorText').value.trim();
                if (!color.startsWith('#')) color = '#' + color;
                if (/^#[0-9A-Fa-f]{6}$/.test(color)) {
                    document.getElementById('backColor').value = color;
                }
            }
            clearPresetActive();
        }

        function applyPreset(fill, back, btn) {
            document.getElementById('fillColor').value = fill;
            document.getElementById('fillColorText').value = fill;
            document.getElementById('backColor').value = back;
            document.getElementById('backColorText').value = back;
            clearPresetActive();
            btn.classList.add('active');
        }

        function clearPresetActive() {
            document.querySelectorAll('.preset-color-btn').forEach(b => b.classList.remove('active'));
        }

        const logoUpload = document.getElementById('logoUpload');
        logoUpload.addEventListener('click', () => document.getElementById('logoInput').click());

        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            logoUpload.addEventListener(eventName, preventDefaults, false);
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            logoUpload.addEventListener(eventName, () => logoUpload.classList.add('dragover'), false);
        });
        ['dragleave', 'drop'].forEach(eventName => {
            logoUpload.addEventListener(eventName, () => logoUpload.classList.remove('dragover'), false);
        });

        logoUpload.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files.length > 0) {
                document.getElementById('logoInput').files = files;
                handleLogoFile(files[0]);
            }
        }, false);

        function handleLogoSelect(event) {
            const file = event.target.files[0];
            if (file) handleLogoFile(file);
        }

        function handleLogoFile(file) {
            selectedLogoFile = file;
            const reader = new FileReader();
            reader.onload = function(e) {
                document.getElementById('logoPreview').src = e.target.result;
                document.getElementById('logoName').textContent = file.name;
                document.getElementById('logoInfo').classList.add('show');
            };
            reader.readAsDataURL(file);
            updateCharCount();
        }

        function removeLogo() {
            selectedLogoFile = null;
            document.getElementById('logoInput').value = '';
            document.getElementById('logoInfo').classList.remove('show');
            document.getElementById('logoPreview').src = '';
            document.getElementById('logoName').textContent = '';
            updateCharCount();
        }

        function switchTab(tabName) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById(tabName).classList.add('active');
            document.querySelectorAll('.result').forEach(r => r.classList.remove('show'));
            document.querySelectorAll('.error').forEach(e => e.classList.remove('show'));
            document.querySelectorAll('.success').forEach(s => s.classList.remove('show'));
        }

        const fileUpload = document.getElementById('fileUpload');
        fileUpload.addEventListener('click', () => document.getElementById('fileInput').click());
        
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            fileUpload.addEventListener(eventName, preventDefaults, false);
        });
        function preventDefaults(e) { e.preventDefault(); e.stopPropagation(); }
        
        ['dragenter', 'dragover'].forEach(eventName => {
            fileUpload.addEventListener(eventName, () => fileUpload.classList.add('dragover'), false);
        });
        ['dragleave', 'drop'].forEach(eventName => {
            fileUpload.addEventListener(eventName, () => fileUpload.classList.remove('dragover'), false);
        });
        
        fileUpload.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files.length > 0) {
                document.getElementById('fileInput').files = files;
                showFilePreview(files[0]);
            }
        }, false);

        function handleFileSelect(event) {
            const file = event.target.files[0];
            if (file) showFilePreview(file);
        }

        function showFilePreview(file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                document.getElementById('filePreview').innerHTML = 
                    '<img src="' + e.target.result + '" alt="预览">';
            };
            reader.readAsDataURL(file);
        }

        async function generateQRCode() {
            const text = document.getElementById('textInput').value.trim();
            if (!text) {
                showError('generateError', '请输入要生成二维码的文本内容');
                return;
            }

            const ecSelect = document.getElementById('errorCorrection');
            let errorCorrection = ecSelect.value;
            let maxLen = getCurrentMaxLength();

            if (selectedLogoFile && !['H', 'Q'].includes(errorCorrection)) {
                errorCorrection = 'H';
                maxLen = MAX_LENGTHS['H'];
            }

            if (text.length > maxLen) {
                showError('generateError', '文本内容过长（' + text.length + ' 字符），超出纠错级别 ' + errorCorrection + ' 的最大容量（' + maxLen + ' 字符）。请精简文本或降低纠错级别。');
                return;
            }

            const boxSize = parseInt(document.getElementById('boxSize').value);
            const border = parseInt(document.getElementById('border').value);
            const fillColor = document.getElementById('fillColorText').value;
            const backColor = document.getElementById('backColorText').value;
            const logoRatio = parseInt(document.getElementById('logoRatio').value) / 100;

            hideError('generateError');
            try {
                const formData = new FormData();
                formData.append('text', text);
                formData.append('error_correction', errorCorrection);
                formData.append('box_size', boxSize);
                formData.append('border', border);
                formData.append('fill_color', fillColor);
                formData.append('back_color', backColor);
                formData.append('logo_ratio', logoRatio);

                if (selectedLogoFile) {
                    formData.append('file', selectedLogoFile);
                }

                const response = await fetch('/api/generate', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                if (data.success) {
                    document.getElementById('qrcodeImg').src = 'data:image/png;base64,' + data.base64;
                    document.getElementById('base64Output').textContent = data.base64;
                    document.getElementById('generateResult').classList.add('show');

                    let resultMsg = '生成成功';
                    if (data.has_logo) {
                        resultMsg += '（已自动启用高纠错 ' + data.error_correction + '）';
                    }
                    showSuccess(resultMsg);
                } else {
                    showError('generateError', data.error);
                }
            } catch (e) {
                showError('generateError', '请求失败：' + e.message);
            }
        }

        async function decodeQRCode() {
            const fileInput = document.getElementById('fileInput');
            if (!fileInput.files.length) {
                showError('decodeError', '请先选择要解析的二维码图片');
                return;
            }

            const formData = new FormData();
            formData.append('file', fileInput.files[0]);

            hideError('decodeError');
            try {
                const response = await fetch('/api/decode', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                if (data.success) {
                    document.getElementById('decodedText').textContent = data.text;
                    document.getElementById('decodeResult').classList.add('show');
                } else {
                    showError('decodeError', data.error);
                }
            } catch (e) {
                showError('decodeError', '请求失败：' + e.message);
            }
        }

        function showError(id, message) {
            const el = document.getElementById(id);
            el.textContent = message;
            el.classList.add('show');
            if (id === 'generateError') {
                hideSuccess('generateSuccess');
            }
        }

        function hideError(id) {
            document.getElementById(id).classList.remove('show');
        }

        function showSuccess(message) {
            const el = document.getElementById('generateSuccess');
            el.textContent = '✓ ' + message;
            el.classList.add('show');
            setTimeout(() => el.classList.remove('show'), 3000);
        }

        function hideSuccess(id) {
            document.getElementById(id).classList.remove('show');
        }

        function copyBase64() {
            const base64 = document.getElementById('base64Output').textContent;
            navigator.clipboard.writeText(base64).then(() => {
                const btn = event.target;
                const originalText = btn.textContent;
                btn.textContent = '已复制！';
                setTimeout(() => btn.textContent = originalText, 1500);
            });
        }
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/capacity', methods=['GET'])
def api_capacity():
    return jsonify({
        'success': True,
        'max_lengths': MAX_QR_CONTENT_LENGTHS,
        'absolute_max': MAX_QR_CONTENT_LENGTH
    })


@app.route('/api/generate', methods=['POST'])
def api_generate():
    try:
        if 'file' in request.files:
            logo_file = request.files['file']
            logo_bytes = logo_file.read() if logo_file.filename else None
        else:
            logo_bytes = None

        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()

        if not data or 'text' not in data:
            return jsonify({'success': False, 'error': '缺少必要参数: text'}), 400

        text = data['text']
        error_correction = data.get('error_correction', 'M')
        box_size = int(data.get('box_size', 10))
        border = int(data.get('border', 4))
        fill_color = data.get('fill_color', '#000000')
        back_color = data.get('back_color', '#FFFFFF')
        logo_ratio = float(data.get('logo_ratio', 0.22))

        if logo_bytes is not None and error_correction.upper() not in ('H', 'Q'):
            error_correction = 'H'

        base64_str = generate_qrcode_base64(
            text=text,
            error_correction=error_correction,
            box_size=box_size,
            border=border,
            fill_color=fill_color,
            back_color=back_color,
            logo_bytes=logo_bytes,
            logo_ratio=logo_ratio,
        )

        return jsonify({
            'success': True,
            'base64': base64_str,
            'length': len(text),
            'error_correction': error_correction,
            'has_logo': logo_bytes is not None,
        })

    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/decode', methods=['POST'])
def api_decode():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': '未找到上传的文件'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': '未选择文件'}), 400

        image_bytes = file.read()
        decoded_text = decode_qrcode_image(image_bytes)

        return jsonify({
            'success': True,
            'text': decoded_text,
            'length': len(decoded_text)
        })

    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except RuntimeError as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    except Exception as e:
        return jsonify({'success': False, 'error': f'解析失败: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
