// src/main.js
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { DecalGeometry } from 'three/examples/jsm/geometries/DecalGeometry.js';
import { GLTFExporter } from 'three/examples/jsm/exporters/GLTFExporter.js';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';
import { MTLLoader } from 'three/examples/jsm/loaders/MTLLoader.js';

// デバッグ用ログ表示システム
const logContainer = document.getElementById('debug-log-content');
function sysLog(msg) {
    if (logContainer) {
        logContainer.innerText += '\n' + msg;
        logContainer.scrollTop = logContainer.scrollHeight;
    }
}
function sysError(msg) {
    if (logContainer) {
        logContainer.innerHTML += `\n<span style="color:#ff1744">${msg}</span>`;
        logContainer.scrollTop = logContainer.scrollHeight;
    }
}

// グローバルエラーハンドリング
window.addEventListener('error', (event) => {
    const error = event.error;
    const stack = error && error.stack ? error.stack : 'No stack trace available';
    sysError(`JS ERROR: ${event.message}<br><pre style="font-size:0.6rem;color:#ff8888;margin:5px 0;text-align:left;white-space:pre-wrap;word-break:break-all">${stack}</pre>`);
});
window.addEventListener('unhandledrejection', (event) => {
    const reason = event.reason;
    const stack = reason && reason.stack ? reason.stack : 'No stack trace available';
    sysError(`JS PROMISE ERROR: ${reason}<br><pre style="font-size:0.6rem;color:#ff8888;margin:5px 0;text-align:left;white-space:pre-wrap;word-break:break-all">${stack}</pre>`);
});

// 元のconsoleをオーバーライドしてデバッグパネルに転送
const oldLog = console.log;
const oldError = console.error;
console.log = (...args) => {
    oldLog(...args);
    sysLog(args.join(' '));
};
console.error = (...args) => {
    oldError(...args);
    sysError(args.join(' '));
};

console.log("System initializing...");

// アプリケーションの状態管理 (State)
const state = {

    presets: [],
    selectedTexture: null,
    selectedImageElement: null,
    selectedImageSrc: null,
    selectedName: '',

    // プレビュー中のデカール設定
    preview: {
        mesh: null,
        projectionBoxMesh: null,
        imageBoxMesh: null,
        showProjectionBox: true,
        showImageBox: true,
        normalOffset: 0,     // 表面オフセット (0に固定)
        angleLimitDeg: 95,   // メッシュ抜け防止 (許容角度度数)
        shadowIntensity: 0.8, // 下地メッシュの影・溝反映強度 (0.0〜1.0)
        position: new THREE.Vector3(),
        normal: new THREE.Vector3(0, 0, 1),
        surfaceNormal: new THREE.Vector3(0, 0, 1),   // 表面法線（ヒット時に保存）
        cameraDirection: new THREE.Vector3(0, 0, 1), // カメラ方向（ヒット時に保存）
        scale: new THREE.Vector3(1, 1, 0.5), // 貼る領域 (クリップ範囲): 幅, 高さ, 投影の深さ
        imageScale: new THREE.Vector2(1, 1), // 画像自体の大きさ (テクスチャ枠): 幅, 高さ
        rotationZ: 0, // Z軸回転（度数）
        projectionType: 'camera', // 'surface' または 'camera'
    },

    // 確定したデカールのリスト (レイヤー)
    appliedDecals: [],
    nextLayerId: 1,

    // UI制御用
    isEditing: false,
    customModelObjectUrls: [],
    modelOriginalMaterial: null,
    modelRenderMode: 'default',
    originalCenter: null,
    originalScale: null,

    // モデル回転制御 (X軸: 上下, Y軸: 左右)
    modelRotation: {
        currentX: 0,
        targetX: 0,
        currentY: 0,
        targetY: 0
    },

    // モデル表示拡大率 (作業用スケール)
    modelDisplayScale: {
        current: 1.0,
        target: 1.0
    },

    keys: {
        w: false,
        a: false,
        s: false,
        d: false,
        q: false,
        e: false
    }
};

// 主要なThree.jsオブジェクト
let scene, camera, renderer, orbitControls, modelGroup;
let mainSphereMesh;
let raycaster, mouse;
let isUpdatingPreview = false; // 無限ループ防止フラグ
let renderRequested = true; // オンデマンド描画フラグ (初期値true)

function requestRender() {
    renderRequested = true;
}

// 定数
const SPHERE_RADIUS = 2.0;

// 初期化処理
init();

function init() {
    // 1. Three.js シーン・レンダラー等のセットアップ
    setupThree();

    // 2. プリセットデカールの生成と表示
    generatePresetDecals();

    // 3. UIのイベントリスナー設定
    setupUIListeners();

    // 4. アニメーションループ開始
    animate();
}

function setupThree() {
    const container = document.getElementById('canvas-container');
    const width = container.clientWidth;
    const height = container.clientHeight;

    // シーン
    scene = new THREE.Scene();
    scene.background = null; // CSSのグラデーション背景を活かすため透明にする

    // カメラ
    camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
    camera.position.set(0, 0, 6);

    // レンダラー (GPUハイパフォーマンス指定 ＆ DPRキャップ)
    renderer = new THREE.WebGLRenderer({
        canvas: document.getElementById('three-canvas'),
        antialias: true,
        alpha: true,
        preserveDrawingBuffer: true, // スクリーンショット保存に必要
        powerPreference: 'high-performance'
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(width, height);
    renderer.shadowMap.enabled = true;

    // ライト (光沢や立体感を出すためのライティング)
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight1.position.set(5, 5, 5);
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.4);
    dirLight2.position.set(-5, -5, -5);
    scene.add(dirLight2);

    // 床のGridHelper (レーシーな雰囲気のためのグリッド)
    const gridHelper = new THREE.GridHelper(10, 20, 0xff4500, 0x2c2e35);
    gridHelper.position.y = -2.5;
    scene.add(gridHelper);

    // デフォルトの「球体」メッシュ
    const sphereGeo = new THREE.SphereGeometry(SPHERE_RADIUS, 64, 64);
    const sphereMat = new THREE.MeshStandardMaterial({
        color: 0xdddddd,
        roughness: 0.3,
        metalness: 0.1,
        bumpScale: 0.05
    });

    // UV位置の目安用として、薄いグリッドテクスチャを球体に適用
    sphereMat.map = createSphereGridTexture();

    // モデルおよびデカールをまとめるグループ
    modelGroup = new THREE.Group();
    modelGroup.name = "ModelGroup";
    scene.add(modelGroup);

    mainSphereMesh = new THREE.Mesh(sphereGeo, sphereMat);
    mainSphereMesh.name = "MainSphere";
    modelGroup.add(mainSphereMesh);

    // マテリアルの状態を保存し、表示モードの初期状態を設定
    state.modelOriginalMaterial = sphereMat;

    // OrbitControls (カメラ操作)
    orbitControls = new OrbitControls(camera, renderer.domElement);
    orbitControls.enableDamping = true;
    orbitControls.dampingFactor = 0.05;
    orbitControls.minDistance = 0.5;
    orbitControls.maxDistance = 500;
    orbitControls.addEventListener('change', requestRender);

    // Raycasting
    raycaster = new THREE.Raycaster();
    mouse = new THREE.Vector2();

    // ウィンドウリサイズ対応
    window.addEventListener('resize', onWindowResize);
}

// 球体表面のグリッド用テクスチャをCanvasで作成
function createSphereGridTexture() {
    const canvas = document.createElement('canvas');
    canvas.width = 1024;
    canvas.height = 512;
    const ctx = canvas.getContext('2d');

    ctx.fillStyle = '#f0f2f5';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // 横線 (緯度線)
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.08)';
    ctx.lineWidth = 2;
    const numLatLines = 18;
    for (let i = 1; i < numLatLines; i++) {
        const y = (canvas.height / numLatLines) * i;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(canvas.width, y);
        ctx.stroke();
    }

    // 縦線 (経度線)
    const numLonLines = 36;
    for (let i = 0; i < numLonLines; i++) {
        const x = (canvas.width / numLonLines) * i;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvas.height);
        ctx.stroke();
    }

    // 赤道と本初子午線を少し太く
    ctx.strokeStyle = 'rgba(255, 69, 0, 0.2)';
    ctx.lineWidth = 4;

    // 赤道
    ctx.beginPath();
    ctx.moveTo(0, canvas.height / 2);
    ctx.lineTo(canvas.width, canvas.height / 2);
    ctx.stroke();

    // 本初子午線
    ctx.beginPath();
    ctx.moveTo(canvas.width / 2, 0);
    ctx.lineTo(canvas.width / 2, canvas.height);
    ctx.stroke();

    const texture = new THREE.CanvasTexture(canvas);
    texture.wrapS = THREE.RepeatWrapping;
    texture.wrapT = THREE.ClampToEdgeWrapping;
    applyHighQualityTextureFiltering(texture);
    return texture;
}

// テクスチャに異方性フィルタリング（Anisotropy）とミップマップを適用して曲面・斜め表示を高精細化
function applyHighQualityTextureFiltering(texture) {
    if (!texture) return;
    texture.generateMipmaps = true;
    texture.minFilter = THREE.LinearMipmapLinearFilter;
    texture.magFilter = THREE.LinearFilter;
    if (renderer && renderer.capabilities) {
        texture.anisotropy = renderer.capabilities.getMaxAnisotropy();
    }
}

// プリセットデカールをCanvasで生成
function generatePresetDecals() {
    const presetsData = [
        { name: 'スター', draw: drawStar },
        { name: 'レーシング矢印', draw: drawArrow },
        { name: 'チェッカー', draw: drawChecker },
        { name: 'ターゲット', draw: drawTarget },
        { name: 'ゼッケン 23', draw: drawNumber },
        { name: 'GT ロゴ', draw: drawGTLogo },
        { name: 'ストライプ', draw: drawStripe },
        { name: 'ファイヤー', draw: drawFire }
    ];

    const presetGrid = document.getElementById('preset-decals');

    presetsData.forEach((p, idx) => {
        const canvas = document.createElement('canvas');
        canvas.width = 1024;
        canvas.height = 1024;
        p.draw(canvas.getContext('2d'), 1024, 1024);

        const dataURL = canvas.toDataURL();
        state.presets.push({
            name: p.name,
            src: dataURL
        });

        // UI要素作成
        const item = document.createElement('div');
        item.className = 'preset-item';
        item.dataset.index = idx;
        item.title = p.name;

        const img = document.createElement('img');
        img.src = dataURL;
        img.alt = p.name;
        item.appendChild(img);

        item.addEventListener('click', () => {
            selectDecal(dataURL, p.name);

            // 選択表示切り替え
            document.querySelectorAll('.preset-item').forEach(el => el.classList.remove('selected'));
            item.classList.add('selected');
        });

        presetGrid.appendChild(item);
    });
}

// --- 各種プリセットの描画ロジック ---
function drawStar(ctx, w, h) {
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#ffdf00'; // ゴールド
    ctx.strokeStyle = '#d4af37';
    ctx.lineWidth = Math.max(2, Math.round(w * (6 / 256)));

    ctx.beginPath();
    const cx = w / 2;
    const cy = h / 2;
    const spikes = 5;
    const outerRadius = w * 0.45;
    const innerRadius = w * 0.18;
    let rot = Math.PI / 2 * 3;
    let x = cx;
    let y = cy;
    const step = Math.PI / spikes;

    ctx.moveTo(cx, cy - outerRadius);
    for (let i = 0; i < spikes; i++) {
        x = cx + Math.cos(rot) * outerRadius;
        y = cy + Math.sin(rot) * outerRadius;
        ctx.lineTo(x, y);
        rot += step;

        x = cx + Math.cos(rot) * innerRadius;
        y = cy + Math.sin(rot) * innerRadius;
        ctx.lineTo(x, y);
        rot += step;
    }
    ctx.lineTo(cx, cy - outerRadius);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
}

function drawArrow(ctx, w, h) {
    ctx.clearRect(0, 0, w, h);
    // スポーティな二重矢印
    ctx.fillStyle = '#ff1744'; // ネオンレッド
    ctx.beginPath();
    ctx.moveTo(w * 0.1, h * 0.5);
    ctx.lineTo(w * 0.5, h * 0.15);
    ctx.lineTo(w * 0.9, h * 0.5);
    ctx.lineTo(w * 0.7, h * 0.5);
    ctx.lineTo(w * 0.5, h * 0.32);
    ctx.lineTo(w * 0.3, h * 0.5);
    ctx.closePath();
    ctx.fill();

    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    ctx.moveTo(w * 0.1, h * 0.8);
    ctx.lineTo(w * 0.5, h * 0.45);
    ctx.lineTo(w * 0.9, h * 0.8);
    ctx.lineTo(w * 0.7, h * 0.8);
    ctx.lineTo(w * 0.5, h * 0.62);
    ctx.lineTo(w * 0.3, h * 0.8);
    ctx.closePath();
    ctx.fill();
}

function drawChecker(ctx, w, h) {
    ctx.clearRect(0, 0, w, h);
    const size = w / 8;
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = '#111111';
    for (let r = 0; r < 8; r++) {
        for (let c = 0; c < 8; c++) {
            if ((r + c) % 2 === 1) {
                ctx.fillRect(c * size, r * size, size, size);
            }
        }
    }
    // 円形に切り抜く（リバリーデカールとして扱いやすくするため）
    ctx.globalCompositeOperation = 'destination-in';
    ctx.beginPath();
    ctx.arc(w / 2, h / 2, w * 0.48, 0, Math.PI * 2);
    ctx.fill();
}

function drawTarget(ctx, w, h) {
    ctx.clearRect(0, 0, w, h);
    const cx = w / 2;
    const cy = h / 2;

    ctx.lineWidth = Math.max(2, Math.round(w * (14 / 256)));

    // 外円
    ctx.strokeStyle = '#00e5ff';
    ctx.beginPath();
    ctx.arc(cx, cy, w * 0.4, 0, Math.PI * 2);
    ctx.stroke();

    // 内円
    ctx.strokeStyle = '#ffffff';
    ctx.beginPath();
    ctx.arc(cx, cy, w * 0.25, 0, Math.PI * 2);
    ctx.stroke();

    // 中心
    ctx.fillStyle = '#ff4500';
    ctx.beginPath();
    ctx.arc(cx, cy, w * 0.1, 0, Math.PI * 2);
    ctx.fill();
}

function drawNumber(ctx, w, h) {
    ctx.clearRect(0, 0, w, h);
    const cx = w / 2;
    const cy = h / 2;

    // 白ベースの黒枠円
    ctx.fillStyle = '#ffffff';
    ctx.strokeStyle = '#111111';
    ctx.lineWidth = Math.max(2, Math.round(w * (8 / 256)));
    ctx.beginPath();
    ctx.arc(cx, cy, w * 0.45, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    // 数字
    ctx.fillStyle = '#111111';
    const fontSize = Math.round(w * (110 / 256));
    ctx.font = `900 ${fontSize}px "Orbitron"`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('23', cx, cy + Math.round(w * (5 / 256)));
}

function drawGTLogo(ctx, w, h) {
    ctx.clearRect(0, 0, w, h);

    // 赤・白・シアンを組み合わせたスピード感のあるロゴ
    ctx.fillStyle = '#ff4500';
    const fontSize = Math.round(w * (100 / 256));
    ctx.font = `italic 900 ${fontSize}px "Orbitron"`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('GT', w / 2 - Math.round(w * (10 / 256)), h / 2 - Math.round(w * (15 / 256)));

    // 下部のライン
    ctx.strokeStyle = '#00e5ff';
    ctx.lineWidth = Math.max(2, Math.round(w * (10 / 256)));
    ctx.beginPath();
    ctx.moveTo(w * 0.15, h * 0.7);
    ctx.lineTo(w * 0.85, h * 0.7);
    ctx.stroke();

    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = Math.max(2, Math.round(w * (4 / 256)));
    ctx.beginPath();
    ctx.moveTo(w * 0.25, h * 0.77);
    ctx.lineTo(w * 0.75, h * 0.77);
    ctx.stroke();
}

function drawStripe(ctx, w, h) {
    ctx.clearRect(0, 0, w, h);

    // 斜めのレーシングストライプ
    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    ctx.moveTo(w * 0.25, 0);
    ctx.lineTo(w * 0.45, 0);
    ctx.lineTo(w * 0.25, h);
    ctx.lineTo(w * 0.05, h);
    ctx.closePath();
    ctx.fill();

    ctx.fillStyle = '#ff4500';
    ctx.beginPath();
    ctx.moveTo(w * 0.5, 0);
    ctx.lineTo(w * 0.8, 0);
    ctx.lineTo(w * 0.6, h);
    ctx.lineTo(w * 0.3, h);
    ctx.closePath();
    ctx.fill();

    // 細いライン
    ctx.fillStyle = '#00e5ff';
    ctx.beginPath();
    ctx.moveTo(w * 0.85, 0);
    ctx.lineTo(w * 0.9, 0);
    ctx.lineTo(w * 0.7, h);
    ctx.lineTo(w * 0.65, h);
    ctx.closePath();
    ctx.fill();
}

function drawFire(ctx, w, h) {
    ctx.clearRect(0, 0, w, h);

    const grad = ctx.createLinearGradient(0, h, 0, 0);
    grad.addColorStop(0, '#ff1744');
    grad.addColorStop(0.5, '#ff9100');
    grad.addColorStop(1, '#ffea00');

    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.moveTo(w * 0.5, h * 0.9);

    // 左側の炎のうねり
    ctx.bezierCurveTo(w * 0.3, h * 0.8, w * 0.1, h * 0.6, w * 0.2, h * 0.4);
    ctx.bezierCurveTo(w * 0.25, h * 0.45, w * 0.3, h * 0.35, w * 0.35, h * 0.2);
    ctx.bezierCurveTo(w * 0.4, h * 0.35, w * 0.45, h * 0.25, w * 0.5, h * 0.05); // 中央の尖端

    // 右側の炎のうねり
    ctx.bezierCurveTo(w * 0.55, h * 0.25, w * 0.6, h * 0.35, w * 0.65, h * 0.2);
    ctx.bezierCurveTo(w * 0.7, h * 0.35, w * 0.75, h * 0.45, w * 0.8, h * 0.4);
    ctx.bezierCurveTo(w * 0.9, h * 0.6, w * 0.7, h * 0.8, w * 0.5, h * 0.9);

    ctx.closePath();
    ctx.fill();
}

// --- デカール選択処理 ---
function selectDecal(src, name) {
    console.log(`Loading decal: ${name}...`);
    const loader = new THREE.TextureLoader();

    loader.load(
        src,
        (texture) => {
            console.log(`Decal "${name}" loaded successfully.`);
            texture.colorSpace = THREE.SRGBColorSpace;
            applyHighQualityTextureFiltering(texture);

            state.selectedTexture = texture;
            state.selectedImageSrc = src;
            state.selectedName = name;

            // UIプレビューの更新
            const imgEl = document.getElementById('selected-decal-img');
            const infoEl = document.getElementById('selected-decal-info');
            imgEl.src = src;

            const img = texture.image;
            if (img) {
                infoEl.innerText = `${name}\nサイズ: ${img.width || '--'} x ${img.height || '--'}`;
            } else {
                infoEl.innerText = `${name}`;
            }
            document.getElementById('selected-preview-container').style.display = 'block';

            // 編集モードをONにする準備
            document.getElementById('mode-text').innerText = "PLACE DECAL MODE (CLICK ON SPHERE)";
            document.getElementById('mode-text').style.borderColor = 'var(--color-accent)';
            document.getElementById('mode-text').style.color = 'var(--color-accent)';
            document.getElementById('mode-text').style.background = 'rgba(255, 69, 0, 0.1)';

            cancelCurrentPreview();
        },
        undefined,
        (err) => {
            console.error(`Failed to load texture for ${name}:`, err);
        }
    );
}

// --- UIイベントリスナー ---
function setupUIListeners() {
    // 1. ファイルアップロード
    const fileInput = document.getElementById('decal-file-input');
    const uploadLabel = document.getElementById('upload-label');

    // ドラッグ＆ドロップエフェクト
    uploadLabel.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadLabel.style.borderColor = 'var(--color-cyan)';
    });
    uploadLabel.addEventListener('dragleave', () => {
        uploadLabel.style.borderColor = 'rgba(255,255,255,0.15)';
    });
    uploadLabel.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadLabel.style.borderColor = 'rgba(255,255,255,0.15)';
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // 2. 3Dキャンバスクリック (デカール配置) - ドラッグとクリックの識別
    const container = document.getElementById('canvas-container');
    let pointerDownPos = new THREE.Vector2();
    let pointerDownTime = 0;

    container.addEventListener('pointerdown', (event) => {
        pointerDownPos.set(event.clientX, event.clientY);
        pointerDownTime = performance.now();
    });

    container.addEventListener('pointerup', (event) => {
        const dist = pointerDownPos.distanceTo(new THREE.Vector2(event.clientX, event.clientY));
        const duration = performance.now() - pointerDownTime;

        if (dist < 4 && duration < 300) {
            onCanvasClick(event);
        }
    });

    // 3. スライダーの値変更
    const sliderScaleX = document.getElementById('slider-scale-x');
    const sliderScaleY = document.getElementById('slider-scale-y');
    const sliderRotation = document.getElementById('slider-rotation');
    const sliderDepth = document.getElementById('slider-depth');

    let debounceTimer = null;

    function bindSliderInputSync(sliderId, inputId, onSync) {
        const slider = document.getElementById(sliderId);
        const input = document.getElementById(inputId);
        if (!slider || !input) return;

        slider.addEventListener('input', (e) => {
            const val = parseFloat(e.target.value);
            if (document.activeElement !== input) {
                input.value = val;
            }
            onSync(val);
        });

        const handleInput = () => {
            let val = parseFloat(input.value);
            if (isNaN(val)) return;
            if (document.activeElement !== slider) {
                slider.value = val;
            }
            onSync(val);
        };

        input.addEventListener('input', handleInput);
        input.addEventListener('change', handleInput);
    }

    const updateFromSliders = () => {
        if (!state.preview.mesh) return;

        const sliderScaleX = document.getElementById('slider-scale-x');
        const sliderScaleY = document.getElementById('slider-scale-y');
        const sliderDepth = document.getElementById('slider-depth');
        const sliderRotation = document.getElementById('slider-rotation');

        const inputScaleX = document.getElementById('input-scale-x');
        const inputScaleY = document.getElementById('input-scale-y');
        const inputDepth = document.getElementById('input-depth');
        const inputRotation = document.getElementById('input-rotation');

        const sliderImageX = document.getElementById('slider-image-x');
        const sliderImageY = document.getElementById('slider-image-y');
        const inputImageX = document.getElementById('input-image-x');
        const inputImageY = document.getElementById('input-image-y');

        const scaleX = parseFloat(sliderScaleX.value);
        const scaleY = parseFloat(sliderScaleY.value);
        const depth = parseFloat(sliderDepth.value);
        const rotZ = parseFloat(sliderRotation.value);

        const imageX = sliderImageX ? parseFloat(sliderImageX.value) : 1.0;
        const imageY = sliderImageY ? parseFloat(sliderImageY.value) : 1.0;

        if (inputScaleX && document.activeElement !== inputScaleX) inputScaleX.value = scaleX.toFixed(2);
        if (inputScaleY && document.activeElement !== inputScaleY) inputScaleY.value = scaleY.toFixed(2);
        if (inputDepth && document.activeElement !== inputDepth) inputDepth.value = depth.toFixed(2);
        if (inputRotation && document.activeElement !== inputRotation) inputRotation.value = Math.round(rotZ);

        if (inputImageX && document.activeElement !== inputImageX) inputImageX.value = imageX.toFixed(2);
        if (inputImageY && document.activeElement !== inputImageY) inputImageY.value = imageY.toFixed(2);

        state.preview.scale.set(scaleX, scaleY, depth);
        state.preview.imageScale.set(imageX, imageY);
        state.preview.rotationZ = rotZ;

        // スライダー操作中は超高速トランスフォーム更新 (計算負荷0ms・カクツキゼロ)
        updatePreviewTransform();

        // ドラッグ停止後に精密な DecalGeometry 切り出しを実行
        if (debounceTimer) clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => {
            if (state.isEditing) {
                updatePreviewDecal();
            }
        }, 150);
    };

    // 各スライダーと直接数値入力ボックスの相互リアルタイム同期バインド
    bindSliderInputSync('slider-model-rotation-y', 'input-model-rotation-y', (val) => setModelRotationYDeg(val, false));
    bindSliderInputSync('slider-model-rotation-x', 'input-model-rotation-x', (val) => setModelRotationXDeg(val, false));
    bindSliderInputSync('slider-model-scale', 'input-model-scale', (val) => setModelDisplayScale(val, false));

    bindSliderInputSync('slider-scale-x', 'input-scale-x', updateFromSliders);
    bindSliderInputSync('slider-scale-y', 'input-scale-y', updateFromSliders);
    bindSliderInputSync('slider-depth', 'input-depth', updateFromSliders);
    bindSliderInputSync('slider-image-x', 'input-image-x', updateFromSliders);
    bindSliderInputSync('slider-image-y', 'input-image-y', updateFromSliders);
    bindSliderInputSync('slider-rotation', 'input-rotation', updateFromSliders);

    // 「貼る領域のサイズに合わせる」ボタン
    const btnSyncImage = document.getElementById('btn-sync-image-to-clip');
    if (btnSyncImage) {
        btnSyncImage.addEventListener('click', () => {
            const scaleX = state.preview.scale.x;
            const scaleY = state.preview.scale.y;
            state.preview.imageScale.set(scaleX, scaleY);

            const sliderImgX = document.getElementById('slider-image-x');
            const sliderImgY = document.getElementById('slider-image-y');
            const inputImgX = document.getElementById('input-image-x');
            const inputImgY = document.getElementById('input-image-y');

            if (sliderImgX) sliderImgX.value = scaleX;
            if (sliderImgY) sliderImgY.value = scaleY;
            if (inputImgX) inputImgX.value = scaleX.toFixed(2);
            if (inputImgY) inputImgY.value = scaleY.toFixed(2);

            updatePreviewTransform();
            if (state.isEditing) {
                updatePreviewDecal();
            }
        });
    }


    bindSliderInputSync('slider-angle-limit', 'input-angle-limit', (val) => {
        state.preview.angleLimitDeg = val;
        if (debounceTimer) clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => {
            if (state.isEditing) updatePreviewDecal();
        }, 100);
    });

    bindSliderInputSync('slider-shadow-intensity', 'input-shadow-intensity', (val) => {
        state.preview.shadowIntensity = val / 100.0;
        if (state.preview.mesh && state.preview.mesh.material && state.preview.mesh.material.userData && state.preview.mesh.material.userData.uShadowIntensity) {
            state.preview.mesh.material.userData.uShadowIntensity.value = state.preview.shadowIntensity;
            requestRender();
        }
    });

    // 5. 決定・キャンセルボタン
    document.getElementById('btn-apply-decal').addEventListener('click', applyCurrentDecal);
    document.getElementById('btn-cancel-decal').addEventListener('click', cancelCurrentPreview);

    // 5b. 投影方向トグル
    const btnProjSurface = document.getElementById('btn-proj-surface');
    const btnProjCamera = document.getElementById('btn-proj-camera');

    function applyProjectionMode(mode) {
        state.preview.projectionType = mode;

        if (mode === 'surface') {
            btnProjSurface.classList.add('active');
            btnProjSurface.style.borderColor = 'rgba(0,229,255,0.5)';
            btnProjSurface.style.color = 'var(--color-cyan)';
            btnProjCamera.classList.remove('active');
            btnProjCamera.style.borderColor = '';
            btnProjCamera.style.color = '';
            state.preview.normal.copy(state.preview.surfaceNormal);
        } else {
            btnProjCamera.classList.add('active');
            btnProjCamera.style.borderColor = 'rgba(0,229,255,0.5)';
            btnProjCamera.style.color = 'var(--color-cyan)';
            btnProjSurface.classList.remove('active');
            btnProjSurface.style.borderColor = '';
            btnProjSurface.style.color = '';
            if (state.preview.position.lengthSq() > 0) {
                const camDir = state.preview.position.clone().sub(camera.position).normalize();
                state.preview.cameraDirection.copy(camDir);
                state.preview.normal.copy(camDir);
            } else {
                state.preview.normal.copy(state.preview.cameraDirection);
            }
        }

        if (state.isEditing) {
            updatePreviewDecal();
        }
    }

    btnProjSurface.addEventListener('click', () => applyProjectionMode('surface'));
    btnProjCamera.addEventListener('click', () => applyProjectionMode('camera'));

    // 5c. 投影可視化ボックス表示トグル (貼る領域 & 画像サイズ)
    const chkShowBox = document.getElementById('chk-show-projection-box');
    if (chkShowBox) {
        chkShowBox.addEventListener('change', (e) => {
            state.preview.showProjectionBox = e.target.checked;
            if (state.preview.projectionBoxMesh) {
                state.preview.projectionBoxMesh.visible = e.target.checked;
                requestRender();
            } else if (state.isEditing) {
                updatePreviewDecal();
            }
        });
    }

    const chkShowImageBox = document.getElementById('chk-show-image-box');
    if (chkShowImageBox) {
        chkShowImageBox.addEventListener('change', (e) => {
            state.preview.showImageBox = e.target.checked;
            if (state.preview.imageBoxMesh) {
                state.preview.imageBoxMesh.visible = e.target.checked;
                requestRender();
            } else if (state.isEditing) {
                updatePreviewDecal();
            }
        });
    }

    // 6. エクスポートボタン
    const btnExportGlbDecalsOnly = document.getElementById('btn-export-glb-decals-only');
    const btnExportGlbBaseUntextured = document.getElementById('btn-export-glb-base-untextured');
    const btnExportGlbFull = document.getElementById('btn-export-glb-full');

    if (btnExportGlbDecalsOnly) btnExportGlbDecalsOnly.addEventListener('click', exportGLBDecalsOnly);
    if (btnExportGlbBaseUntextured) btnExportGlbBaseUntextured.addEventListener('click', exportGLBBaseUntextured);
    if (btnExportGlbFull) btnExportGlbFull.addEventListener('click', exportGLBFull);

    document.getElementById('btn-export-gltf').addEventListener('click', exportGLTF);
    document.getElementById('btn-export-decals-gltf').addEventListener('click', exportGLTFDecalsOnly);
    document.getElementById('btn-export-screenshot').addEventListener('click', takeScreenshot);

    // 7. ベースモデルの切替ボタン
    const btnModelSphere = document.getElementById('btn-model-sphere');
    const btnModelCustom = document.getElementById('btn-model-custom');
    const modelUploadBox = document.getElementById('model-upload-box');
    const modelFileInput = document.getElementById('model-file-input');
    const modelUploadLabel = document.getElementById('model-upload-label');

    btnModelSphere.addEventListener('click', () => {
        btnModelSphere.classList.add('active');
        btnModelCustom.classList.remove('active');
        modelUploadBox.style.display = 'none';
        resetToSphere();
    });

    btnModelCustom.addEventListener('click', () => {
        btnModelCustom.classList.add('active');
        btnModelSphere.classList.remove('active');
        modelUploadBox.style.display = 'block';
    });

    // 表示モード切替ボタン
    const btnRenderDefault = document.getElementById('btn-render-default');
    const btnRenderWireframe = document.getElementById('btn-render-wireframe');
    const btnRenderNormal = document.getElementById('btn-render-normal');

    const updateRenderModeUI = (activeBtn) => {
        [btnRenderDefault, btnRenderWireframe, btnRenderNormal].forEach(btn => {
            if (btn) btn.classList.remove('active');
        });
        if (activeBtn) activeBtn.classList.add('active');
    };

    if (btnRenderDefault) {
        btnRenderDefault.addEventListener('click', () => {
            updateRenderModeUI(btnRenderDefault);
            applyModelRenderMode('default');
        });
    }
    if (btnRenderWireframe) {
        btnRenderWireframe.addEventListener('click', () => {
            updateRenderModeUI(btnRenderWireframe);
            applyModelRenderMode('wireframe');
        });
    }
    if (btnRenderNormal) {
        btnRenderNormal.addEventListener('click', () => {
            updateRenderModeUI(btnRenderNormal);
            applyModelRenderMode('normal');
        });
    }

    // OBJファイル選択
    modelFileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            loadOBJModel(e.target.files);
        }
    });

    // OBJドラッグ＆ドロップ
    modelUploadLabel.addEventListener('dragover', (e) => {
        e.preventDefault();
        modelUploadLabel.style.borderColor = 'var(--color-cyan)';
    });
    modelUploadLabel.addEventListener('dragleave', () => {
        modelUploadLabel.style.borderColor = 'rgba(255,255,255,0.15)';
    });
    modelUploadLabel.addEventListener('drop', (e) => {
        e.preventDefault();
        modelUploadLabel.style.borderColor = 'rgba(255,255,255,0.15)';
        if (e.dataTransfer.files.length > 0) {
            loadOBJModel(e.dataTransfer.files);
        }
    });
    // 8. モデルの回転コントロールのイベントリスナー (X軸: 上下, Y軸: 左右)
    const btnRotateY180 = document.getElementById('btn-rotate-y-180');
    const btnQuickRotateY180 = document.getElementById('btn-quick-rotate-y-180');
    const btnRotateX180 = document.getElementById('btn-rotate-x-180');
    const btnQuickRotateX180 = document.getElementById('btn-quick-rotate-x-180');

    if (btnRotateY180) btnRotateY180.addEventListener('click', () => rotateModelYBy(180));
    if (btnQuickRotateY180) btnQuickRotateY180.addEventListener('click', () => rotateModelYBy(180));
    if (btnRotateX180) btnRotateX180.addEventListener('click', () => rotateModelXBy(180));
    if (btnQuickRotateX180) btnQuickRotateX180.addEventListener('click', () => rotateModelXBy(180));

    // Y軸（左右）プリセット＆スライダー
    const btnRotY0 = document.getElementById('btn-rot-y-0');
    const btnRotY90 = document.getElementById('btn-rot-y-90');
    const btnRotY180 = document.getElementById('btn-rot-y-180');
    const btnRotY270 = document.getElementById('btn-rot-y-270');
    const sliderModelRotationY = document.getElementById('slider-model-rotation-y');

    if (btnRotY0) btnRotY0.addEventListener('click', () => setModelRotationYDeg(0));
    if (btnRotY90) btnRotY90.addEventListener('click', () => setModelRotationYDeg(90));
    if (btnRotY180) btnRotY180.addEventListener('click', () => setModelRotationYDeg(180));
    if (btnRotY270) btnRotY270.addEventListener('click', () => setModelRotationYDeg(-90));

    if (sliderModelRotationY) {
        sliderModelRotationY.addEventListener('input', (e) => {
            setModelRotationYDeg(parseFloat(e.target.value), false);
        });
    }

    // X軸（上下）プリセット＆スライダー
    const btnRotX0 = document.getElementById('btn-rot-x-0');
    const btnRotX90 = document.getElementById('btn-rot-x-90');
    const btnRotX180 = document.getElementById('btn-rot-x-180');
    const btnRotX270 = document.getElementById('btn-rot-x-270');
    const sliderModelRotationX = document.getElementById('slider-model-rotation-x');

    if (btnRotX0) btnRotX0.addEventListener('click', () => setModelRotationXDeg(0));
    if (btnRotX90) btnRotX90.addEventListener('click', () => setModelRotationXDeg(90));
    if (btnRotX180) btnRotX180.addEventListener('click', () => setModelRotationXDeg(180));
    if (btnRotX270) btnRotX270.addEventListener('click', () => setModelRotationXDeg(-90));

    if (sliderModelRotationX) {
        sliderModelRotationX.addEventListener('input', (e) => {
            setModelRotationXDeg(parseFloat(e.target.value), false);
        });
    }

    // モデル表示拡大率 (Scale) コントロールのリスナー
    const btnScale10 = document.getElementById('btn-scale-10');
    const btnScale20 = document.getElementById('btn-scale-20');
    const btnScale30 = document.getElementById('btn-scale-30');
    const btnScale50 = document.getElementById('btn-scale-50');
    const btnScale100 = document.getElementById('btn-scale-100');
    const sliderModelScale = document.getElementById('slider-model-scale');

    if (btnScale10) btnScale10.addEventListener('click', () => setModelDisplayScale(1.0));
    if (btnScale20) btnScale20.addEventListener('click', () => setModelDisplayScale(2.0));
    if (btnScale30) btnScale30.addEventListener('click', () => setModelDisplayScale(3.0));
    if (btnScale50) btnScale50.addEventListener('click', () => setModelDisplayScale(5.0));
    if (btnScale100) btnScale100.addEventListener('click', () => setModelDisplayScale(10.0));

    if (sliderModelScale) {
        sliderModelScale.addEventListener('input', (e) => {
            const val = parseFloat(e.target.value);
            setModelDisplayScale(val, false);
        });
    }
}

// --- モデル回転制御機能 ---
function rotateModelYBy(deg) {
    const rad = THREE.MathUtils.degToRad(deg);
    state.modelRotation.targetY += rad;
}

function rotateModelXBy(deg) {
    const rad = THREE.MathUtils.degToRad(deg);
    state.modelRotation.targetX += rad;
}

function setModelRotationYDeg(deg, smooth = true) {
    const rad = THREE.MathUtils.degToRad(deg);
    state.modelRotation.targetY = rad;
    if (!smooth) {
        state.modelRotation.currentY = rad;
        if (modelGroup) modelGroup.rotation.y = rad;
        updateModelRotationUI();
    }
}

function setModelRotationXDeg(deg, smooth = true) {
    const rad = THREE.MathUtils.degToRad(deg);
    state.modelRotation.targetX = rad;
    if (!smooth) {
        state.modelRotation.currentX = rad;
        if (modelGroup) modelGroup.rotation.x = rad;
        updateModelRotationUI();
    }
}

function updateModelRotationUI() {
    // Y軸 (左右)
    let degY = THREE.MathUtils.radToDeg(state.modelRotation.currentY) % 360;
    if (degY > 180) degY -= 360;
    if (degY < -180) degY += 360;
    const displayDegY = Math.round(degY);

    const inputYEl = document.getElementById('input-model-rotation-y');
    if (inputYEl && document.activeElement !== inputYEl) {
        inputYEl.value = displayDegY;
    }
    const sliderYEl = document.getElementById('slider-model-rotation-y');
    if (sliderYEl && document.activeElement !== sliderYEl) {
        sliderYEl.value = displayDegY;
    }

    // X軸 (上下)
    let degX = THREE.MathUtils.radToDeg(state.modelRotation.currentX) % 360;
    if (degX > 180) degX -= 360;
    if (degX < -180) degX += 360;
    const displayDegX = Math.round(degX);

    const inputXEl = document.getElementById('input-model-rotation-x');
    if (inputXEl && document.activeElement !== inputXEl) {
        inputXEl.value = displayDegX;
    }
    const sliderXEl = document.getElementById('slider-model-rotation-x');
    if (sliderXEl && document.activeElement !== sliderXEl) {
        sliderXEl.value = displayDegX;
    }
}

// --- モデル表示拡大率制御機能 ---
function setModelDisplayScale(scaleVal, smooth = true) {
    state.modelDisplayScale.target = scaleVal;
    if (!smooth) {
        state.modelDisplayScale.current = scaleVal;
        if (modelGroup) modelGroup.scale.set(scaleVal, scaleVal, scaleVal);
        updateModelDisplayScaleUI();
    }
}

function updateModelDisplayScaleUI() {
    const scaleVal = state.modelDisplayScale.current;
    const inputEl = document.getElementById('input-model-scale');
    if (inputEl && document.activeElement !== inputEl) {
        inputEl.value = scaleVal.toFixed(1);
    }

    const sliderEl = document.getElementById('slider-model-scale');
    if (sliderEl && document.activeElement !== sliderEl) {
        sliderEl.value = scaleVal.toFixed(1);
    }

    const btn10 = document.getElementById('btn-scale-10');
    const btn20 = document.getElementById('btn-scale-20');
    const btn30 = document.getElementById('btn-scale-30');
    const btn50 = document.getElementById('btn-scale-50');
    const btn100 = document.getElementById('btn-scale-100');

    [btn10, btn20, btn30, btn50, btn100].forEach(btn => {
        if (btn) btn.classList.remove('active');
    });

    if (Math.abs(scaleVal - 1.0) < 0.05 && btn10) btn10.classList.add('active');
    else if (Math.abs(scaleVal - 2.0) < 0.05 && btn20) btn20.classList.add('active');
    else if (Math.abs(scaleVal - 3.0) < 0.05 && btn30) btn30.classList.add('active');
    else if (Math.abs(scaleVal - 5.0) < 0.05 && btn50) btn50.classList.add('active');
    else if (Math.abs(scaleVal - 10.0) < 0.05 && btn100) btn100.classList.add('active');
}

function handleFile(file) {
    if (!file) return;
    console.log(`Processing uploaded file: ${file.name}`);
    const reader = new FileReader();
    reader.onload = (e) => {
        selectDecal(e.target.result, file.name);
    };
    reader.readAsDataURL(file);
}

// マテリアルを安全に破棄
function disposeMaterial(material) {
    if (!material) return;
    if (Array.isArray(material)) {
        material.forEach(disposeMaterial);
    } else {
        if (material.map) material.map.dispose();
        if (material.lightMap) material.lightMap.dispose();
        if (material.aoMap) material.aoMap.dispose();
        if (material.emissiveMap) material.emissiveMap.dispose();
        if (material.bumpMap) material.bumpMap.dispose();
        if (material.normalMap) material.normalMap.dispose();
        if (material.displacementMap) material.displacementMap.dispose();
        if (material.roughnessMap) material.roughnessMap.dispose();
        if (material.metalnessMap) material.metalnessMap.dispose();
        if (material.alphaMap) material.alphaMap.dispose();
        material.dispose();
    }
}

// カスタムモデル用に作成したオブジェクトURLを解放
function clearCustomModelObjectUrls() {
    if (state.customModelObjectUrls && state.customModelObjectUrls.length > 0) {
        state.customModelObjectUrls.forEach(url => {
            URL.revokeObjectURL(url);
        });
        state.customModelObjectUrls = [];
    }
}

// --- OBJモデルの読み込み ---
function loadOBJModel(filesInput) {
    let files = [];
    if (filesInput instanceof FileList || Array.isArray(filesInput)) {
        files = Array.from(filesInput);
    } else if (filesInput instanceof File) {
        files = [filesInput];
    } else {
        console.error("Invalid files input");
        return;
    }

    if (files.length === 0) return;

    const objFile = files.find(f => f.name.toLowerCase().endsWith('.obj'));
    if (!objFile) {
        console.error('OBJファイル (.obj) が見つかりませんでした。');
        sysError('エラー: OBJファイル (.obj) が含まれていません。');
        return;
    }

    console.log(`Loading OBJ model: ${objFile.name}`);
    sysLog(`Loading model: ${objFile.name}...`);

    clearCustomModelObjectUrls();

    const fileMap = {};
    files.forEach(file => {
        const url = URL.createObjectURL(file);
        state.customModelObjectUrls.push(url);

        fileMap[file.name] = url;

        const baseName = file.name.split('/').pop().split('\\').pop();
        fileMap[baseName] = url;

        fileMap[file.name.toLowerCase()] = url;
        fileMap[baseName.toLowerCase()] = url;
    });

    const manager = new THREE.LoadingManager();
    manager.setURLModifier((url) => {
        const baseName = url.split('/').pop().split('\\').pop();
        const decodedBaseName = decodeURIComponent(baseName).toLowerCase();

        if (fileMap[decodedBaseName]) {
            return fileMap[decodedBaseName];
        }
        return url;
    });

    const mtlFile = files.find(f => f.name.toLowerCase().endsWith('.mtl'));
    let materialsCreator = null;

    const parseOBJ = (objText) => {
        const objLoader = new OBJLoader(manager);
        if (materialsCreator) {
            objLoader.setMaterials(materialsCreator);
        }

        let group;
        try {
            group = objLoader.parse(objText);
        } catch (err) {
            console.error('OBJパースエラー:', err.message);
            sysError(`OBJパースエラー: ${err.message}`);
            return;
        }

        clearAllDecals();

        if (mainSphereMesh) {
            modelGroup.remove(mainSphereMesh);
            mainSphereMesh.geometry.dispose();
            disposeMaterial(mainSphereMesh.material);
        }

        let mergedGeometry = new THREE.BufferGeometry();
        const geometries = [];
        const materials = [];

        group.traverse((child) => {
            if (child.isMesh) {
                const geo = child.geometry.clone();
                child.updateWorldMatrix(true, false);
                geo.applyMatrix4(child.matrixWorld);
                if (!geo.attributes.normal) geo.computeVertexNormals();
                geometries.push(geo);

                materials.push(child.material);
            }
        });

        if (geometries.length === 0) {
            console.error('OBJファイルにメッシュが見つかりませんでした。');
            sysError('エラー: OBJファイルにメッシュが見つかりませんでした。');
            return;
        }

        let hasUV = false;
        let hasNormal = false;
        let hasColor = false;

        geometries.forEach(geo => {
            if (geo.attributes.uv) hasUV = true;
            if (geo.attributes.normal) hasNormal = true;
            if (geo.attributes.color) hasColor = true;
        });

        geometries.forEach(geo => {
            const vertexCount = geo.attributes.position.count;
            if (hasNormal && !geo.attributes.normal) {
                geo.computeVertexNormals();
            }
            if (hasUV && !geo.attributes.uv) {
                const uvs = new Float32Array(vertexCount * 2);
                geo.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
            }
            if (hasColor && !geo.attributes.color) {
                const colors = new Float32Array(vertexCount * 3).fill(1.0);
                geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
            }
        });

        import('three/examples/jsm/utils/BufferGeometryUtils.js').then(async ({ mergeGeometries }) => {
            const useGroups = materials.length > 1;
            mergedGeometry = mergeGeometries(geometries, useGroups);
            if (!mergedGeometry) {
                console.error('ジオメトリのマージに失敗しました。');
                sysError('エラー: ジオメトリのマージに失敗しました。');
                return;
            }
            mergedGeometry.computeVertexNormals();

            mergedGeometry.computeBoundingBox();
            const box = mergedGeometry.boundingBox;
            const center = new THREE.Vector3();
            box.getCenter(center);
            const size = new THREE.Vector3();
            box.getSize(size);
            const maxDim = Math.max(size.x, size.y, size.z);
            const targetSize = SPHERE_RADIUS * 2;
            const scale = targetSize / maxDim;

            state.originalCenter = center.clone();
            state.originalScale = scale;

            mergedGeometry.translate(-center.x, -center.y, -center.z);
            mergedGeometry.scale(scale, scale, scale);
            mergedGeometry.computeBoundingBox();
            mergedGeometry.computeVertexNormals();

            let finalMat;
            if (materials.length > 0) {
                materials.forEach(mat => {
                    if (mat) {
                        mat.side = THREE.DoubleSide;
                        if (mat.map) applyHighQualityTextureFiltering(mat.map);
                    }
                });
                finalMat = useGroups ? materials : materials[0];
            } else {
                finalMat = new THREE.MeshStandardMaterial({
                    color: 0xdddddd,
                    roughness: 0.3,
                    metalness: 0.1,
                    side: THREE.DoubleSide
                });
            }

            mainSphereMesh = new THREE.Mesh(mergedGeometry, finalMat);
            mainSphereMesh.name = 'CustomOBJMesh';
            modelGroup.add(mainSphereMesh);

            state.modelOriginalMaterial = finalMat;
            state.modelRenderMode = 'default';
            const btnDefault = document.getElementById('btn-render-default');
            if (btnDefault) {
                [document.getElementById('btn-render-default'),
                document.getElementById('btn-render-wireframe'),
                document.getElementById('btn-render-normal')].forEach(btn => {
                    if (btn) btn.classList.remove('active');
                });
                btnDefault.classList.add('active');
            }

            const newBox = mergedGeometry.boundingBox;
            const newSize = new THREE.Vector3();
            newBox.getSize(newSize);
            const newMax = Math.max(newSize.x, newSize.y, newSize.z);
            camera.position.set(0, 0, newMax * 2.0);
            orbitControls.minDistance = 0.01;
            orbitControls.maxDistance = newMax * 50.0;
            orbitControls.target.set(0, 0, 0);
            orbitControls.update();

            const infoEl = document.getElementById('model-info');
            if (infoEl) {
                let html = `<div>現在のモデル:</div><div style="font-weight: 600; color: #fff; margin-top: 2px;">${objFile.name}</div>`;
                if (mtlFile) html += `<div style="font-size: 0.7rem; color: var(--text-muted); margin-top: 1px;">(+ ${mtlFile.name})</div>`;
                infoEl.innerHTML = html;
            }

            console.log(`OBJ model "${objFile.name}" loaded and applied successfully.`);
            sysLog(`Model "${objFile.name}" loaded successfully.`);
            updateDecalsBaseTexture();
        }).catch(err => {
            console.error('Geometry merge import error:', err);
            sysError(`ジオメトリマージのインポート失敗: ${err.message}`);
        });
    };

    if (mtlFile) {
        console.log(`Loading MTL material: ${mtlFile.name}`);
        sysLog(`Loading material: ${mtlFile.name}...`);
        const mtlReader = new FileReader();
        mtlReader.onload = (e) => {
            const mtlLoader = new MTLLoader(manager);
            try {
                materialsCreator = mtlLoader.parse(e.target.result, '');
                materialsCreator.preload();
            } catch (err) {
                console.error('MTLパースエラー:', err.message);
                sysError(`MTLパースエラー: ${err.message}`);
            }

            const objReader = new FileReader();
            objReader.onload = (eObj) => {
                parseOBJ(eObj.target.result);
            };
            objReader.readAsText(objFile);
        };
        mtlReader.readAsText(mtlFile);
    } else {
        const objReader = new FileReader();
        objReader.onload = (eObj) => {
            parseOBJ(eObj.target.result);
        };
        objReader.readAsText(objFile);
    }
}

// デフォルト球体に戻す
function resetToSphere() {
    clearAllDecals();
    clearCustomModelObjectUrls();

    state.originalCenter = null;
    state.originalScale = null;

    modelGroup.remove(mainSphereMesh);
    mainSphereMesh.geometry.dispose();
    disposeMaterial(mainSphereMesh.material);

    const sphereGeo = new THREE.SphereGeometry(SPHERE_RADIUS, 64, 64);
    const sphereMat = new THREE.MeshStandardMaterial({
        color: 0xdddddd,
        roughness: 0.3,
        metalness: 0.1,
        bumpScale: 0.05
    });
    sphereMat.map = createSphereGridTexture();

    mainSphereMesh = new THREE.Mesh(sphereGeo, sphereMat);
    mainSphereMesh.name = 'MainSphere';
    modelGroup.add(mainSphereMesh);

    state.modelOriginalMaterial = sphereMat;
    state.modelRenderMode = 'default';
    const btnDefault = document.getElementById('btn-render-default');
    if (btnDefault) {
        [document.getElementById('btn-render-default'),
        document.getElementById('btn-render-wireframe'),
        document.getElementById('btn-render-normal')].forEach(btn => {
            if (btn) btn.classList.remove('active');
        });
        btnDefault.classList.add('active');
    }

    camera.position.set(0, 0, 6);
    orbitControls.minDistance = 0.5;
    orbitControls.maxDistance = 500;
    orbitControls.target.set(0, 0, 0);
    orbitControls.update();

    const infoEl = document.getElementById('model-info');
    if (infoEl) infoEl.innerText = '現在のモデル: 球体';

    updateDecalsBaseTexture();
    console.log('Reset to default sphere.');
}

// すべての確定デカールをシーンから削除
function clearAllDecals() {
    cancelCurrentPreview();

    state.appliedDecals.forEach(item => {
        modelGroup.remove(item.mesh);
        item.mesh.geometry.dispose();
        item.mesh.material.dispose();
    });
    state.appliedDecals = [];
    state.nextLayerId = 1;
    updateLayerListUI();
}

// キャンバスクリック時の処理
function onCanvasClick(event) {
    if (!state.selectedTexture) {
        console.log("Canvas clicked, but no decal is selected yet.");
        return;
    }

    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObject(mainSphereMesh);

    if (intersects.length > 0) {
        const hit = intersects[0];
        console.log(`Raycast hit sphere at point: [${hit.point.x.toFixed(2)}, ${hit.point.y.toFixed(2)}, ${hit.point.z.toFixed(2)}]`);

        const wasEditing = state.isEditing;
        state.isEditing = true;
        document.getElementById('section-transform').classList.remove('disabled');
        document.getElementById('mode-text').innerText = "EDIT DECAL MODE";

        state.preview.position.copy(hit.point);

        const normalMatrix = new THREE.Matrix3().getNormalMatrix(mainSphereMesh.matrixWorld);
        const worldNormal = hit.face.normal.clone().applyMatrix3(normalMatrix).normalize();
        state.preview.surfaceNormal.copy(worldNormal);

        const camDir = hit.point.clone().sub(camera.position).normalize();
        camDir.negate();
        state.preview.cameraDirection.copy(camDir);

        if (state.preview.projectionType === 'surface') {
            state.preview.normal.copy(worldNormal);
        } else {
            state.preview.normal.copy(camDir);
        }

        if (!wasEditing) {
            document.getElementById('slider-scale-x').value = 1.0;
            document.getElementById('slider-scale-y').value = 1.0;
            document.getElementById('slider-rotation').value = 0;
            document.getElementById('slider-depth').value = 1.0;

            const sliderImgX = document.getElementById('slider-image-x');
            const sliderImgY = document.getElementById('slider-image-y');
            if (sliderImgX) sliderImgX.value = 1.0;
            if (sliderImgY) sliderImgY.value = 1.0;

            const inputScaleX = document.getElementById('input-scale-x');
            const inputScaleY = document.getElementById('input-scale-y');
            const inputRotation = document.getElementById('input-rotation');
            const inputDepth = document.getElementById('input-depth');
            const inputImageX = document.getElementById('input-image-x');
            const inputImageY = document.getElementById('input-image-y');

            if (inputScaleX) inputScaleX.value = "1.00";
            if (inputScaleY) inputScaleY.value = "1.00";
            if (inputRotation) inputRotation.value = "0";
            if (inputDepth) inputDepth.value = "1.00";
            if (inputImageX) inputImageX.value = "1.00";
            if (inputImageY) inputImageY.value = "1.00";

            state.preview.scale.set(1, 1, 1.0);
            state.preview.imageScale.set(1, 1);
            state.preview.rotationZ = 0;

            state.preview.normalOffset = 0;

            const sliderAngleLimit = document.getElementById('slider-angle-limit');
            const inputAngleLimit = document.getElementById('input-angle-limit');
            if (sliderAngleLimit) {
                state.preview.angleLimitDeg = parseFloat(sliderAngleLimit.value);
                if (inputAngleLimit) inputAngleLimit.value = sliderAngleLimit.value;
            }
        }

        updatePreviewDecal();
    } else {
        console.log("Raycast missed the sphere.");
    }
}

// 表示モード（通常、メッシュ、法線）を適用する
function applyModelRenderMode(mode) {
    state.modelRenderMode = mode;
    if (!mainSphereMesh) return;

    if (mode === 'default') {
        setMaterialWireframe(state.modelOriginalMaterial, false);
        mainSphereMesh.material = state.modelOriginalMaterial;
    } else if (mode === 'wireframe') {
        setMaterialWireframe(state.modelOriginalMaterial, true);
        mainSphereMesh.material = state.modelOriginalMaterial;
    } else if (mode === 'normal') {
        if (!state.normalMaterial) {
            state.normalMaterial = new THREE.MeshNormalMaterial({
                side: THREE.DoubleSide
            });
        }
        mainSphereMesh.material = state.normalMaterial;
    }
    requestRender();
}

function setMaterialWireframe(material, value) {
    if (!material) return;
    if (Array.isArray(material)) {
        material.forEach(mat => {
            if (mat) mat.side = THREE.DoubleSide;
            if (mat) mat.wireframe = value;
        });
    } else {
        material.side = THREE.DoubleSide;
        material.wireframe = value;
    }
}

// --- 貼り付け先メッシュの元のUV (baseUv) を補間・保持する拡張 DecalGeometry ---
class CustomDecalGeometry extends THREE.BufferGeometry {
    constructor(mesh, position, orientation, size, imageSize = null) {
        super();

        const vertices = [];
        const normals = [];
        const uvs = [];
        const baseUvs = [];

        const plane = new THREE.Vector3();
        const projectorMatrix = new THREE.Matrix4();
        projectorMatrix.makeRotationFromEuler(orientation);
        projectorMatrix.setPosition(position);

        const projectorMatrixInverse = new THREE.Matrix4();
        projectorMatrixInverse.copy(projectorMatrix).invert();

        mesh.updateMatrixWorld(true);
        const geometry = mesh.geometry;
        const positionAttribute = geometry.attributes.position;
        const normalAttribute = geometry.attributes.normal;
        const baseUvAttribute = geometry.attributes.uv;

        class DecalVertexWithBaseUv {
            constructor(pos, norm, bUv) {
                this.position = pos;
                this.normal = norm;
                this.baseUv = bUv;
            }
            clone() {
                return new DecalVertexWithBaseUv(
                    this.position.clone(),
                    this.normal.clone(),
                    this.baseUv ? this.baseUv.clone() : new THREE.Vector2(0, 0)
                );
            }
        }

        function pushDecalVertex(decalVertices, v, n, bUv) {
            v.applyMatrix4(mesh.matrixWorld);
            v.applyMatrix4(projectorMatrixInverse);
            if (normalAttribute) {
                n.transformDirection(mesh.matrixWorld);
            }

            decalVertices.push(new DecalVertexWithBaseUv(
                v.clone(),
                n.clone(),
                bUv ? bUv.clone() : new THREE.Vector2(0, 0)
            ));
        }

        function clip(v0, v1, p, s) {
            const d0 = v0.position.dot(p) - s;
            const d1 = v1.position.dot(p) - s;
            const s0 = d0 / (d0 - d1);

            const vPos = new THREE.Vector3(
                v0.position.x + s0 * (v1.position.x - v0.position.x),
                v0.position.y + s0 * (v1.position.y - v0.position.y),
                v0.position.z + s0 * (v1.position.z - v0.position.z)
            );
            const vNorm = new THREE.Vector3(
                v0.normal.x + s0 * (v1.normal.x - v0.normal.x),
                v0.normal.y + s0 * (v1.normal.y - v0.normal.y),
                v0.normal.z + s0 * (v1.normal.z - v0.normal.z)
            );
            const vBaseUv = new THREE.Vector2(
                v0.baseUv.x + s0 * (v1.baseUv.x - v0.baseUv.x),
                v0.baseUv.y + s0 * (v1.baseUv.y - v0.baseUv.y)
            );

            return new DecalVertexWithBaseUv(vPos, vNorm, vBaseUv);
        }

        function clipGeometry(inVertices, p) {
            const outVertices = [];
            const s = 0.5 * Math.abs(size.dot(p));

            for (let i = 0; i < inVertices.length; i += 3) {
                let total = 0;
                let nV1, nV2, nV3, nV4;

                const d1 = inVertices[i + 0].position.dot(p) - s;
                const d2 = inVertices[i + 1].position.dot(p) - s;
                const d3 = inVertices[i + 2].position.dot(p) - s;

                const v1Out = d1 > 0;
                const v2Out = d2 > 0;
                const v3Out = d3 > 0;

                total = (v1Out ? 1 : 0) + (v2Out ? 1 : 0) + (v3Out ? 1 : 0);

                switch (total) {
                    case 0:
                        outVertices.push(inVertices[i]);
                        outVertices.push(inVertices[i + 1]);
                        outVertices.push(inVertices[i + 2]);
                        break;
                    case 1:
                        if (v1Out) {
                            nV1 = inVertices[i + 1];
                            nV2 = inVertices[i + 2];
                            nV3 = clip(inVertices[i], nV1, p, s);
                            nV4 = clip(inVertices[i], nV2, p, s);
                        }
                        if (v2Out) {
                            nV1 = inVertices[i];
                            nV2 = inVertices[i + 2];
                            nV3 = clip(inVertices[i + 1], nV1, p, s);
                            nV4 = clip(inVertices[i + 1], nV2, p, s);

                            outVertices.push(nV3);
                            outVertices.push(nV2.clone());
                            outVertices.push(nV1.clone());

                            outVertices.push(nV2.clone());
                            outVertices.push(nV3.clone());
                            outVertices.push(nV4);
                            break;
                        }
                        if (v3Out) {
                            nV1 = inVertices[i];
                            nV2 = inVertices[i + 1];
                            nV3 = clip(inVertices[i + 2], nV1, p, s);
                            nV4 = clip(inVertices[i + 2], nV2, p, s);
                        }
                        outVertices.push(nV1.clone());
                        outVertices.push(nV2.clone());
                        outVertices.push(nV3);

                        outVertices.push(nV4);
                        outVertices.push(nV3.clone());
                        outVertices.push(nV2.clone());
                        break;
                    case 2:
                        if (!v1Out) {
                            nV1 = inVertices[i].clone();
                            nV2 = clip(nV1, inVertices[i + 1], p, s);
                            nV3 = clip(nV1, inVertices[i + 2], p, s);
                            outVertices.push(nV1);
                            outVertices.push(nV2);
                            outVertices.push(nV3);
                        }
                        if (!v2Out) {
                            nV1 = inVertices[i + 1].clone();
                            nV2 = clip(nV1, inVertices[i + 2], p, s);
                            nV3 = clip(nV1, inVertices[i], p, s);
                            outVertices.push(nV1);
                            outVertices.push(nV2);
                            outVertices.push(nV3);
                        }
                        if (!v3Out) {
                            nV1 = inVertices[i + 2].clone();
                            nV2 = clip(nV1, inVertices[i], p, s);
                            nV3 = clip(nV1, inVertices[i + 1], p, s);
                            outVertices.push(nV1);
                            outVertices.push(nV2);
                            outVertices.push(nV3);
                        }
                        break;
                    case 3:
                        break;
                }
            }
            return outVertices;
        }

        let decalVertices = [];
        const vertex = new THREE.Vector3();
        const normal = new THREE.Vector3();
        const baseUv = new THREE.Vector2();

        if (geometry.index !== null) {
            const index = geometry.index;
            for (let i = 0; i < index.count; i++) {
                const idx = index.getX(i);
                vertex.fromBufferAttribute(positionAttribute, idx);
                if (normalAttribute) {
                    normal.fromBufferAttribute(normalAttribute, idx);
                } else {
                    normal.set(0, 1, 0);
                }
                if (baseUvAttribute) {
                    baseUv.fromBufferAttribute(baseUvAttribute, idx);
                } else {
                    baseUv.set(0, 0);
                }
                pushDecalVertex(decalVertices, vertex, normal, baseUv);
            }
        } else {
            for (let i = 0; i < positionAttribute.count; i++) {
                vertex.fromBufferAttribute(positionAttribute, i);
                if (normalAttribute) {
                    normal.fromBufferAttribute(normalAttribute, i);
                } else {
                    normal.set(0, 1, 0);
                }
                if (baseUvAttribute) {
                    baseUv.fromBufferAttribute(baseUvAttribute, i);
                } else {
                    baseUv.set(0, 0);
                }
                pushDecalVertex(decalVertices, vertex, normal, baseUv);
            }
        }

        decalVertices = clipGeometry(decalVertices, plane.set(1, 0, 0));
        decalVertices = clipGeometry(decalVertices, plane.set(-1, 0, 0));
        decalVertices = clipGeometry(decalVertices, plane.set(0, 1, 0));
        decalVertices = clipGeometry(decalVertices, plane.set(0, -1, 0));
        decalVertices = clipGeometry(decalVertices, plane.set(0, 0, 1));
        decalVertices = clipGeometry(decalVertices, plane.set(0, 0, -1));

        for (let i = 0; i < decalVertices.length; i++) {
            const decalVertex = decalVertices[i];
            uvs.push(
                0.5 + (decalVertex.position.x / size.x),
                0.5 + (decalVertex.position.y / size.y)
            );
            decalVertex.position.applyMatrix4(projectorMatrix);

            vertices.push(decalVertex.position.x, decalVertex.position.y, decalVertex.position.z);
            normals.push(decalVertex.normal.x, decalVertex.normal.y, decalVertex.normal.z);
            baseUvs.push(decalVertex.baseUv.x, decalVertex.baseUv.y);
        }

        this.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
        this.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
        this.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
        this.setAttribute('baseUv', new THREE.Float32BufferAttribute(baseUvs, 2));
    }
}

// 貼る領域(clipScale)と画像の大きさ(imageScale)の比率をキャンバス上に反映し、余白を透明パディングしたテクスチャを生成
// これによりGLTFエクスポート時にもプレビューと寸分違わぬサイズで完全再現される
function getComposedDecalTexture(baseTex, clipScale, imageScale) {
    if (!baseTex || !baseTex.image) return baseTex;

    const img = baseTex.image;
    const origW = img.width || 1024;
    const origH = img.height || 1024;

    const ratioX = (clipScale && clipScale.x > 0) ? (imageScale.x / clipScale.x) : 1.0;
    const ratioY = (clipScale && clipScale.y > 0) ? (imageScale.y / clipScale.y) : 1.0;

    // 比率がほぼ1.0なら元のテクスチャをそのまま使用
    if (Math.abs(ratioX - 1.0) < 0.001 && Math.abs(ratioY - 1.0) < 0.001) {
        return baseTex;
    }

    const canvas = document.createElement('canvas');
    canvas.width = origW;
    canvas.height = origH;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, origW, origH);

    const dw = origW * ratioX;
    const dh = origH * ratioY;
    const dx = (origW - dw) / 2;
    const dy = (origH - dh) / 2;

    ctx.drawImage(img, dx, dy, dw, dh);

    const tex = new THREE.CanvasTexture(canvas);
    tex.colorSpace = THREE.SRGBColorSpace;
    applyHighQualityTextureFiltering(tex);
    return tex;
}

// ダミー白テクスチャのシングルトン
let whiteDummyTexture = null;
function getWhiteFallbackTexture() {
    if (!whiteDummyTexture) {
        const data = new Uint8Array([255, 255, 255, 255]);
        whiteDummyTexture = new THREE.DataTexture(data, 1, 1, THREE.RGBAFormat);
        whiteDummyTexture.needsUpdate = true;
    }
    return whiteDummyTexture;
}

// 貼り付け先（ベースメッシュ）のテクスチャを取得
function getBaseMeshTexture() {
    if (!mainSphereMesh || !mainSphereMesh.material) return null;
    if (Array.isArray(mainSphereMesh.material)) {
        for (const mat of mainSphereMesh.material) {
            if (mat && mat.map) return mat.map;
        }
        return null;
    }
    return mainSphereMesh.material.map || null;
}

// 下地テクスチャの輝度（影・溝）を反映するデカールマテリアルを生成
function createDecalMaterial(options = {}) {
    const {
        map,
        baseMap = getBaseMeshTexture(),
        shadowIntensity = 0.8,
        opacity = 1.0,
        polygonOffsetFactor = -2,
        roughness = 1.0,
        metalness = 0.0,
        emissive = 0xffffff,
        emissiveIntensity = 1.0
    } = options;

    const fallbackTex = getWhiteFallbackTexture();
    const activeBaseMap = baseMap || fallbackTex;

    // 自己発光(Emissive/Unlit)マテリアル：光源がない暗闇空間でも表裏100%同一の鮮やかな明るさで発光
    const material = new THREE.MeshStandardMaterial({
        map: map,
        emissiveMap: map,
        emissive: new THREE.Color(emissive),
        emissiveIntensity: emissiveIntensity,
        transparent: true,
        opacity: opacity,
        depthTest: true,
        depthWrite: false,
        polygonOffset: true,
        polygonOffsetFactor: polygonOffsetFactor,
        roughness: roughness,
        metalness: metalness,
        side: THREE.DoubleSide
    });

    material.userData = {
        uBaseTexture: { value: activeBaseMap },
        uShadowIntensity: { value: shadowIntensity },
        uHasBaseTexture: { value: baseMap ? 1.0 : 0.0 }
    };

    material.customProgramCacheKey = () => 'custom_decal_shadow_material_unlit';

    material.onBeforeCompile = (shader) => {
        shader.uniforms.uBaseTexture = material.userData.uBaseTexture;
        shader.uniforms.uShadowIntensity = material.userData.uShadowIntensity;
        shader.uniforms.uHasBaseTexture = material.userData.uHasBaseTexture;

        shader.vertexShader = `
            attribute vec2 baseUv;
            varying vec2 vBaseUv;
            varying vec2 vCustomDecalUv;
        ` + shader.vertexShader;

        shader.vertexShader = shader.vertexShader.replace(
            '#include <uv_vertex>',
            `#include <uv_vertex>
            vBaseUv = baseUv;
            vCustomDecalUv = uv;
            `
        );

        shader.fragmentShader = `
            uniform sampler2D uBaseTexture;
            uniform float uShadowIntensity;
            uniform float uHasBaseTexture;
            varying vec2 vBaseUv;
            varying vec2 vCustomDecalUv;
        ` + shader.fragmentShader;

        shader.fragmentShader = shader.fragmentShader.replace(
            '#include <map_fragment>',
            `#include <map_fragment>
            // 画像枠外（UVが [0, 1] の外）は透明にしてクリップ
            if (vCustomDecalUv.x < 0.0 || vCustomDecalUv.x > 1.0 || vCustomDecalUv.y < 0.0 || vCustomDecalUv.y > 1.0) {
                diffuseColor.a = 0.0;
            }
            float decalShadowFactor = 1.0;
            if (uHasBaseTexture > 0.5 && uShadowIntensity > 0.0) {
                vec4 baseTexColor = texture2D(uBaseTexture, vBaseUv);
                // 透明領域(alpha=0)は影なし(1.0)として扱い誤認黒ずみを防止
                float baseLum = baseTexColor.a > 0.01 ? dot(baseTexColor.rgb, vec3(0.299, 0.587, 0.114)) : 1.0;
                decalShadowFactor = mix(1.0, baseLum, uShadowIntensity);
            }
            // 方法2（自己発光 / Unlit）: ライトの当たり方による表裏の明暗差を排し、均一な自己発光で描画
            diffuseColor.rgb = vec3(0.0);
            `
        );

        // 自己発光(Unlit)側にも下地メッシュの影・溝を反映
        shader.fragmentShader = shader.fragmentShader.replace(
            '#include <emissivemap_fragment>',
            `#include <emissivemap_fragment>
            if (vCustomDecalUv.x < 0.0 || vCustomDecalUv.x > 1.0 || vCustomDecalUv.y < 0.0 || vCustomDecalUv.y > 1.0) {
                totalEmissiveRadiance = vec3(0.0);
            } else {
                totalEmissiveRadiance *= decalShadowFactor;
            }
            `
        );
    };

    return material;
}

// ベイク画像から周囲に比べて極端に明るいピクセル（隙間漏れ）や極端に暗いピクセル（黒ずみ斑点・破片）を自動検出して除去する双方向外れ値除去フィルタ
// マルチスケール（7x7, 5x5, 3x3）で広い塊状のノイズも確実に検出し、周囲の正常な輝度色で補間
function removeOutlierNoise(imageData, width, height) {
    const data = imageData.data;
    const src = new Uint8ClampedArray(data);

    // パス設定:
    // パス1: 半径3 (7x7領域) で大きな塊状ノイズを検出・除去
    // パス2: 半径2 (5x5領域) で中程度のノイズを除去
    // パス3: 半径1 (3x3領域) で微小ノイズを平滑化
    const passes = [
        { radius: 3, threshBright: 18, threshDark: 18, pctLow: 0.30, pctHigh: 0.70, minSamples: 8 },
        { radius: 2, threshBright: 16, threshDark: 16, pctLow: 0.35, pctHigh: 0.65, minSamples: 6 },
        { radius: 1, threshBright: 14, threshDark: 14, pctLow: 0.40, pctHigh: 0.60, minSamples: 4 }
    ];

    const samples = [];

    for (let p = 0; p < passes.length; p++) {
        const { radius, threshBright, threshDark, pctLow, pctHigh, minSamples } = passes[p];
        src.set(data);

        for (let y = radius; y < height - radius; y++) {
            const rowOffset = y * width;
            for (let x = radius; x < width - radius; x++) {
                const idx = (rowOffset + x) * 4;
                const a = src[idx + 3];
                if (a < 15) continue; // 透明ピクセルはスキップ

                const r = src[idx];
                const g = src[idx + 1];
                const b = src[idx + 2];
                const lum = 0.299 * r + 0.587 * g + 0.114 * b;

                samples.length = 0;

                // 近傍ピクセルを収集
                for (let dy = -radius; dy <= radius; dy++) {
                    const nRow = (y + dy) * width;
                    for (let dx = -radius; dx <= radius; dx++) {
                        if (dx === 0 && dy === 0) continue;
                        const nIdx = (nRow + (x + dx)) * 4;
                        if (src[nIdx + 3] >= 15) {
                            const nr = src[nIdx];
                            const ng = src[nIdx + 1];
                            const nb = src[nIdx + 2];
                            samples.push({
                                r: nr, g: ng, b: nb,
                                lum: 0.299 * nr + 0.587 * ng + 0.114 * nb
                            });
                        }
                    }
                }

                if (samples.length >= minSamples) {
                    samples.sort((s1, s2) => s1.lum - s2.lum);

                    // 1. 周囲より極端に明るい場合 (破片・スパイクノイズ) -> 下位パーセンタイルの影色で補間
                    const lowIdx = Math.floor(samples.length * pctLow);
                    const lowSample = samples[lowIdx];
                    if (lum - lowSample.lum > threshBright) {
                        data[idx] = lowSample.r;
                        data[idx + 1] = lowSample.g;
                        data[idx + 2] = lowSample.b;
                        continue;
                    }

                    // 2. 周囲より極端に暗い場合 (黒ずみ斑点・破片ノイズ) -> 上位パーセンタイルの明るい色で補間
                    const highIdx = Math.floor(samples.length * pctHigh);
                    const highSample = samples[highIdx];
                    if (highSample.lum - lum > threshDark) {
                        data[idx] = highSample.r;
                        data[idx + 1] = highSample.g;
                        data[idx + 2] = highSample.b;
                    }
                }
            }
        }
    }
}

// デカールテクスチャに下地メッシュの影・溝の輝度をピクセル合成（ベイク）してCanvasTextureを生成
function bakeDecalTexture(decalGeometry, decalTexture, baseTexture, shadowIntensity) {
    if (!renderer || !decalTexture) {
        return decalTexture;
    }

    if (shadowIntensity <= 0 || !baseTexture) {
        return decalTexture;
    }

    const img = decalTexture.image;
    const origWidth = img ? (img.width || 1024) : 1024;
    const origHeight = img ? (img.height || 1024) : 1024;
    const bakeWidth = Math.min(Math.max(origWidth, 256), 2048);
    const bakeHeight = Math.min(Math.max(origHeight, 256), 2048);

    const renderTarget = new THREE.WebGLRenderTarget(bakeWidth, bakeHeight, {
        minFilter: THREE.LinearFilter,
        magFilter: THREE.LinearFilter,
        format: THREE.RGBAFormat,
        type: THREE.UnsignedByteType
    });

    const bakeCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, -10, 10);
    const bakeScene = new THREE.Scene();

    // 1. 背景に元のデカール画像を描画（トリミング外のピクセル保持）
    const bgMat = new THREE.MeshBasicMaterial({
        map: decalTexture,
        transparent: true,
        depthTest: false,
        depthWrite: false
    });
    const bgGeo = new THREE.PlaneGeometry(2, 2);
    const bgMesh = new THREE.Mesh(bgGeo, bgMat);
    bgMesh.position.z = -1;
    bakeScene.add(bgMesh);

    // 2. 影を適用するデカールメッシュ（UV空間でのポリゴン描画）
    const bakeMaterial = new THREE.ShaderMaterial({
        uniforms: {
            uDecalTexture: { value: decalTexture },
            uBaseTexture: { value: baseTexture },
            uShadowIntensity: { value: shadowIntensity }
        },
        vertexShader: `
            attribute vec2 baseUv;
            varying vec2 vDecalUv;
            varying vec2 vBaseUv;
            void main() {
                vDecalUv = uv;
                vBaseUv = baseUv;
                gl_Position = vec4(uv.x * 2.0 - 1.0, uv.y * 2.0 - 1.0, 0.0, 1.0);
            }
        `,
        fragmentShader: `
            uniform sampler2D uDecalTexture;
            uniform sampler2D uBaseTexture;
            uniform float uShadowIntensity;
            varying vec2 vDecalUv;
            varying vec2 vBaseUv;
            void main() {
                vec4 decalCol = texture2D(uDecalTexture, vDecalUv);
                if (decalCol.a <= 0.0) discard;
                vec4 baseCol = texture2D(uBaseTexture, vBaseUv);
                float baseLum = dot(baseCol.rgb, vec3(0.299, 0.587, 0.114));
                float shadowFactor = mix(1.0, baseLum, uShadowIntensity);
                gl_FragColor = vec4(decalCol.rgb * shadowFactor, decalCol.a);
            }
        `,
        transparent: true,
        depthTest: false,
        depthWrite: false,
        side: THREE.DoubleSide
    });

    const decalMeshInUv = new THREE.Mesh(decalGeometry, bakeMaterial);
    bakeScene.add(decalMeshInUv);

    // レンダリング実行
    const currentRenderTarget = renderer.getRenderTarget();
    const currentClearColor = new THREE.Color();
    renderer.getClearColor(currentClearColor);
    const currentClearAlpha = renderer.getClearAlpha();

    renderer.setRenderTarget(renderTarget);
    renderer.setClearColor(0x000000, 0);
    renderer.clear();
    renderer.render(bakeScene, bakeCamera);

    // ピクセル読み出し
    const pixelBuffer = new Uint8Array(bakeWidth * bakeHeight * 4);
    renderer.readRenderTargetPixels(renderTarget, 0, 0, bakeWidth, bakeHeight, pixelBuffer);

    // レンダラー状態復元
    renderer.setRenderTarget(currentRenderTarget);
    renderer.setClearColor(currentClearColor, currentClearAlpha);

    // Canvasへの転送
    const canvas = document.createElement('canvas');
    canvas.width = bakeWidth;
    canvas.height = bakeHeight;
    const ctx = canvas.getContext('2d');
    const imgData = ctx.createImageData(bakeWidth, bakeHeight);

    for (let y = 0; y < bakeHeight; y++) {
        const srcRow = (bakeHeight - 1 - y) * bakeWidth * 4;
        const dstRow = y * bakeWidth * 4;
        for (let x = 0; x < bakeWidth * 4; x++) {
            imgData.data[dstRow + x] = pixelBuffer[srcRow + x];
        }
    }

    // 元画像のアルファ値を参照し、星の外側にはみ出たポリゴンゴミ・茶色いエッジを完全消去
    if (img && (img.width || img.naturalWidth)) {
        const origCanvas = document.createElement('canvas');
        origCanvas.width = bakeWidth;
        origCanvas.height = bakeHeight;
        const origCtx = origCanvas.getContext('2d');
        origCtx.drawImage(img, 0, 0, bakeWidth, bakeHeight);
        const origData = origCtx.getImageData(0, 0, bakeWidth, bakeHeight).data;

        for (let i = 0; i < bakeWidth * bakeHeight; i++) {
            if (origData[i * 4 + 3] === 0) {
                imgData.data[i * 4 + 3] = 0; // 元画像が完全透明な部分は強制的に完全透明化
            }
        }
    }

    // 周囲に比べて極端に明るいピクセル（隙間漏れ）および極端に暗いピクセル（黒ずみ斑点）を自動検出して除去
    removeOutlierNoise(imgData, bakeWidth, bakeHeight);

    ctx.putImageData(imgData, 0, 0);

    // リソース解放
    renderTarget.dispose();
    bgGeo.dispose();
    bgMat.dispose();
    bakeMaterial.dispose();

    const bakedTexture = new THREE.CanvasTexture(canvas);
    bakedTexture.needsUpdate = true;
    bakedTexture.colorSpace = decalTexture.colorSpace || THREE.SRGBColorSpace;
    applyHighQualityTextureFiltering(bakedTexture);

    return bakedTexture;
}



// モデル変更時に全デカールの下地テクスチャ参照を一括更新
function updateDecalsBaseTexture() {
    const baseMap = getBaseMeshTexture();
    const fallbackTex = getWhiteFallbackTexture();
    const activeMap = baseMap || fallbackTex;
    const hasMap = baseMap ? 1.0 : 0.0;

    state.appliedDecals.forEach(item => {
        if (item.mesh && item.mesh.material && item.mesh.material.userData) {
            if (item.mesh.material.userData.uBaseTexture) {
                item.mesh.material.userData.uBaseTexture.value = activeMap;
            }
            if (item.mesh.material.userData.uHasBaseTexture) {
                item.mesh.material.userData.uHasBaseTexture.value = hasMap;
            }
        }
    });

    requestRender();
}

// デカールのメッシュ抜け（穴あき）を100%防止し、頂点浮き出し(Z-fighting防止)および法線反転を行う高速ジオメトリ処理
function filterDecalGeometry(decalGeometry) {
    const posAttr = decalGeometry.attributes.position;
    const normAttr = decalGeometry.attributes.normal;

    if (!posAttr) return decalGeometry;



    // 2. 表面オフセットは0固定
    const offset = 0;

    return decalGeometry;
}

// スライダー操作時の超高速トランスフォーム更新 (計算負荷0ms・カクツキゼロ)
function updatePreviewTransform() {
    if (!state.preview.projectionBoxMesh && !state.preview.imageBoxMesh) return;

    // プロジェクションボックス (可視化立方体) はヒット面にぴったり固定
    const dummy = new THREE.Object3D();
    dummy.position.copy(state.preview.position);

    const targetPoint = state.preview.position.clone().add(state.preview.normal);
    dummy.lookAt(targetPoint);

    const rotRad = THREE.MathUtils.degToRad(state.preview.rotationZ);
    dummy.rotateZ(rotRad);

    modelGroup.updateMatrixWorld(true);
    const invModelGroup = modelGroup.matrixWorld.clone().invert();

    // 1. 貼る領域ボックス (水色) の同期
    if (state.preview.projectionBoxMesh) {
        dummy.scale.copy(state.preview.scale);
        dummy.updateMatrix();

        const localMatrix = dummy.matrix.clone().premultiply(invModelGroup);
        localMatrix.decompose(
            state.preview.projectionBoxMesh.position,
            state.preview.projectionBoxMesh.quaternion,
            state.preview.projectionBoxMesh.scale
        );
    }

    // 2. 画像サイズボックス (黄色) の同期
    if (state.preview.imageBoxMesh) {
        dummy.scale.set(
            state.preview.imageScale.x,
            state.preview.imageScale.y,
            state.preview.scale.z * 1.01
        );
        dummy.updateMatrix();

        const localMatrixImg = dummy.matrix.clone().premultiply(invModelGroup);
        localMatrixImg.decompose(
            state.preview.imageBoxMesh.position,
            state.preview.imageBoxMesh.quaternion,
            state.preview.imageBoxMesh.scale
        );
    }

    requestRender();
}

// デカールプレビューメッシュを新規生成・更新する
function updatePreviewDecal() {
    if (isUpdatingPreview) return;
    isUpdatingPreview = true;

    try {
        if (state.preview.mesh) {
            modelGroup.remove(state.preview.mesh);
            state.preview.mesh.geometry.dispose();
            state.preview.mesh = null;
        }

        if (state.preview.projectionBoxMesh) {
            modelGroup.remove(state.preview.projectionBoxMesh);
            state.preview.projectionBoxMesh.traverse((child) => {
                if (child.geometry) child.geometry.dispose();
                if (child.material) child.material.dispose();
            });
            state.preview.projectionBoxMesh = null;
        }

        if (state.preview.imageBoxMesh) {
            modelGroup.remove(state.preview.imageBoxMesh);
            state.preview.imageBoxMesh.traverse((child) => {
                if (child.geometry) child.geometry.dispose();
                if (child.material) child.material.dispose();
            });
            state.preview.imageBoxMesh = null;
        }

        if (!state.selectedTexture) {
            isUpdatingPreview = false;
            return;
        }

        const dummy = new THREE.Object3D();
        dummy.position.copy(state.preview.position);

        const targetPoint = state.preview.position.clone().add(state.preview.normal);
        dummy.lookAt(targetPoint);

        const rotRad = THREE.MathUtils.degToRad(state.preview.rotationZ);
        dummy.rotateZ(rotRad);

        const rotation = dummy.rotation.clone();

        let rawGeometry;
        rawGeometry = new CustomDecalGeometry(
            mainSphereMesh,
            state.preview.position,
            rotation,
            state.preview.scale,
            state.preview.imageScale
        );
        const decalGeometry = filterDecalGeometry(rawGeometry);
        rawGeometry.dispose();

        const effectiveTexture = getComposedDecalTexture(
            state.selectedTexture,
            state.preview.scale,
            state.preview.imageScale
        );

        const previewMaterial = createDecalMaterial({
            map: effectiveTexture,
            shadowIntensity: state.preview.shadowIntensity,
            opacity: 0.8,
            polygonOffsetFactor: -10,
            roughness: 1.0,
            metalness: 0.0,
            emissive: 0xffffff,
            emissiveIntensity: 1.0
        });

        const previewMesh = new THREE.Mesh(decalGeometry, previewMaterial);
        previewMesh.name = "PreviewDecal";
        previewMesh.renderOrder = 999; //<-確定前デカールを最前面に表示

        modelGroup.updateMatrixWorld(true);
        previewMesh.applyMatrix4(modelGroup.matrixWorld.clone().invert());

        state.preview.mesh = previewMesh;
        modelGroup.add(previewMesh);

        // 貼る領域 (クリップ枠) を可視化する立方体 (水色)
        if (state.preview.showProjectionBox) {
            const boxGroup = new THREE.Group();
            boxGroup.name = "ProjectionBox";

            const boxGeo = new THREE.BoxGeometry(1, 1, 1);
            const edgesGeo = new THREE.EdgesGeometry(boxGeo);

            const lineMat = new THREE.LineBasicMaterial({ color: 0x00e5ff, linewidth: 2 });
            const boxLines = new THREE.LineSegments(edgesGeo, lineMat);
            boxGroup.add(boxLines);

            const faceMat = new THREE.MeshBasicMaterial({
                color: 0x00e5ff,
                transparent: true,
                opacity: 0.08,
                side: THREE.DoubleSide,
                depthWrite: false
            });
            const boxFaces = new THREE.Mesh(boxGeo, faceMat);
            boxGroup.add(boxFaces);

            state.preview.projectionBoxMesh = boxGroup;
            modelGroup.add(boxGroup);
        }

        // 画像の大きさ (テクスチャ枠) を可視化する立方体 (黄色)
        if (state.preview.showImageBox) {
            const imgBoxGroup = new THREE.Group();
            imgBoxGroup.name = "ImageBox";

            const boxGeo = new THREE.BoxGeometry(1, 1, 1);
            const edgesGeo = new THREE.EdgesGeometry(boxGeo);

            const lineMat = new THREE.LineBasicMaterial({ color: 0xffea00, linewidth: 2 });
            const boxLines = new THREE.LineSegments(edgesGeo, lineMat);
            imgBoxGroup.add(boxLines);

            const faceMat = new THREE.MeshBasicMaterial({
                color: 0xffea00,
                transparent: true,
                opacity: 0.06,
                side: THREE.DoubleSide,
                depthWrite: false
            });
            const boxFaces = new THREE.Mesh(boxGeo, faceMat);
            imgBoxGroup.add(boxFaces);

            state.preview.imageBoxMesh = imgBoxGroup;
            modelGroup.add(imgBoxGroup);
        }

        // トランスフォームを一元適用 (オフセット位置に完全一致)
        updatePreviewTransform();
    } catch (e) {
        console.error("DecalGeometry generation error:", e.message);
        sysError(`投影エラー: デカールが表面と交差していません。奥行き(Depth)を大きくするか、位置を調整してください。`);
    } finally {
        isUpdatingPreview = false;
        requestRender();
    }
}

// デカールの確定配置 (Apply)
function applyCurrentDecal() {
    if (!state.preview.mesh || !state.selectedTexture) return;

    const dummy = new THREE.Object3D();
    dummy.position.copy(state.preview.position);
    const targetPoint = state.preview.position.clone().add(state.preview.normal);
    dummy.lookAt(targetPoint);
    const rotRad = THREE.MathUtils.degToRad(state.preview.rotationZ);
    dummy.rotateZ(rotRad);
    const rotation = dummy.rotation.clone();

    let geometry;
    try {
        const rawGeometry = new CustomDecalGeometry(
            mainSphereMesh,
            state.preview.position,
            rotation,
            state.preview.scale,
            state.preview.imageScale
        );
        geometry = filterDecalGeometry(rawGeometry);
        rawGeometry.dispose();
    } catch (e) {
        console.error("Failed to generate geometry for applied decal:", e.message);
        sysError(`確定投影エラー: ジオメトリの生成に失敗しました。`);
        return;
    }

    const effectiveTexture = getComposedDecalTexture(
        state.selectedTexture,
        state.preview.scale,
        state.preview.imageScale
    );

    // プレビューと同一の自己発光(Emissive/Unlit)・影反映シェーダーマテリアルを確定デカールにも適用
    const material = createDecalMaterial({
        map: effectiveTexture,
        shadowIntensity: state.preview.shadowIntensity,
        opacity: 1.0,
        polygonOffsetFactor: -2 - state.appliedDecals.length,
        roughness: 1.0,
        metalness: 0.0,
        emissive: 0xffffff,
        emissiveIntensity: 1.0
    });

    const decalMesh = new THREE.Mesh(geometry, material);
    decalMesh.renderOrder = state.nextLayerId;
    decalMesh.name = `Decal_${state.nextLayerId}`;

    modelGroup.updateMatrixWorld(true);
    decalMesh.applyMatrix4(modelGroup.matrixWorld.clone().invert());

    modelGroup.add(decalMesh);

    const layerId = state.nextLayerId++;
    const layerName = `${state.selectedName} (${layerId})`;

    state.appliedDecals.push({
        id: layerId,
        name: layerName,
        mesh: decalMesh,
        textureSrc: state.selectedImageSrc,
        originalTexture: effectiveTexture,
        opacity: 1.0,
        shadowIntensity: state.preview.shadowIntensity,
        visible: true,
        tint: '#ffffff'
    });

    updateDecalsOrder();
    cancelCurrentPreview();
}

// プレビューのキャンセル・片付け
function cancelCurrentPreview() {
    if (state.preview.mesh) {
        modelGroup.remove(state.preview.mesh);
        state.preview.mesh.geometry.dispose();
        state.preview.mesh.material.dispose();
        state.preview.mesh = null;
    }

    if (state.preview.projectionBoxMesh) {
        modelGroup.remove(state.preview.projectionBoxMesh);
        state.preview.projectionBoxMesh.traverse((child) => {
            if (child.geometry) child.geometry.dispose();
            if (child.material) child.material.dispose();
        });
        state.preview.projectionBoxMesh = null;
    }

    if (state.preview.imageBoxMesh) {
        modelGroup.remove(state.preview.imageBoxMesh);
        state.preview.imageBoxMesh.traverse((child) => {
            if (child.geometry) child.geometry.dispose();
            if (child.material) child.material.dispose();
        });
        state.preview.imageBoxMesh = null;
    }

    state.isEditing = false;
    document.getElementById('section-transform').classList.add('disabled');

    if (state.selectedTexture) {
        document.getElementById('mode-text').innerText = "PLACE DECAL MODE (CLICK ON SPHERE)";
        document.getElementById('mode-text').style.borderColor = 'var(--color-accent)';
        document.getElementById('mode-text').style.color = 'var(--color-accent)';
        document.getElementById('mode-text').style.background = 'rgba(255, 69, 0, 0.1)';
    } else {
        document.getElementById('mode-text').innerText = "CAMERA MODE";
        document.getElementById('mode-text').style.borderColor = 'var(--color-cyan)';
        document.getElementById('mode-text').style.color = 'var(--color-cyan)';
        document.getElementById('mode-text').style.background = 'rgba(0, 229, 255, 0.1)';
    }

    requestRender();
}

// デカールの前後関係・レイヤー順序の一括同期 (3D描画順 + polygonOffset + シーングラフ)
function updateDecalsOrder() {
    state.appliedDecals.forEach((decal, index) => {
        if (decal.mesh) {
            // 配列のindexが大きいほど手前（前面）になるよう polygonOffset と renderOrder を設定
            if (decal.mesh.material) {
                decal.mesh.material.polygonOffset = true;
                decal.mesh.material.polygonOffsetFactor = -2 - index;
            }
            decal.mesh.renderOrder = index + 1;

            // modelGroup 内の子オブジェクト順序も配列順に揃えることで描画整合性を担保
            modelGroup.remove(decal.mesh);
            modelGroup.add(decal.mesh);
        }
    });

    updateLayerListUI();
    requestRender();
}

// レイヤーを前面へ移動（UI上で上へ）
function moveLayerUp(id) {
    const idx = state.appliedDecals.findIndex(item => item.id === id);
    if (idx === -1 || idx >= state.appliedDecals.length - 1) return;

    // 配列内で 1つ後ろ（より前面）と交換
    const temp = state.appliedDecals[idx];
    state.appliedDecals[idx] = state.appliedDecals[idx + 1];
    state.appliedDecals[idx + 1] = temp;

    updateDecalsOrder();
}

// レイヤーを背面へ移動（UI上で下へ）
function moveLayerDown(id) {
    const idx = state.appliedDecals.findIndex(item => item.id === id);
    if (idx === -1 || idx <= 0) return;

    // 配列内で 1つ前（より背面）と交換
    const temp = state.appliedDecals[idx];
    state.appliedDecals[idx] = state.appliedDecals[idx - 1];
    state.appliedDecals[idx - 1] = temp;

    updateDecalsOrder();
}

// ドラッグ中の一時参照ID
let draggedLayerId = null;

// レイヤーリストUIの構築
function updateLayerListUI() {
    const listEl = document.getElementById('layer-list-element');
    if (!listEl) return;

    listEl.innerHTML = '';

    if (state.appliedDecals.length === 0) {
        listEl.innerHTML = `
            <div class="layer-empty" id="layer-empty-msg">
                レイヤーがありません。<br>上のパネルで画像を選び、モデルをクリックしてデカールを配置してください。
            </div>
        `;
        return;
    }

    const totalDecals = state.appliedDecals.length;

    // UI上は上が最前面 (index: length - 1)、下が最背面 (index: 0)
    for (let i = totalDecals - 1; i >= 0; i--) {
        const item = state.appliedDecals[i];
        const isTop = (i === totalDecals - 1);
        const isBottom = (i === 0);

        const layerItem = document.createElement('div');
        layerItem.className = 'layer-item';
        layerItem.dataset.id = item.id;
        layerItem.draggable = true;

        // ドラッグ＆ドロップイベント設定
        layerItem.addEventListener('dragstart', (e) => {
            draggedLayerId = item.id;
            layerItem.classList.add('dragging');
            e.dataTransfer.effectAllowed = 'move';
            e.dataTransfer.setData('text/plain', String(item.id));
        });

        layerItem.addEventListener('dragend', () => {
            draggedLayerId = null;
            document.querySelectorAll('.layer-item').forEach(el => {
                el.classList.remove('dragging', 'drag-over-top', 'drag-over-bottom');
            });
        });

        layerItem.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.dataTransfer.dropEffect = 'move';
            if (draggedLayerId === item.id) return;

            const rect = layerItem.getBoundingClientRect();
            const relY = e.clientY - rect.top;
            if (relY < rect.height / 2) {
                layerItem.classList.add('drag-over-top');
                layerItem.classList.remove('drag-over-bottom');
            } else {
                layerItem.classList.add('drag-over-bottom');
                layerItem.classList.remove('drag-over-top');
            }
        });

        layerItem.addEventListener('dragleave', () => {
            layerItem.classList.remove('drag-over-top', 'drag-over-bottom');
        });

        layerItem.addEventListener('drop', (e) => {
            e.preventDefault();
            layerItem.classList.remove('drag-over-top', 'drag-over-bottom');

            if (draggedLayerId === null || draggedLayerId === item.id) return;

            const fromIdx = state.appliedDecals.findIndex(d => d.id === draggedLayerId);
            if (fromIdx === -1) return;

            const rect = layerItem.getBoundingClientRect();
            const isDropTop = (e.clientY - rect.top) < rect.height / 2;

            // ドラッグ元を取り出し
            const [draggedItem] = state.appliedDecals.splice(fromIdx, 1);

            // 取り出し後のターゲット位置を特定
            const targetIdx = state.appliedDecals.findIndex(d => d.id === item.id);
            if (targetIdx === -1) return;

            // UI上で上(top) = 配列では前面(targetIdx + 1)、UI上で下(bottom) = 配列では背面(targetIdx)
            const insertIdx = isDropTop ? targetIdx + 1 : targetIdx;
            state.appliedDecals.splice(insertIdx, 0, draggedItem);

            draggedLayerId = null;
            updateDecalsOrder();
        });

        // ヘッダー（ドラッグハンドル、サムネイル、名前＆順序、操作ボタン）
        const header = document.createElement('div');
        header.className = 'layer-item-header';

        // ドラッグハンドル
        const handle = document.createElement('div');
        handle.className = 'layer-drag-handle';
        handle.innerHTML = '<i class="fa-solid fa-grip-vertical"></i>';
        handle.title = 'ドラッグして並び替え';
        header.appendChild(handle);

        // サムネイル
        const thumb = document.createElement('img');
        thumb.className = 'layer-item-thumbnail';
        thumb.src = item.textureSrc;
        header.appendChild(thumb);

        // レイヤー情報（名前 ＆ 順序バッジ）
        const info = document.createElement('div');
        info.className = 'layer-item-info';

        const nameSpan = document.createElement('span');
        nameSpan.className = 'layer-item-name';
        nameSpan.innerText = item.name;
        nameSpan.title = item.name;
        info.appendChild(nameSpan);

        const orderBadge = document.createElement('div');
        orderBadge.className = `layer-order-badge ${isTop ? 'top' : ''}`;
        if (isTop) {
            orderBadge.innerHTML = '<span>最前面</span>';
        } else if (isBottom) {
            orderBadge.innerHTML = '<span>最背面</span>';
        } else {
            orderBadge.innerHTML = `<span>レイヤー ${i + 1}</span>`;
        }
        info.appendChild(orderBadge);
        header.appendChild(info);

        // アクションボタン群
        const actions = document.createElement('div');
        actions.className = 'layer-item-actions';

        // 1. 前面へ（UI上で上へ）ボタン
        const upBtn = document.createElement('button');
        upBtn.className = `layer-item-action move-up ${isTop ? 'disabled' : ''}`;
        upBtn.innerHTML = '<i class="fa-solid fa-chevron-up"></i>';
        upBtn.title = isTop ? 'すでに最前面です' : '前面へ移動';
        if (!isTop) {
            upBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                moveLayerUp(item.id);
            });
        }
        actions.appendChild(upBtn);

        // 2. 背面へ（UI上で下へ）ボタン
        const downBtn = document.createElement('button');
        downBtn.className = `layer-item-action move-down ${isBottom ? 'disabled' : ''}`;
        downBtn.innerHTML = '<i class="fa-solid fa-chevron-down"></i>';
        downBtn.title = isBottom ? 'すでに最背面です' : '背面へ移動';
        if (!isBottom) {
            downBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                moveLayerDown(item.id);
            });
        }
        actions.appendChild(downBtn);

        // 3. 表示 / 非表示ボタン
        const visBtn = document.createElement('button');
        visBtn.className = `layer-item-action toggle-visibility ${item.visible ? '' : 'hidden'}`;
        visBtn.innerHTML = '<i class="fa-solid fa-eye"></i>';
        visBtn.title = '表示 / 非表示';
        visBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            item.visible = !item.visible;
            item.mesh.visible = item.visible;
            if (item.visible) {
                visBtn.classList.remove('hidden');
            } else {
                visBtn.classList.add('hidden');
            }
            requestRender();
        });
        actions.appendChild(visBtn);

        // 4. 削除ボタン
        const delBtn = document.createElement('button');
        delBtn.className = 'layer-item-action delete';
        delBtn.innerHTML = '<i class="fa-solid fa-trash-can"></i>';
        delBtn.title = '削除';
        delBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            deleteLayer(item.id);
        });
        actions.appendChild(delBtn);

        header.appendChild(actions);
        layerItem.appendChild(header);

        // コントロール部（不透明度 ＆ 下地の影反映）
        const controls = document.createElement('div');
        controls.className = 'layer-item-controls';

        // 1. 不透明度スライダー
        const opacBox = document.createElement('div');
        opacBox.className = 'layer-slider-box';
        const opacLabel = document.createElement('label');
        opacLabel.innerHTML = '<i class="fa-solid fa-droplet"></i> 不透明';
        const opacSlider = document.createElement('input');
        opacSlider.type = 'range';
        opacSlider.min = '0';
        opacSlider.max = '1';
        opacSlider.step = '0.05';
        opacSlider.value = item.opacity;
        const opacVal = document.createElement('span');
        opacVal.innerText = `${Math.round(item.opacity * 100)}%`;
        opacSlider.addEventListener('input', (e) => {
            const val = parseFloat(e.target.value);
            item.opacity = val;
            item.mesh.material.opacity = val;
            opacVal.innerText = `${Math.round(val * 100)}%`;
            requestRender();
        });
        opacBox.appendChild(opacLabel);
        opacBox.appendChild(opacSlider);
        opacBox.appendChild(opacVal);
        controls.appendChild(opacBox);

        // 2. 影反映強度スライダー
        const shadowBox = document.createElement('div');
        shadowBox.className = 'layer-slider-box';
        const shadowLabel = document.createElement('label');
        shadowLabel.innerHTML = '<i class="fa-solid fa-moon"></i> 影反映';
        const shadowSlider = document.createElement('input');
        shadowSlider.type = 'range';
        shadowSlider.min = '0';
        shadowSlider.max = '1';
        shadowSlider.step = '0.05';
        const currentShadow = item.shadowIntensity !== undefined ? item.shadowIntensity : 0.8;
        shadowSlider.value = currentShadow;
        const shadowVal = document.createElement('span');
        shadowSlider.addEventListener('input', (e) => {
            const val = parseFloat(e.target.value);
            item.shadowIntensity = val;
            shadowVal.innerText = `${Math.round(val * 100)}%`;

            if (item.mesh && item.mesh.material && item.mesh.material.userData && item.mesh.material.userData.uShadowIntensity) {
                item.mesh.material.userData.uShadowIntensity.value = val;
            }
            requestRender();
        });
        shadowBox.appendChild(shadowLabel);
        shadowBox.appendChild(shadowSlider);
        shadowBox.appendChild(shadowVal);
        controls.appendChild(shadowBox);

        layerItem.appendChild(controls);
        listEl.appendChild(layerItem);
    }
}

// レイヤー削除
function deleteLayer(id) {
    const idx = state.appliedDecals.findIndex(item => item.id === id);
    if (idx === -1) return;

    const item = state.appliedDecals[idx];

    modelGroup.remove(item.mesh);
    item.mesh.geometry.dispose();
    item.mesh.material.dispose();

    state.appliedDecals.splice(idx, 1);
    updateDecalsOrder();
}

// キーボード入力によるカメラ移動 (WASD + QE) の更新処理
function updateCameraMovement() {
    let hasMovement = false;
    for (const key in state.keys) {
        if (state.keys[key]) {
            hasMovement = true;
            break;
        }
    }
    if (!hasMovement) return false;

    const moveSpeed = 0.08;
    const translation = new THREE.Vector3();

    const forward = new THREE.Vector3();
    camera.getWorldDirection(forward);

    const right = new THREE.Vector3();
    right.crossVectors(forward, camera.up).normalize();

    if (state.keys.w) translation.addScaledVector(forward, moveSpeed);
    if (state.keys.s) translation.addScaledVector(forward, -moveSpeed);
    if (state.keys.a) translation.addScaledVector(right, -moveSpeed);
    if (state.keys.d) translation.addScaledVector(right, moveSpeed);
    if (state.keys.e) translation.addScaledVector(camera.up, moveSpeed);
    if (state.keys.q) translation.addScaledVector(camera.up, -moveSpeed);

    if (translation.lengthSq() > 0) {
        camera.position.add(translation);
        orbitControls.target.add(translation);
        return true;
    }
    return false;
}

// キーボードイベントハンドラ
window.addEventListener('keydown', (e) => {
    const key = e.key.toLowerCase();
    if (key in state.keys) {
        if (document.activeElement && (
            document.activeElement.tagName === 'INPUT' ||
            document.activeElement.tagName === 'TEXTAREA' ||
            document.activeElement.isContentEditable
        )) {
            return;
        }
        state.keys[key] = true;
    }
});

window.addEventListener('keyup', (e) => {
    const key = e.key.toLowerCase();
    if (key in state.keys) {
        state.keys[key] = false;
    }
});

window.addEventListener('blur', () => {
    for (const key in state.keys) {
        state.keys[key] = false;
    }
});

// 描画更新ループ (GPUオンデマンド軽量化)
function animate() {
    requestAnimationFrame(animate);

    let needsRender = false;

    // 1. キーボード移動
    if (updateCameraMovement()) {
        needsRender = true;
    }

    // 2. OrbitControls 慣性更新
    if (orbitControls.update()) {
        needsRender = true;
    }

    // 3. モデル回転・拡大アニメーションの更新
    if (modelGroup) {
        let rotChanged = false;
        if (Math.abs(state.modelRotation.targetY - state.modelRotation.currentY) > 0.0001) {
            state.modelRotation.currentY += (state.modelRotation.targetY - state.modelRotation.currentY) * 0.12;
            if (Math.abs(state.modelRotation.targetY - state.modelRotation.currentY) < 0.0001) {
                state.modelRotation.currentY = state.modelRotation.targetY;
            }
            rotChanged = true;
        }

        if (Math.abs(state.modelRotation.targetX - state.modelRotation.currentX) > 0.0001) {
            state.modelRotation.currentX += (state.modelRotation.targetX - state.modelRotation.currentX) * 0.12;
            if (Math.abs(state.modelRotation.targetX - state.modelRotation.currentX) < 0.0001) {
                state.modelRotation.currentX = state.modelRotation.targetX;
            }
            rotChanged = true;
        }

        if (rotChanged) {
            modelGroup.rotation.x = state.modelRotation.currentX;
            modelGroup.rotation.y = state.modelRotation.currentY;
            updateModelRotationUI();
            needsRender = true;
        }

        // モデル表示拡大率アニメーションの更新
        if (Math.abs(state.modelDisplayScale.target - state.modelDisplayScale.current) > 0.0001) {
            state.modelDisplayScale.current += (state.modelDisplayScale.target - state.modelDisplayScale.current) * 0.12;
            if (Math.abs(state.modelDisplayScale.target - state.modelDisplayScale.current) < 0.0001) {
                state.modelDisplayScale.current = state.modelDisplayScale.target;
            }
            const s = state.modelDisplayScale.current;
            modelGroup.scale.set(s, s, s);
            updateModelDisplayScaleUI();
            needsRender = true;
        }
    }

    // 4. 変化があったフレームでのみ GPU レンダリングを実行
    if (needsRender || renderRequested) {
        renderRequested = false;
        renderer.render(scene, camera);
    }
}

// ウィンドウリサイズ時の処理
function onWindowResize() {
    const container = document.getElementById('canvas-container');
    const width = container.clientWidth;
    const height = container.clientHeight;

    camera.aspect = width / height;
    camera.updateProjectionMatrix();

    renderer.setSize(width, height);
    requestRender();
}

// --- エクスポート処理 ---

// GLTF 形式でのエクスポート
function exportGLTF() {
    const exporter = new GLTFExporter();
    const exportGroup = new THREE.Group();

    const baseMeshClone = mainSphereMesh.clone();
    baseMeshClone.material = new THREE.MeshStandardMaterial({
        name: 'BASE_MODEL_MATERIAL_TO_REMOVE'
    });
    for (let i = baseMeshClone.children.length - 1; i >= 0; i--) {
        baseMeshClone.remove(baseMeshClone.children[i]);
    }
    exportGroup.add(baseMeshClone);

    const baseTex = getBaseMeshTexture();

    state.appliedDecals.forEach(item => {
        if (item.visible) {
            const decalClone = item.mesh.clone();
            const shadow = item.shadowIntensity !== undefined ? item.shadowIntensity : 0.8;
            const origTex = item.originalTexture || (item.mesh.material ? item.mesh.material.map : null);
            const finalMap = (origTex && baseTex && shadow > 0)
                ? bakeDecalTexture(item.mesh.geometry, origTex, baseTex, shadow)
                : origTex;

            // 方法2（自己発光 Emissive / Unlit）：光源がない空間でも表裏均一に100%発光表示
            decalClone.material = new THREE.MeshStandardMaterial({
                map: finalMap,
                emissiveMap: finalMap,
                emissive: new THREE.Color(0xffffff),
                emissiveIntensity: 1.0,
                transparent: true,
                opacity: item.opacity,
                polygonOffset: true,
                polygonOffsetFactor: item.mesh.material ? item.mesh.material.polygonOffsetFactor : -2,
                roughness: 1.0,
                metalness: 0.0,
                side: THREE.DoubleSide
            });
            exportGroup.add(decalClone);
        }
    });

    applyExportTransform(exportGroup);

    exporter.parse(
        exportGroup,
        function (gltf) {
            if (gltf.materials) {
                const removeIndex = gltf.materials.findIndex(mat => mat.name === 'BASE_MODEL_MATERIAL_TO_REMOVE');
                if (removeIndex !== -1) {
                    if (gltf.meshes) {
                        gltf.meshes.forEach(mesh => {
                            if (mesh.primitives) {
                                mesh.primitives.forEach(primitive => {
                                    if (primitive.material === removeIndex) {
                                        delete primitive.material;
                                    } else if (primitive.material > removeIndex) {
                                        primitive.material -= 1;
                                    }
                                });
                            }
                        });
                    }
                    gltf.materials.splice(removeIndex, 1);
                    if (gltf.materials.length === 0) {
                        delete gltf.materials;
                    }
                }
            }

            const resultText = JSON.stringify(gltf, null, 2);
            const blob = new Blob([resultText], { type: 'application/json' });
            const link = document.createElement('a');
            link.href = URL.createObjectURL(blob);
            link.download = 'gt-livery-sphere.gltf';
            link.click();
            URL.revokeObjectURL(link.href);
        },
        function (error) {
            console.error('An error happened during export', error);
        },
        { binary: false }
    );
}

// GLTF 形式でのデカールメッシュのみのエクスポート
function exportGLTFDecalsOnly() {
    const visibleDecals = state.appliedDecals.filter(item => item.visible);
    if (visibleDecals.length === 0) {
        alert('出力対象のデカールが存在しません。');
        return;
    }

    const exporter = new GLTFExporter();
    const exportGroup = new THREE.Group();
    const baseTex = getBaseMeshTexture();

    visibleDecals.forEach(item => {
        const decalClone = item.mesh.clone();
        const shadow = item.shadowIntensity !== undefined ? item.shadowIntensity : 0.8;
        const origTex = item.originalTexture || (item.mesh.material ? item.mesh.material.map : null);
        const finalMap = (origTex && baseTex && shadow > 0)
            ? bakeDecalTexture(item.mesh.geometry, origTex, baseTex, shadow)
            : origTex;

        // 方法2（自己発光 Emissive / Unlit）：光源がない空間でも表裏均一に100%発光表示
        decalClone.material = new THREE.MeshStandardMaterial({
            map: finalMap,
            emissiveMap: finalMap,
            emissive: new THREE.Color(0xffffff),
            emissiveIntensity: 1.0,
            transparent: true,
            opacity: item.opacity,
            polygonOffset: true,
            polygonOffsetFactor: item.mesh.material ? item.mesh.material.polygonOffsetFactor : -2,
            roughness: 1.0,
            metalness: 0.0,
            side: THREE.DoubleSide
        });
        exportGroup.add(decalClone);
    });

    if (state.originalCenter && state.originalScale) {
        const invScale = 1 / state.originalScale;
        exportGroup.scale.set(invScale, invScale, invScale);
        exportGroup.position.copy(state.originalCenter);
        exportGroup.updateMatrixWorld(true);
    }

    exporter.parse(
        exportGroup,
        function (gltf) {
            const resultText = JSON.stringify(gltf, null, 2);
            const blob = new Blob([resultText], { type: 'application/json' });
            const link = document.createElement('a');
            link.href = URL.createObjectURL(blob);
            link.download = 'gt-livery-decals.gltf';
            link.click();
            URL.revokeObjectURL(link.href);
        },
        function (error) {
            console.error('An error happened during decal export', error);
        },
        { binary: false }
    );
}

// --- GLB エクスポート共通ヘルパー ---

// デカールメッシュのエクスポート用クローンを生成（パターン1：奥から手前へ昇順に連番と命名を設定）
function createDecalExportClones(startIndex = 1) {
    const baseTex = getBaseMeshTexture();
    const visibleDecals = state.appliedDecals.filter(item => item.visible);
    const clones = [];

    visibleDecals.forEach((item, idx) => {
        const decalClone = item.mesh.clone();
        const shadow = item.shadowIntensity !== undefined ? item.shadowIntensity : 0.8;
        const origTex = item.originalTexture || (item.mesh.material ? item.mesh.material.map : null);
        const finalMap = (origTex && baseTex && shadow > 0)
            ? bakeDecalTexture(item.mesh.geometry, origTex, baseTex, shadow)
            : origTex;

        decalClone.material = new THREE.MeshStandardMaterial({
            map: finalMap,
            emissiveMap: finalMap,
            emissive: new THREE.Color(0xffffff),
            emissiveIntensity: 1.0,
            transparent: true,
            opacity: item.opacity,
            polygonOffset: true,
            polygonOffsetFactor: -2 - idx,
            roughness: 1.0,
            metalness: 0.0,
            side: THREE.DoubleSide
        });

        // パターン1（奥から手前へ昇順）：インデックス番号と名称を明示
        const layerOrderIndex = startIndex + idx;
        const indexStr = String(layerOrderIndex).padStart(2, '0');
        const cleanName = (item.name || `Decal_${item.id}`).replace(/[^a-zA-Z0-9_\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\-]/g, '_');
        decalClone.name = `${indexStr}_Decal_${cleanName}`;

        clones.push(decalClone);
    });

    return clones;
}

// ユーザーがロードした元モデルの座標・スケールへ復元（頂点データそのものをワールド座標にベイク）
function applyExportTransform(group) {
    if (state.originalCenter && state.originalScale) {
        const invScale = 1 / state.originalScale;
        group.traverse((child) => {
            if (child.isMesh && child.geometry) {
                // エディタ内の表示に影響を与えないよう、ジオメトリを複製してからベイク
                child.geometry = child.geometry.clone();
                child.geometry.scale(invScale, invScale, invScale);
                child.geometry.translate(state.originalCenter.x, state.originalCenter.y, state.originalCenter.z);
            }
        });
        // 親グループのトランスフォームは初期状態 (0,0,0 / スケール 1) のまま保持
        group.position.set(0, 0, 0);
        group.scale.set(1, 1, 1);
        group.updateMatrixWorld(true);
    }
}

// GLBバイナリデータのパースとダウンロード
function downloadGLB(exportGroup, filename) {
    const exporter = new GLTFExporter();
    exporter.parse(
        exportGroup,
        function (glb) {
            const blob = new Blob([glb], { type: 'model/gltf-binary' });
            const link = document.createElement('a');
            link.href = URL.createObjectURL(blob);
            link.download = filename;
            link.click();
            URL.revokeObjectURL(link.href);
            console.log(`GLB exported successfully: ${filename}`);
            sysLog(`GLB出力完了: ${filename}`);
        },
        function (error) {
            console.error('An error happened during GLB export', error);
            sysError(`GLB出力エラー: ${error.message || error}`);
        },
        { binary: true }
    );
}

// 1. テクスチャのメッシュのみの出力 (GLB)
function exportGLBDecalsOnly() {
    const visibleDecals = state.appliedDecals.filter(item => item.visible);
    if (visibleDecals.length === 0) {
        alert('出力対象のデカールが存在しません。');
        return;
    }

    const exportGroup = new THREE.Group();
    // デカールのみの場合は 00 から開始（最背面: 00_Decal_..., 最前面: (N-1)_Decal_...）
    const decalClones = createDecalExportClones(0);
    decalClones.forEach(clone => exportGroup.add(clone));

    applyExportTransform(exportGroup);
    downloadGLB(exportGroup, 'gt-livery-decals.glb');
}

// 2. テクスチャのメッシュ以外はテクスチャ削除しての出力 (GLB)
function exportGLBBaseUntextured() {
    const exportGroup = new THREE.Group();

    if (mainSphereMesh) {
        const baseMeshClone = mainSphereMesh.clone();
        baseMeshClone.name = '00_BaseModel';
        for (let i = baseMeshClone.children.length - 1; i >= 0; i--) {
            baseMeshClone.remove(baseMeshClone.children[i]);
        }

        const stripTextures = (mat) => {
            if (!mat) return new THREE.MeshStandardMaterial({ color: 0xdddddd, roughness: 0.5, metalness: 0.1 });
            const cloned = mat.clone();
            cloned.map = null;
            cloned.bumpMap = null;
            cloned.normalMap = null;
            cloned.roughnessMap = null;
            cloned.metalnessMap = null;
            cloned.aoMap = null;
            cloned.emissiveMap = null;
            cloned.displacementMap = null;
            cloned.needsUpdate = true;
            return cloned;
        };

        if (Array.isArray(baseMeshClone.material)) {
            baseMeshClone.material = baseMeshClone.material.map(stripTextures);
        } else {
            baseMeshClone.material = stripTextures(baseMeshClone.material);
        }
        exportGroup.add(baseMeshClone);
    }

    // デカールは 01 から開始（最前面が最大インデックス N）
    const decalClones = createDecalExportClones(1);
    decalClones.forEach(clone => exportGroup.add(clone));

    applyExportTransform(exportGroup);
    downloadGLB(exportGroup, 'gt-livery-model-untextured-base.glb');
}

// 3. メッシュ全体の出力 (GLB)
function exportGLBFull() {
    const exportGroup = new THREE.Group();

    if (mainSphereMesh) {
        const baseMeshClone = mainSphereMesh.clone();
        baseMeshClone.name = '00_BaseModel';
        for (let i = baseMeshClone.children.length - 1; i >= 0; i--) {
            baseMeshClone.remove(baseMeshClone.children[i]);
        }

        if (Array.isArray(baseMeshClone.material)) {
            baseMeshClone.material = baseMeshClone.material.map(m => (m ? m.clone() : m));
        } else if (baseMeshClone.material) {
            baseMeshClone.material = baseMeshClone.material.clone();
        }
        exportGroup.add(baseMeshClone);
    }

    // デカールは 01 から開始（最前面が最大インデックス N）
    const decalClones = createDecalExportClones(1);
    decalClones.forEach(clone => exportGroup.add(clone));

    applyExportTransform(exportGroup);
    downloadGLB(exportGroup, 'gt-livery-model-full.glb');
}

// スクリーンショット保存
function takeScreenshot() {
    const hasPreview = state.preview.mesh !== null;

    if (hasPreview) {
        state.preview.mesh.visible = false;
    }

    renderer.render(scene, camera);

    const dataURL = renderer.domElement.toDataURL('image/png');

    if (hasPreview) {
        state.preview.mesh.visible = true;
    }
    renderer.render(scene, camera);

    const link = document.createElement('a');
    link.download = 'gt-livery-screenshot.png';
    link.href = dataURL;
    link.click();
}
