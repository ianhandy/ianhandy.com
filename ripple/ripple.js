// Ripple — WebGL Navier-Stokes Fluid Simulation
// Adapted from proven fluid-sim engine with autopilot + recording + FunForrest palette

'use strict';

// ── Palettes ────────────────────────────────────────────────────────────────
const PALETTES = [
    { name: 'FunForrest', bg: [0.141, 0.071, 0.0],
      colors: [[0.867,0.757,0.396],[1.0,0.914,0.639],[0.898,0.349,0.110],[0.60,0.45,0.15],[0.95,0.55,0.20]] },
    { name: 'Ember', bg: [0.06, 0.01, 0.0],
      colors: [[1.0,0.3,0.05],[1.0,0.6,0.1],[0.8,0.15,0.0],[1.0,0.85,0.3],[0.6,0.1,0.0]] },
    { name: 'Ocean', bg: [0.0, 0.02, 0.06],
      colors: [[0.1,0.4,0.8],[0.2,0.7,0.9],[0.05,0.2,0.5],[0.3,0.9,0.8],[0.0,0.3,0.6]] },
    { name: 'Aurora', bg: [0.01, 0.0, 0.03],
      colors: [[0.2,0.9,0.4],[0.5,0.3,0.9],[0.1,0.7,0.8],[0.8,0.2,0.6],[0.3,1.0,0.5]] },
    { name: 'Mono', bg: [0.03, 0.03, 0.03],
      colors: [[0.9,0.9,0.9],[0.7,0.7,0.7],[1.0,1.0,1.0],[0.5,0.5,0.5],[0.85,0.85,0.85]] },
];

// ── Configuration ───────────────────────────────────────────────────────────
const config = {
    SIM_RESOLUTION: 128,
    DYE_RESOLUTION: 1024,
    DENSITY_DISSIPATION: 0.95,
    VELOCITY_DISSIPATION: 0.98,
    PRESSURE: 0.8,
    PRESSURE_ITERATIONS: 20,
    CURL: 30,
    SPLAT_RADIUS: 0.25,
    SPLAT_FORCE: 6000,
    VISCOSITY: 0.3,
    BLOOM_INTENSITY: 0.5,
    BLOOM_THRESHOLD: 0.5,
    BLOOM_ITERATIONS: 8,
    PALETTE: 0,
    AUTOPILOT: true,
};

// ── WebGL Setup ─────────────────────────────────────────────────────────────
const canvas = document.getElementById('c');
const params = { alpha: false, depth: false, stencil: false, antialias: false, preserveDrawingBuffer: true };

let gl = canvas.getContext('webgl2', params);
const isWebGL2 = !!gl;
if (!gl) gl = canvas.getContext('webgl', params) || canvas.getContext('experimental-webgl', params);
if (!gl) { document.body.innerHTML = '<p style="color:#DDC165;padding:40px;font-family:sans-serif">WebGL required</p>'; throw new Error('No WebGL'); }

let ext;
if (isWebGL2) {
    gl.getExtension('EXT_color_buffer_float');
    ext = {
        formatRGBA: { internalFormat: gl.RGBA16F, format: gl.RGBA },
        formatRG: { internalFormat: gl.RG16F, format: gl.RG },
        formatR: { internalFormat: gl.R16F, format: gl.RED },
        halfFloatTexType: gl.HALF_FLOAT,
        supportLinearFiltering: !!gl.getExtension('OES_texture_float_linear'),
    };
} else {
    const halfFloat = gl.getExtension('OES_texture_half_float');
    ext = {
        formatRGBA: { internalFormat: gl.RGBA, format: gl.RGBA },
        formatRG: { internalFormat: gl.RGBA, format: gl.RGBA },
        formatR: { internalFormat: gl.RGBA, format: gl.RGBA },
        halfFloatTexType: halfFloat ? halfFloat.HALF_FLOAT_OES : gl.UNSIGNED_BYTE,
        supportLinearFiltering: !!gl.getExtension('OES_texture_half_float_linear'),
    };
}

const filterType = ext.supportLinearFiltering ? gl.LINEAR : gl.NEAREST;

function resizeCanvas() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.floor(canvas.clientWidth * dpr);
    const h = Math.floor(canvas.clientHeight * dpr);
    if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
    }
}
resizeCanvas();

// ── Shader Compilation ──────────────────────────────────────────────────────
function compileShader(type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS))
        throw new Error(gl.getShaderInfoLog(shader));
    return shader;
}

function createProgram(vertSrc, fragSrc) {
    const prog = gl.createProgram();
    gl.attachShader(prog, compileShader(gl.VERTEX_SHADER, vertSrc));
    gl.attachShader(prog, compileShader(gl.FRAGMENT_SHADER, fragSrc));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS))
        throw new Error(gl.getProgramInfoLog(prog));
    const uniforms = {};
    const count = gl.getProgramParameter(prog, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < count; i++) {
        const info = gl.getActiveUniform(prog, i);
        uniforms[info.name] = gl.getUniformLocation(prog, info.name);
    }
    return { program: prog, uniforms, bind() { gl.useProgram(prog); } };
}

// ── Shader Sources ──────────────────────────────────────────────────────────
const T = isWebGL2 ? 'texture' : 'texture2D';

function vertexShader() {
    if (isWebGL2) return `#version 300 es
precision highp float;
in vec2 aPosition;
out vec2 vUv;
out vec2 vL, vR, vT, vB;
uniform vec2 texelSize;
void main() {
    vUv = aPosition * 0.5 + 0.5;
    vL = vUv - vec2(texelSize.x, 0.0);
    vR = vUv + vec2(texelSize.x, 0.0);
    vT = vUv + vec2(0.0, texelSize.y);
    vB = vUv - vec2(0.0, texelSize.y);
    gl_Position = vec4(aPosition, 0.0, 1.0);
}`;
    return `precision highp float;
attribute vec2 aPosition;
varying vec2 vUv;
varying vec2 vL, vR, vT, vB;
uniform vec2 texelSize;
void main() {
    vUv = aPosition * 0.5 + 0.5;
    vL = vUv - vec2(texelSize.x, 0.0);
    vR = vUv + vec2(texelSize.x, 0.0);
    vT = vUv + vec2(0.0, texelSize.y);
    vB = vUv - vec2(0.0, texelSize.y);
    gl_Position = vec4(aPosition, 0.0, 1.0);
}`;
}

function frag(body) {
    if (isWebGL2) return `#version 300 es
precision highp float;
precision highp sampler2D;
in vec2 vUv;
in vec2 vL, vR, vT, vB;
out vec4 fragColor;
${body}`;
    return `precision highp float;
precision highp sampler2D;
varying vec2 vUv;
varying vec2 vL, vR, vT, vB;
#define fragColor gl_FragColor
${body}`;
}

const baseVert = vertexShader();

const clearShader = createProgram(baseVert, frag(`
uniform sampler2D uTexture;
uniform float value;
void main() { fragColor = value * ${T}(uTexture, vUv); }
`));

const splatShader = createProgram(baseVert, frag(`
uniform sampler2D uTarget;
uniform float aspectRatio;
uniform vec3 color;
uniform vec2 point;
uniform float radius;
void main() {
    vec2 p = vUv - point;
    p.x *= aspectRatio;
    vec3 splat = exp(-dot(p,p) / radius) * color;
    vec3 base = ${T}(uTarget, vUv).xyz;
    fragColor = vec4(base + splat, 1.0);
}
`));

const advectionShader = createProgram(baseVert, frag(`
uniform sampler2D uVelocity;
uniform sampler2D uSource;
uniform vec2 texelSize;
uniform float dt;
uniform float dissipation;
void main() {
    vec2 coord = vUv - dt * ${T}(uVelocity, vUv).xy * texelSize;
    fragColor = dissipation * ${T}(uSource, coord);
}
`));

const divergenceShader = createProgram(baseVert, frag(`
uniform sampler2D uVelocity;
void main() {
    float L = ${T}(uVelocity, vL).x;
    float R = ${T}(uVelocity, vR).x;
    float T_ = ${T}(uVelocity, vT).y;
    float B = ${T}(uVelocity, vB).y;
    fragColor = vec4(0.5 * (R - L + T_ - B), 0.0, 0.0, 1.0);
}
`));

const curlShader = createProgram(baseVert, frag(`
uniform sampler2D uVelocity;
void main() {
    float L = ${T}(uVelocity, vL).y;
    float R = ${T}(uVelocity, vR).y;
    float T_ = ${T}(uVelocity, vT).x;
    float B = ${T}(uVelocity, vB).x;
    fragColor = vec4(0.5 * (R - L - T_ + B), 0.0, 0.0, 1.0);
}
`));

const vorticityShader = createProgram(baseVert, frag(`
uniform sampler2D uVelocity;
uniform sampler2D uCurl;
uniform float curl;
uniform float dt;
void main() {
    float L = ${T}(uCurl, vL).x;
    float R = ${T}(uCurl, vR).x;
    float T_ = ${T}(uCurl, vT).x;
    float B = ${T}(uCurl, vB).x;
    float C = ${T}(uCurl, vUv).x;
    vec2 force = 0.5 * vec2(abs(T_) - abs(B), abs(R) - abs(L));
    force /= length(force) + 0.0001;
    force *= curl * C;
    force.y *= -1.0;
    vec2 velocity = ${T}(uVelocity, vUv).xy;
    velocity += force * dt;
    fragColor = vec4(velocity, 0.0, 1.0);
}
`));

const pressureShader = createProgram(baseVert, frag(`
uniform sampler2D uPressure;
uniform sampler2D uDivergence;
void main() {
    float L = ${T}(uPressure, vL).x;
    float R = ${T}(uPressure, vR).x;
    float T_ = ${T}(uPressure, vT).x;
    float B = ${T}(uPressure, vB).x;
    float divergence = ${T}(uDivergence, vUv).x;
    fragColor = vec4((L + R + B + T_ - divergence) * 0.25, 0.0, 0.0, 1.0);
}
`));

const gradientSubtractShader = createProgram(baseVert, frag(`
uniform sampler2D uPressure;
uniform sampler2D uVelocity;
void main() {
    float L = ${T}(uPressure, vL).x;
    float R = ${T}(uPressure, vR).x;
    float T_ = ${T}(uPressure, vT).x;
    float B = ${T}(uPressure, vB).x;
    vec2 velocity = ${T}(uVelocity, vUv).xy;
    velocity.xy -= vec2(R - L, T_ - B);
    fragColor = vec4(velocity, 0.0, 1.0);
}
`));

const displayShader = createProgram(baseVert, frag(`
uniform sampler2D uTexture;
uniform sampler2D uBloom;
uniform float bloomIntensity;
uniform vec3 backgroundColor;
void main() {
    vec3 c = ${T}(uTexture, vUv).rgb;
    vec3 bloom = ${T}(uBloom, vUv).rgb;
    c += bloom * bloomIntensity;
    float a = max(c.r, max(c.g, c.b));
    c = mix(backgroundColor, c, clamp(a * 2.0, 0.0, 1.0));
    fragColor = vec4(c, 1.0);
}
`));

const bloomPrefilterShader = createProgram(baseVert, frag(`
uniform sampler2D uTexture;
uniform vec3 curve;
uniform float threshold;
void main() {
    vec3 c = ${T}(uTexture, vUv).rgb;
    float br = max(c.r, max(c.g, c.b));
    float rq = clamp(br - curve.x, 0.0, curve.y);
    rq = curve.z * rq * rq;
    c *= max(rq, br - threshold) / max(br, 0.0001);
    fragColor = vec4(c, 0.0);
}
`));

const bloomBlurShader = createProgram(baseVert, frag(`
uniform sampler2D uTexture;
uniform vec2 texelSize;
uniform vec2 direction;
void main() {
    vec3 sum = vec3(0.0);
    float weights[5];
    weights[0] = 0.227027;
    weights[1] = 0.1945946;
    weights[2] = 0.1216216;
    weights[3] = 0.054054;
    weights[4] = 0.016216;
    vec2 off = direction * texelSize;
    sum += ${T}(uTexture, vUv).rgb * weights[0];
    for (int i = 1; i < 5; i++) {
        sum += ${T}(uTexture, vUv + off * float(i)).rgb * weights[i];
        sum += ${T}(uTexture, vUv - off * float(i)).rgb * weights[i];
    }
    fragColor = vec4(sum, 1.0);
}
`));

const bloomFinalShader = createProgram(baseVert, frag(`
uniform sampler2D uTexture;
uniform float intensity;
void main() { fragColor = vec4(${T}(uTexture, vUv).rgb * intensity, 1.0); }
`));

// ── Fullscreen Quad ─────────────────────────────────────────────────────────
const quadBuf = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, -1,1, 1,1, 1,-1]), gl.STATIC_DRAW);
gl.enableVertexAttribArray(0);
gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

function blit(target) {
    if (target == null) {
        gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    } else {
        gl.viewport(0, 0, target.width, target.height);
        gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbo);
    }
    gl.drawArrays(gl.TRIANGLE_FAN, 0, 4);
}

// ── Framebuffer Objects ─────────────────────────────────────────────────────
function getResolution(resolution) {
    let aspectRatio = gl.drawingBufferWidth / gl.drawingBufferHeight;
    if (aspectRatio < 1) aspectRatio = 1.0 / aspectRatio;
    const min = Math.round(resolution);
    const max = Math.round(resolution * aspectRatio);
    return gl.drawingBufferWidth > gl.drawingBufferHeight
        ? { width: max, height: min }
        : { width: min, height: max };
}

function createFBO(w, h, internalFormat, format, type, filter) {
    gl.activeTexture(gl.TEXTURE0);
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, w, h, 0, format, type, null);
    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    gl.viewport(0, 0, w, h);
    gl.clear(gl.COLOR_BUFFER_BIT);
    return { texture, fbo, width: w, height: h,
        texelSizeX: 1.0 / w, texelSizeY: 1.0 / h,
        attach(id) { gl.activeTexture(gl.TEXTURE0 + id); gl.bindTexture(gl.TEXTURE_2D, texture); return id; }
    };
}

function createDoubleFBO(w, h, internalFormat, format, type, filter) {
    let fbo1 = createFBO(w, h, internalFormat, format, type, filter);
    let fbo2 = createFBO(w, h, internalFormat, format, type, filter);
    return {
        width: w, height: h,
        texelSizeX: fbo1.texelSizeX, texelSizeY: fbo1.texelSizeY,
        get read() { return fbo1; },
        get write() { return fbo2; },
        swap() { const tmp = fbo1; fbo1 = fbo2; fbo2 = tmp; },
    };
}

// ── State ───────────────────────────────────────────────────────────────────
let dye, velocity, divergence, curl, pressure;
let bloomFramebuffers = [];

function initFramebuffers() {
    const simRes = getResolution(config.SIM_RESOLUTION);
    const dyeRes = getResolution(config.DYE_RESOLUTION);
    const texType = ext.halfFloatTexType;
    const rgba = ext.formatRGBA;
    const rg = ext.formatRG;
    const r = ext.formatR;

    dye = createDoubleFBO(dyeRes.width, dyeRes.height, rgba.internalFormat, rgba.format, texType, filterType);
    velocity = createDoubleFBO(simRes.width, simRes.height, rg.internalFormat, rg.format, texType, filterType);
    divergence = createFBO(simRes.width, simRes.height, r.internalFormat, r.format, texType, gl.NEAREST);
    curl = createFBO(simRes.width, simRes.height, r.internalFormat, r.format, texType, gl.NEAREST);
    pressure = createDoubleFBO(simRes.width, simRes.height, r.internalFormat, r.format, texType, gl.NEAREST);
    initBloomFramebuffers();
}

function initBloomFramebuffers() {
    const res = getResolution(256);
    bloomFramebuffers = [];
    const rgba = ext.formatRGBA;
    const texType = ext.halfFloatTexType;
    let w = res.width, h = res.height;
    for (let i = 0; i < config.BLOOM_ITERATIONS; i++) {
        const w1 = w >> 1, h1 = h >> 1;
        if (w1 < 2 || h1 < 2) break;
        bloomFramebuffers.push(createFBO(w1, h1, rgba.internalFormat, rgba.format, texType, filterType));
        w = w1; h = h1;
    }
}

initFramebuffers();

// ── Color Generation ────────────────────────────────────────────────────────
function generateColor() {
    const pal = PALETTES[config.PALETTE];
    const c = pal.colors[Math.floor(Math.random() * pal.colors.length)];
    return [c[0] * 0.8, c[1] * 0.8, c[2] * 0.8];
}

// ── Input ───────────────────────────────────────────────────────────────────
class Pointer {
    constructor() {
        this.id = -1;
        this.texcoordX = 0; this.texcoordY = 0;
        this.prevTexcoordX = 0; this.prevTexcoordY = 0;
        this.deltaX = 0; this.deltaY = 0;
        this.down = false; this.moved = false;
        this.color = [0.3, 0, 0];
    }
}

let pointers = [new Pointer()];

function updatePointerDownData(pointer, id, posX, posY) {
    pointer.id = id;
    pointer.down = true;
    pointer.moved = false;
    pointer.texcoordX = posX / canvas.clientWidth;
    pointer.texcoordY = 1.0 - posY / canvas.clientHeight;
    pointer.prevTexcoordX = pointer.texcoordX;
    pointer.prevTexcoordY = pointer.texcoordY;
    pointer.deltaX = 0; pointer.deltaY = 0;
    pointer.color = generateColor();
}

function updatePointerMoveData(pointer, posX, posY) {
    pointer.prevTexcoordX = pointer.texcoordX;
    pointer.prevTexcoordY = pointer.texcoordY;
    pointer.texcoordX = posX / canvas.clientWidth;
    pointer.texcoordY = 1.0 - posY / canvas.clientHeight;
    pointer.deltaX = correctDeltaX(pointer.texcoordX - pointer.prevTexcoordX);
    pointer.deltaY = correctDeltaY(pointer.texcoordY - pointer.prevTexcoordY);
    pointer.moved = Math.abs(pointer.deltaX) > 0 || Math.abs(pointer.deltaY) > 0;
}

function correctDeltaX(delta) {
    const ar = canvas.clientWidth / canvas.clientHeight;
    return ar < 1 ? delta * ar : delta;
}

function correctDeltaY(delta) {
    const ar = canvas.clientWidth / canvas.clientHeight;
    return ar > 1 ? delta / ar : delta;
}

canvas.addEventListener('mousedown', e => {
    updatePointerDownData(pointers[0], -1, e.offsetX, e.offsetY);
});
canvas.addEventListener('mousemove', e => {
    if (!pointers[0].down) return;
    updatePointerMoveData(pointers[0], e.offsetX, e.offsetY);
});
window.addEventListener('mouseup', () => { pointers[0].down = false; });

canvas.addEventListener('touchstart', e => {
    e.preventDefault();
    const touches = e.targetTouches;
    while (pointers.length < touches.length) pointers.push(new Pointer());
    for (let i = 0; i < touches.length; i++)
        updatePointerDownData(pointers[i], touches[i].identifier, touches[i].pageX, touches[i].pageY);
}, { passive: false });

canvas.addEventListener('touchmove', e => {
    e.preventDefault();
    for (const touch of e.targetTouches) {
        const ptr = pointers.find(p => p.id === touch.identifier);
        if (ptr) updatePointerMoveData(ptr, touch.pageX, touch.pageY);
    }
}, { passive: false });

window.addEventListener('touchend', e => {
    for (const touch of e.changedTouches) {
        const ptr = pointers.find(p => p.id === touch.identifier);
        if (ptr) ptr.down = false;
    }
});

// ── Simulation ──────────────────────────────────────────────────────────────
function splat(x, y, dx, dy, color) {
    splatShader.bind();
    gl.uniform1i(splatShader.uniforms.uTarget, velocity.read.attach(0));
    gl.uniform1f(splatShader.uniforms.aspectRatio, canvas.clientWidth / canvas.clientHeight);
    gl.uniform2f(splatShader.uniforms.point, x, y);
    gl.uniform3f(splatShader.uniforms.color, dx, dy, 0.0);
    gl.uniform1f(splatShader.uniforms.radius, correctRadius(config.SPLAT_RADIUS / 100.0));
    blit(velocity.write);
    velocity.swap();

    gl.uniform1i(splatShader.uniforms.uTarget, dye.read.attach(0));
    gl.uniform3f(splatShader.uniforms.color, color[0], color[1], color[2]);
    blit(dye.write);
    dye.swap();
}

function correctRadius(radius) {
    const ar = canvas.clientWidth / canvas.clientHeight;
    return ar > 1 ? radius * ar : radius;
}

function multipleSplats(amount) {
    for (let i = 0; i < amount; i++) {
        const color = generateColor();
        color[0] *= 3.0; color[1] *= 3.0; color[2] *= 3.0;
        const x = Math.random();
        const y = Math.random();
        const dx = 600 * (Math.random() - 0.5);
        const dy = 600 * (Math.random() - 0.5);
        splat(x, y, dx, dy, color);
    }
}

function step(dt) {
    gl.disable(gl.BLEND);

    curlShader.bind();
    gl.uniform2f(curlShader.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY);
    gl.uniform1i(curlShader.uniforms.uVelocity, velocity.read.attach(0));
    blit(curl);

    vorticityShader.bind();
    gl.uniform2f(vorticityShader.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY);
    gl.uniform1i(vorticityShader.uniforms.uVelocity, velocity.read.attach(0));
    gl.uniform1i(vorticityShader.uniforms.uCurl, curl.attach(1));
    gl.uniform1f(vorticityShader.uniforms.curl, config.CURL);
    gl.uniform1f(vorticityShader.uniforms.dt, dt);
    blit(velocity.write);
    velocity.swap();

    divergenceShader.bind();
    gl.uniform2f(divergenceShader.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY);
    gl.uniform1i(divergenceShader.uniforms.uVelocity, velocity.read.attach(0));
    blit(divergence);

    clearShader.bind();
    gl.uniform1i(clearShader.uniforms.uTexture, pressure.read.attach(0));
    gl.uniform1f(clearShader.uniforms.value, config.PRESSURE);
    blit(pressure.write);
    pressure.swap();

    pressureShader.bind();
    gl.uniform2f(pressureShader.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY);
    gl.uniform1i(pressureShader.uniforms.uDivergence, divergence.attach(0));
    for (let i = 0; i < config.PRESSURE_ITERATIONS; i++) {
        gl.uniform1i(pressureShader.uniforms.uPressure, pressure.read.attach(1));
        blit(pressure.write);
        pressure.swap();
    }

    gradientSubtractShader.bind();
    gl.uniform2f(gradientSubtractShader.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY);
    gl.uniform1i(gradientSubtractShader.uniforms.uPressure, pressure.read.attach(0));
    gl.uniform1i(gradientSubtractShader.uniforms.uVelocity, velocity.read.attach(1));
    blit(velocity.write);
    velocity.swap();

    advectionShader.bind();
    gl.uniform2f(advectionShader.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY);
    gl.uniform1i(advectionShader.uniforms.uVelocity, velocity.read.attach(0));
    gl.uniform1i(advectionShader.uniforms.uSource, velocity.read.attach(0));
    gl.uniform1f(advectionShader.uniforms.dt, dt);
    gl.uniform1f(advectionShader.uniforms.dissipation, config.VELOCITY_DISSIPATION);
    blit(velocity.write);
    velocity.swap();

    gl.uniform2f(advectionShader.uniforms.texelSize, dye.texelSizeX, dye.texelSizeY);
    gl.uniform1i(advectionShader.uniforms.uVelocity, velocity.read.attach(0));
    gl.uniform1i(advectionShader.uniforms.uSource, dye.read.attach(1));
    gl.uniform1f(advectionShader.uniforms.dissipation, config.DENSITY_DISSIPATION);
    blit(dye.write);
    dye.swap();
}

// ── Bloom ───────────────────────────────────────────────────────────────────
function applyBloom(source, destination) {
    if (bloomFramebuffers.length < 2) return;

    bloomPrefilterShader.bind();
    const knee = config.BLOOM_THRESHOLD * 0.7;
    gl.uniform3f(bloomPrefilterShader.uniforms.curve,
        config.BLOOM_THRESHOLD - knee, knee * 2.0, 0.25 / knee);
    gl.uniform1f(bloomPrefilterShader.uniforms.threshold, config.BLOOM_THRESHOLD);
    gl.uniform1i(bloomPrefilterShader.uniforms.uTexture, source.attach(0));
    blit(bloomFramebuffers[0]);

    bloomBlurShader.bind();
    for (let i = 0; i < bloomFramebuffers.length - 1; i++) {
        const dest = bloomFramebuffers[i + 1];
        gl.uniform2f(bloomBlurShader.uniforms.texelSize,
            bloomFramebuffers[i].texelSizeX, bloomFramebuffers[i].texelSizeY);
        gl.uniform2f(bloomBlurShader.uniforms.direction, 1.0, 0.0);
        gl.uniform1i(bloomBlurShader.uniforms.uTexture, bloomFramebuffers[i].attach(0));
        blit(dest);
    }

    for (let i = bloomFramebuffers.length - 1; i > 0; i--) {
        gl.uniform2f(bloomBlurShader.uniforms.texelSize,
            bloomFramebuffers[i].texelSizeX, bloomFramebuffers[i].texelSizeY);
        gl.uniform2f(bloomBlurShader.uniforms.direction, 0.0, 1.0);
        gl.uniform1i(bloomBlurShader.uniforms.uTexture, bloomFramebuffers[i].attach(0));
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.ONE, gl.ONE);
        blit(bloomFramebuffers[i - 1]);
        gl.disable(gl.BLEND);
    }

    bloomFinalShader.bind();
    gl.uniform1i(bloomFinalShader.uniforms.uTexture, bloomFramebuffers[0].attach(0));
    gl.uniform1f(bloomFinalShader.uniforms.intensity, config.BLOOM_INTENSITY);
    blit(destination);
}

function render(target) {
    const bg = PALETTES[config.PALETTE].bg;

    if (config.BLOOM_INTENSITY > 0 && bloomFramebuffers.length > 1) {
        const bloomDest = bloomFramebuffers[bloomFramebuffers.length - 1];
        applyBloom(dye.read, bloomDest);
        displayShader.bind();
        gl.uniform1i(displayShader.uniforms.uTexture, dye.read.attach(0));
        gl.uniform1i(displayShader.uniforms.uBloom, bloomFramebuffers[0].attach(1));
        gl.uniform1f(displayShader.uniforms.bloomIntensity, config.BLOOM_INTENSITY);
    } else {
        displayShader.bind();
        gl.uniform1i(displayShader.uniforms.uTexture, dye.read.attach(0));
        gl.uniform1i(displayShader.uniforms.uBloom, dye.read.attach(1));
        gl.uniform1f(displayShader.uniforms.bloomIntensity, 0.0);
    }
    gl.uniform3f(displayShader.uniforms.backgroundColor, bg[0], bg[1], bg[2]);
    blit(target);
}

// ── Autopilot ───────────────────────────────────────────────────────────────
let autopilotTime = 0;
const autopilotAgents = [];

function initAutopilotAgents() {
    autopilotAgents.length = 0;
    for (let i = 0; i < 4; i++) {
        autopilotAgents.push({
            x: 0.2 + Math.random() * 0.6,
            y: 0.2 + Math.random() * 0.6,
            vx: (Math.random() - 0.5) * 0.01,
            vy: (Math.random() - 0.5) * 0.01,
            phase: Math.random() * Math.PI * 2,
            freq: 0.3 + Math.random() * 0.6,
            color: generateColor(),
            colorTimer: 0,
        });
    }
}
initAutopilotAgents();

function updateAutopilot(dt) {
    if (!config.AUTOPILOT) return;
    autopilotTime += dt;

    for (const agent of autopilotAgents) {
        const t = autopilotTime * agent.freq;
        agent.vx += Math.sin(t + agent.phase) * 0.0004;
        agent.vy += Math.cos(t * 1.3 + agent.phase * 0.7) * 0.0004;
        agent.vx += (Math.random() - 0.5) * 0.0002;
        agent.vy += (Math.random() - 0.5) * 0.0002;
        agent.vx *= 0.99;
        agent.vy *= 0.99;

        const maxSpd = 0.02;
        const spd = Math.sqrt(agent.vx * agent.vx + agent.vy * agent.vy);
        if (spd > maxSpd) { agent.vx = (agent.vx / spd) * maxSpd; agent.vy = (agent.vy / spd) * maxSpd; }

        agent.x += agent.vx;
        agent.y += agent.vy;

        if (agent.x < 0.05 || agent.x > 0.95) { agent.vx *= -0.8; agent.x = Math.max(0.05, Math.min(0.95, agent.x)); }
        if (agent.y < 0.05 || agent.y > 0.95) { agent.vy *= -0.8; agent.y = Math.max(0.05, Math.min(0.95, agent.y)); }

        agent.colorTimer += dt;
        if (agent.colorTimer > 2 + Math.random() * 3) {
            agent.color = generateColor();
            agent.colorTimer = 0;
        }

        // Only splat every few frames to avoid oversaturation
        agent.splatTimer = (agent.splatTimer || 0) + dt;
        if (agent.splatTimer > 0.05) {
            agent.splatTimer = 0;
            const dx = correctDeltaX(agent.vx) * config.SPLAT_FORCE * 0.3;
            const dy = correctDeltaY(agent.vy) * config.SPLAT_FORCE * 0.3;
            const color = [agent.color[0] * 0.8, agent.color[1] * 0.8, agent.color[2] * 0.8];
            splat(agent.x, agent.y, dx, dy, color);
        }
    }

    // Random burst
    if (Math.random() < 0.01) {
        const color = generateColor();
        color[0] *= 4; color[1] *= 4; color[2] *= 4;
        const x = 0.1 + Math.random() * 0.8;
        const y = 0.1 + Math.random() * 0.8;
        const dx = 500 * (Math.random() - 0.5);
        const dy = 500 * (Math.random() - 0.5);
        splat(x, y, dx, dy, color);
    }
}

// ── Recording ───────────────────────────────────────────────────────────────
let mediaRecorder = null;
let recordedChunks = [];
let recordStartTime = 0;
const recIndicator = document.getElementById('recIndicator');
const recTimeEl = document.getElementById('recTime');

function startRecording() {
    const duration = parseInt(document.getElementById('recDuration').value, 10);
    const stream = canvas.captureStream(60);
    recordedChunks = [];
    const mimeType = MediaRecorder.isTypeSupported('video/webm;codecs=vp9')
        ? 'video/webm;codecs=vp9' : 'video/webm';
    mediaRecorder = new MediaRecorder(stream, { mimeType, videoBitsPerSecond: 8000000 });
    mediaRecorder.ondataavailable = e => { if (e.data.size > 0) recordedChunks.push(e.data); };
    mediaRecorder.onstop = () => {
        const blob = new Blob(recordedChunks, { type: 'video/webm' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = `ripple-wallpaper-${Date.now()}.webm`;
        a.click();
        URL.revokeObjectURL(a.href);
        btnRecord.classList.remove('active');
        recIndicator.classList.remove('visible');
    };
    mediaRecorder.start();
    recordStartTime = performance.now();
    btnRecord.classList.add('active');
    recIndicator.classList.add('visible');
    setTimeout(() => { if (mediaRecorder && mediaRecorder.state === 'recording') mediaRecorder.stop(); }, duration * 1000);
}

// ── UI ──────────────────────────────────────────────────────────────────────
const btnAutopilot = document.getElementById('btnAutopilot');
const btnRecord = document.getElementById('btnRecord');
const btnPalette = document.getElementById('btnPalette');
const btnSettings = document.getElementById('btnSettings');
const settingsPanel = document.getElementById('settingsPanel');
const titleOverlay = document.getElementById('titleOverlay');
const uiPanel = document.getElementById('uiPanel');

btnAutopilot.addEventListener('click', () => {
    config.AUTOPILOT = !config.AUTOPILOT;
    btnAutopilot.classList.toggle('active', config.AUTOPILOT);
    titleOverlay.querySelector('p').textContent = config.AUTOPILOT
        ? 'Touch to create \u00b7 Auto-pilot active' : 'Touch to create';
});

btnRecord.addEventListener('click', () => {
    if (mediaRecorder && mediaRecorder.state === 'recording') { mediaRecorder.stop(); }
    else { startRecording(); }
});

btnPalette.addEventListener('click', () => {
    config.PALETTE = (config.PALETTE + 1) % PALETTES.length;
    btnPalette.textContent = PALETTES[config.PALETTE].name;
    for (const agent of autopilotAgents) agent.color = generateColor();
    // Set body background to match palette
    const bg = PALETTES[config.PALETTE].bg;
    document.body.style.background = `rgb(${Math.round(bg[0]*255)},${Math.round(bg[1]*255)},${Math.round(bg[2]*255)})`;
});
btnPalette.textContent = PALETTES[0].name;

btnSettings.addEventListener('click', () => settingsPanel.classList.toggle('open'));
document.getElementById('closeSettings').addEventListener('click', () => settingsPanel.classList.remove('open'));

function bindSlider(id, valId, configKey, transform) {
    const slider = document.getElementById(id);
    const valEl = document.getElementById(valId);
    slider.addEventListener('input', () => {
        const v = parseFloat(slider.value);
        config[configKey] = transform ? transform(v) : v;
        valEl.textContent = slider.value;
    });
}

bindSlider('viscosity', 'viscVal', 'VISCOSITY');
bindSlider('curl', 'curlVal', 'CURL');
bindSlider('pressure', 'pressVal', 'PRESSURE_ITERATIONS', v => Math.round(v));
bindSlider('splatRadius', 'splatVal', 'SPLAT_RADIUS');
bindSlider('splatForce', 'forceVal', 'SPLAT_FORCE');
bindSlider('dissipation', 'dissVal', 'DENSITY_DISSIPATION');
bindSlider('bloomIntensity', 'bloomVal', 'BLOOM_INTENSITY');
bindSlider('bloomThreshold', 'bloomThreshVal', 'BLOOM_THRESHOLD');
document.getElementById('recDuration').addEventListener('input', function() {
    document.getElementById('durVal').textContent = this.value;
});

// Fade UI after inactivity
let uiTimer = null;
function resetUIFade() {
    clearTimeout(uiTimer);
    uiPanel.classList.remove('faded');
    titleOverlay.classList.remove('faded');
    uiTimer = setTimeout(() => { uiPanel.classList.add('faded'); titleOverlay.classList.add('faded'); }, 2000);
}
canvas.addEventListener('pointermove', resetUIFade);
canvas.addEventListener('pointerdown', resetUIFade);
resetUIFade();

// ── Main Loop ───────────────────────────────────────────────────────────────
let lastUpdateTime = Date.now();

// Initial splashes
multipleSplats(Math.floor(Math.random() * 4) + 3);

function update() {
    const now = Date.now();
    let dt = (now - lastUpdateTime) / 1000;
    dt = Math.min(dt, 0.016666);
    lastUpdateTime = now;

    resizeCanvas();
    updateAutopilot(dt);

    for (const p of pointers) {
        if (p.moved) {
            p.moved = false;
            const dx = p.deltaX * config.SPLAT_FORCE;
            const dy = p.deltaY * config.SPLAT_FORCE;
            splat(p.texcoordX, p.texcoordY, dx, dy, p.color);
        }
    }

    step(dt);
    render(null);

    // Update record timer
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        const s = Math.floor((performance.now() - recordStartTime) / 1000);
        recTimeEl.textContent = `${String(Math.floor(s/60)).padStart(2,'0')}:${String(s%60).padStart(2,'0')}`;
    }

    requestAnimationFrame(update);
}

requestAnimationFrame(update);
