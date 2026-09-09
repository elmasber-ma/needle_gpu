// Kernels WGSL del forward Needle v2 en GPU (todo f32).
// Convenciones: buffers row-major, un dispatch por op, entry points chicos.

// ---------------------------------------------------------------- RMSNorm
struct NormP { n: u32, has_gamma: u32, _p0: u32, _p1: u32 };
@group(0) @binding(0) var<storage, read_write> nrm_x: array<f32>;
@group(0) @binding(1) var<storage, read> nrm_g: array<f32>;
@group(0) @binding(2) var<uniform> nrm_p: NormP;
var<workgroup> nrm_acc: array<f32, 256>;
@compute @workgroup_size(256)
fn rms_norm(@builtin(local_invocation_id) li: vec3<u32>) {
  let t = li.x;
  let n = nrm_p.n;
  var s: f32 = 0.0;
  var i = t;
  while (i < n) {
    let v = nrm_x[i];
    s += v * v;
    i += 256u;
  }
  nrm_acc[t] = s;
  workgroupBarrier();
  var stride = 128u;
  while (stride > 0u) {
    if (t < stride) { nrm_acc[t] += nrm_acc[t + stride]; }
    workgroupBarrier();
    stride = stride / 2u;
  }
  let inv = inverseSqrt(nrm_acc[0u] / f32(n) + 0.000001);
  i = t;
  if (nrm_p.has_gamma == 1u) {
    while (i < n) {
      nrm_x[i] = (1.0 + nrm_g[i]) * nrm_x[i] * inv;
      i += 256u;
    }
  } else {
    while (i < n) {
      nrm_x[i] = nrm_x[i] * inv;
      i += 256u;
    }
  }
}

// ---------------------------------------------------------------- norm por head
// cada bloque de 64 con SU media, gamma[64] compartido.
@group(0) @binding(0) var<storage, read_write> rh_x: array<f32>;
@group(0) @binding(1) var<storage, read> rh_g: array<f32>;
@group(0) @binding(2) var<uniform> rh_n: vec4<u32>;
@compute @workgroup_size(64)
fn rms_heads(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= rh_n.x) { return; }
  let base = (i / 64u) * 64u;
  var s: f32 = 0.0;
  for (var j = 0u; j < 64u; j++) {
    let v = rh_x[base + j];
    s += v * v;
  }
  let inv = inverseSqrt(s / 64.0 + 0.000001);
  rh_x[i] = (1.0 + rh_g[i % 64u]) * rh_x[i] * inv;
}

// ---------------------------------------------------------------- matvec
// y[off..off+rows] = W[(off..off+rows), :] @ x ; W row-major [Wrows, cols]
struct MvP { rows: u32, cols: u32, off: u32, _p: u32 };
@group(0) @binding(0) var<storage, read_write> mv_y: array<f32>;
@group(0) @binding(1) var<storage, read> mv_w: array<f32>;
@group(0) @binding(2) var<storage, read> mv_x: array<f32>;
@group(0) @binding(3) var<uniform> mv_p: MvP;
@compute @workgroup_size(64)
fn matvec(@builtin(global_invocation_id) id: vec3<u32>) {
  let r = id.x;
  if (r >= mv_p.rows) { return; }
  let cols = mv_p.cols;
  let base = (mv_p.off + r) * cols;
  var acc: f32 = 0.0;
  for (var c = 0u; c < cols; c++) {
    acc += mv_w[base + c] * mv_x[c];
  }
  mv_y[mv_p.off + r] = acc;
}

// ---------------------------------------------------------------- RoPE
// split-half sobre n_heads bloques de 64: [a(32)|b(32)] -> rotados.
struct RopeP { pos: u32, n_heads: u32, _p0: u32, _p1: u32 };
@group(0) @binding(0) var<storage, read_write> rp_b: array<f32>;
@group(0) @binding(1) var<storage, read> rp_t: array<f32>; // [t*64 + 2i]=cos,[+1]=sin
@group(0) @binding(2) var<uniform> rp_p: RopeP;
@compute @workgroup_size(64)
fn rope(@builtin(global_invocation_id) id: vec3<u32>) {
  let pair = id.x; // 0 .. n_heads*32
  if (pair >= rp_p.n_heads * 32u) { return; }
  let h = pair / 32u;
  let i = pair % 32u;
  let c = rp_t[rp_p.pos * 64u + 2u * i];
  let s = rp_t[rp_p.pos * 64u + 2u * i + 1u];
  let a = rp_b[h * 64u + i];
  let b = rp_b[h * 64u + 32u + i];
  rp_b[h * 64u + i] = a * c - b * s;
  rp_b[h * 64u + 32u + i] = b * c + a * s;
}

// ---------------------------------------------------------------- CQ prepare
// xh = FWHT128 por grupo de x (zero-pad más allá de in_feat).
// Un workgroup (128 hilos) por grupo.
struct CqpP { in_feat: u32, ngroups: u32, _p0: u32, _p1: u32 };
@group(0) @binding(0) var<storage, read_write> cqp_xh: array<f32>; // [ngroups*128]
@group(0) @binding(1) var<storage, read> cqp_x: array<f32>;
@group(0) @binding(2) var<uniform> cqp_p: CqpP;
var<workgroup> cqp_s: array<f32, 128>;
@compute @workgroup_size(128)
fn cq_prepare(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) li: vec3<u32>
) {
  let g = wg.x;
  if (g >= cqp_p.ngroups) { return; }
  let t = li.x;
  let src = g * 128u + t;
  cqp_s[t] = select(0.0, cqp_x[src], src < cqp_p.in_feat);
  workgroupBarrier();
  var s = 0u;
  while (s < 7u) {
    let stride = 1u << s;
    let base = (t / stride) * (stride * 2u);
    let i0 = base + (t % stride);
    let i1 = i0 + stride;
    let a = cqp_s[i0];
    let b = cqp_s[i1];
    cqp_s[i0] = (a + b) * 0.70710678;
    cqp_s[i1] = (a - b) * 0.70710678;
    workgroupBarrier();
    s += 1u;
  }
  cqp_xh[g * 128u + t] = cqp_s[t];
}

// ---------------------------------------------------------------- CQ matvec
// y[row_start..] = W_cq @ xh ; W en bits LSB-first, niveles+normas f32.
// Un hilo por fila.
struct CqmP {
  rows: u32, row_start: u32, ngroups: u32, group: u32,
  bits: u32, row_u32: u32, is_ternary: u32, nlevels: u32,
};
@group(0) @binding(0) var<storage, read_write> cqm_y: array<f32>;
@group(0) @binding(1) var<storage, read> cqm_packed: array<u32>;
@group(0) @binding(2) var<storage, read> cqm_norms: array<f32>; // [out*ngroups]
@group(0) @binding(3) var<storage, read> cqm_xh: array<f32>;
@group(0) @binding(4) var<storage, read> cqm_lv: array<f32>;
@group(0) @binding(5) var<uniform> cqm_p: CqmP;
@compute @workgroup_size(64)
fn cq_matvec(@builtin(global_invocation_id) id: vec3<u32>) {
  let r = id.x;
  if (r >= cqm_p.rows) { return; }
  let row = cqm_p.row_start + r;
  let group = cqm_p.group;
  // P = niveles por byte (2 bits->4, 4 bits->2, ternario->4).
  var P = 2u;
  if (cqm_p.is_ternary == 1u) { P = 4u; }
  else if (cqm_p.bits == 2u) { P = 4u; }
  let width = select(cqm_p.bits, 2u, cqm_p.is_ternary == 1u);
  let mask = (1u << width) - 1u;
  let per_iter = 8u / P;
  let nbytes = group / P;
  var total: f32 = 0.0;
  var g = 0u;
  while (g < cqm_p.ngroups) {
    let norm = cqm_norms[row * cqm_p.ngroups + g];
    let row_base = row * cqm_p.row_u32;
    let gx_base = g * group;
    var lanes: array<f32, 8>;
    for (var q = 0u; q < 8u; q++) { lanes[q] = 0.0; }
    var bi = 0u;
    let full = nbytes - nbytes % per_iter;
    while (bi < full) {
      var vals: array<f32, 8>;
      for (var t = 0u; t < per_iter; t++) {
        let bb = g * nbytes + bi + t;
        let word = bb / 4u;
        let byte = cqm_packed[row_base + word] >> ((bb % 4u) * 8u);
        for (var k = 0u; k < P; k++) {
          var code = (byte >> (k * width)) & mask;
          var idx = code;
          if (cqm_p.is_ternary == 1u) {
            if (code == 3u) { idx = 0u; }
            else if (code == 0u) { idx = 1u; }
            else { idx = 2u; }
          }
          vals[t * P + k] = cqm_lv[idx];
        }
      }
      for (var k = 0u; k < 8u; k++) {
        lanes[k] += vals[k] * cqm_xh[gx_base + bi * P + k];
      }
      bi += per_iter;
    }
    var acc = ((((((lanes[0u] + lanes[1u]) + lanes[2u]) + lanes[3u]) + lanes[4u]) + lanes[5u]) + lanes[6u]) + lanes[7u];
    var b = bi;
    while (b < nbytes) {
      let bb = g * nbytes + b;
      let word = bb / 4u;
      let byte = cqm_packed[row_base + word] >> ((bb % 4u) * 8u);
      for (var k = 0u; k < P; k++) {
        var code = (byte >> (k * width)) & mask;
        var idx = code;
        if (cqm_p.is_ternary == 1u) {
          if (code == 3u) { idx = 0u; }
          else if (code == 0u) { idx = 1u; }
          else { idx = 2u; }
        }
        acc += cqm_lv[idx] * cqm_xh[gx_base + b * P + k];
      }
      b += 1u;
    }
    total += norm * acc;
    g += 1u;
  }
  cqm_y[r] = total;
}

// ---------------------------------------------------------------- KV write
// K[slot*256..] = k[0..256], V igual. Un dispatch por capa.
struct KvP { slot: u32, _p0: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<storage, read> kv_k: array<f32>;
@group(0) @binding(1) var<storage, read> kv_v: array<f32>;
@group(0) @binding(2) var<storage, read_write> kv_K: array<f32>;
@group(0) @binding(3) var<storage, read_write> kv_V: array<f32>;
@group(0) @binding(4) var<uniform> kv_p: KvP;
@compute @workgroup_size(64)
fn kv_write(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= 256u) { return; }
  let base = kv_p.slot * 256u + i;
  kv_K[base] = kv_k[i];
  kv_V[base] = kv_v[i];
}

// ---------------------------------------------------------------- Attention
// 8 heads GQA (kv head = h/2), ventana [lo..pos] sobre ring de 256.
struct AttnP { pos: u32, lo: u32, scale: f32, _p: u32 };
@group(0) @binding(0) var<storage, read> at_q: array<f32>;   // [512]
@group(0) @binding(1) var<storage, read> at_K: array<f32>;   // [65536]
@group(0) @binding(2) var<storage, read> at_V: array<f32>;   // [65536]
@group(0) @binding(3) var<storage, read_write> at_o: array<f32>; // [512]
@group(0) @binding(4) var<uniform> at_p: AttnP;
var<workgroup> at_acc: array<f32, 64>;
var<workgroup> at_scr: array<f32, 256>;
@compute @workgroup_size(64)
fn attn(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) li: vec3<u32>
) {
  let h = wg.x;
  if (h >= 8u) { return; }
  let kvh = h / 2u;
  let d = li.x;
  let qd = at_q[h * 64u + d];
  let lo = at_p.lo;
  let pos = at_p.pos;
  var mx: f32 = -1e30;
  var t = lo;
  while (t <= pos) {
    at_acc[d] = qd * at_K[(t % 256u) * 256u + kvh * 64u + d];
    workgroupBarrier();
    if (d < 32u) { at_acc[d] += at_acc[d + 32u]; }
    workgroupBarrier();
    if (d < 16u) { at_acc[d] += at_acc[d + 16u]; }
    workgroupBarrier();
    if (d < 8u) { at_acc[d] += at_acc[d + 8u]; }
    workgroupBarrier();
    if (d < 4u) { at_acc[d] += at_acc[d + 4u]; }
    workgroupBarrier();
    if (d < 2u) { at_acc[d] += at_acc[d + 2u]; }
    workgroupBarrier();
    if (d < 1u) { at_acc[d] += at_acc[d + 1u]; }
    workgroupBarrier();
    let st = at_acc[0u] * at_p.scale;
    at_scr[t - lo] = st;
    if (st > mx) { mx = st; }
    workgroupBarrier();
    t += 1u;
  }
  let n = pos - lo + 1u;
  var sum: f32 = 0.0;
  var i = 0u;
  while (i < n) {
    sum += exp(at_scr[i] - mx);
    i += 1u;
  }
  var ov: f32 = 0.0;
  t = lo;
  while (t <= pos) {
    let pr = exp(at_scr[t - lo] - mx) / sum;
    ov += pr * at_V[(t % 256u) * 256u + kvh * 64u + d];
    t += 1u;
  }
  at_o[h * 64u + d] = ov;
}

// ---------------------------------------------------------------- Elementales
@group(0) @binding(0) var<storage, read_write> ew_a: array<f32>;
@group(0) @binding(1) var<storage, read> ew_b: array<f32>;
@group(0) @binding(2) var<uniform> ew_p: vec4<u32>; // n, modo, 0, 0
// modo 0: a *= sigmoid(b) | 1: a = a*b | 2: a = silu(a*b) | 3: a = sigmoid(a)
@compute @workgroup_size(64)
fn elem(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= ew_p.x) { return; }
  let m = ew_p.y;
  if (m == 0u) {
    ew_a[i] = ew_a[i] * (1.0 / (1.0 + exp(-ew_b[i])));
  } else if (m == 1u) {
    ew_a[i] = ew_a[i] * ew_b[i];
  } else if (m == 2u) {
    let v = ew_a[i] * ew_b[i];
    ew_a[i] = v / (1.0 + exp(-v));
  } else {
    ew_a[i] = 1.0 / (1.0 + exp(-ew_a[i]));
  }
}

// y = a + s*b  (s escalar f32 en uniform)
struct AxP { s: f32, n: u32, _p0: u32, _p1: u32 };
@group(0) @binding(0) var<storage, read_write> ax_y: array<f32>;
@group(0) @binding(1) var<storage, read> ax_a: array<f32>;
@group(0) @binding(2) var<storage, read> ax_b: array<f32>;
@group(0) @binding(3) var<uniform> ax_p: AxP;
@compute @workgroup_size(64)
fn axpy(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= ax_p.n) { return; }
  ax_y[i] = ax_a[i] + ax_p.s * ax_b[i];
}

// y = a + s[0]*b  (escalar desde buffer: alpha queda en GPU)
@group(0) @binding(0) var<storage, read_write> ab_y: array<f32>;
@group(0) @binding(1) var<storage, read> ab_a: array<f32>;
@group(0) @binding(2) var<storage, read> ab_s: array<f32>;
@group(0) @binding(3) var<storage, read> ab_b: array<f32>;
@group(0) @binding(4) var<uniform> ab_n: vec4<u32>;
@compute @workgroup_size(64)
fn axpy_b(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= ab_n.x) { return; }
  ab_y[i] = ab_a[i] + ab_s[0u] * ab_b[i];
}

// y = y + m - u
@group(0) @binding(0) var<storage, read_write> cm_y: array<f32>;
@group(0) @binding(1) var<storage, read> cm_m: array<f32>;
@group(0) @binding(2) var<storage, read> cm_u: array<f32>;
@group(0) @binding(3) var<uniform> cm_n: vec4<u32>;
@compute @workgroup_size(64)
fn combine(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= cm_n.x) { return; }
  cm_y[i] = cm_y[i] + cm_m[i] - cm_u[i];
}

// dst = src (n elems)
@group(0) @binding(0) var<storage, read_write> cp_d: array<f32>;
@group(0) @binding(1) var<storage, read> cp_s: array<f32>;
@group(0) @binding(2) var<uniform> cp_n: vec4<u32>;
@compute @workgroup_size(64)
fn copy_buf(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= cp_n.x) { return; }
  cp_d[i] = cp_s[i];
}

// ---------------------------------------------------------------- mHC chico
// z[i] = k*sigmoid(a*x[i] + b[i] + off[i])  (k=1 hpre, k=2 hpost)
struct SgP { a: f32, k: f32, n: u32, _p: u32 };
@group(0) @binding(0) var<storage, read_write> sg_z: array<f32>;
@group(0) @binding(1) var<storage, read> sg_x: array<f32>;
@group(0) @binding(2) var<storage, read> sg_b: array<f32>;
@group(0) @binding(3) var<storage, read> sg_o: array<f32>;
@group(0) @binding(4) var<uniform> sg_p: SgP;
@compute @workgroup_size(64)
fn sigmoid_affine(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= sg_p.n) { return; }
  sg_z[i] = sg_p.k / (1.0 + exp(-(sg_p.a * sg_x[i] + sg_b[i] + sg_o[i])));
}

// u[i] = sum_l h[l]*lanes[l*512+i]  (4 lanes, n=512)
@group(0) @binding(0) var<storage, read_write> mx_u: array<f32>;
@group(0) @binding(1) var<storage, read> mx_h: array<f32>; // [4]
@group(0) @binding(2) var<storage, read> mx_l: array<f32>; // [2048]
@compute @workgroup_size(64)
fn lane_mix(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= 512u) { return; }
  mx_u[i] = mx_h[0u] * mx_l[i] + mx_h[1u] * mx_l[512u + i]
    + mx_h[2u] * mx_l[1024u + i] + mx_h[3u] * mx_l[1536u + i];
}

// lanesN = hres*lanes + hpost*y  (todo [2048]/[4]/[16])
@group(0) @binding(0) var<storage, read_write> ln_n: array<f32>; // [2048]
@group(0) @binding(1) var<storage, read> ln_l: array<f32>;       // [2048]
@group(0) @binding(2) var<storage, read> ln_r: array<f32>;       // [16] hres
@group(0) @binding(3) var<storage, read> ln_p: array<f32>;       // [4] hpost
@group(0) @binding(4) var<storage, read> ln_y: array<f32>;       // [512]
@compute @workgroup_size(64)
fn lanes_new(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= 2048u) { return; }
  let lane = i / 512u;
  let k = i % 512u;
  let base = lane * 4u;
  ln_n[i] = ln_r[base] * ln_l[k] + ln_r[base + 1u] * ln_l[512u + k]
    + ln_r[base + 2u] * ln_l[1024u + k] + ln_r[base + 3u] * ln_l[1536u + k]
    + ln_p[lane] * ln_y[k];
}

// lanes = [x0,x0,x0,x0]
@group(0) @binding(0) var<storage, read_write> il_l: array<f32>; // [2048]
@group(0) @binding(1) var<storage, read> il_x: array<f32>;       // [512]
@compute @workgroup_size(64)
fn init_lanes(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= 512u) { return; }
  let v = il_x[i];
  il_l[i] = v;
  il_l[512u + i] = v;
  il_l[1024u + i] = v;
  il_l[1536u + i] = v;
}

// lm[i] = mean de los 4 lanes
@group(0) @binding(0) var<storage, read_write> ml_o: array<f32>; // [512]
@group(0) @binding(1) var<storage, read> ml_l: array<f32>;       // [2048]
@compute @workgroup_size(64)
fn mean_lanes(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= 512u) { return; }
  ml_o[i] = 0.25 * (ml_l[i] + ml_l[512u + i] + ml_l[1024u + i] + ml_l[1536u + i]);
}

// ---------------------------------------------------------------- FWHT n=512
// butterfly normalizada (/sqrt2 por etapa, 9 etapas) en shared mem.
@group(0) @binding(0) var<storage, read_write> fw_b: array<f32>;
var<workgroup> fw_s: array<f32, 512>;
@compute @workgroup_size(256)
fn fwht(@builtin(local_invocation_id) li: vec3<u32>) {
  let t = li.x;
  fw_s[t] = fw_b[t];
  fw_s[t + 256u] = fw_b[t + 256u];
  workgroupBarrier();
  var s = 0u;
  while (s < 9u) {
    let stride = 1u << s;
    let base = (t / stride) * (stride * 2u);
    let i0 = base + (t % stride);
    let i1 = i0 + stride;
    let a = fw_s[i0];
    let b = fw_s[i1];
    fw_s[i0] = (a + b) * 0.70710678;
    fw_s[i1] = (a - b) * 0.70710678;
    workgroupBarrier();
    s += 1u;
  }
  fw_b[t] = fw_s[t];
  fw_b[t + 256u] = fw_s[t + 256u];
}

// dst[i<d] = x[i]*dd[i], resto 0  (entrada al HadamardMLP)
@group(0) @binding(0) var<storage, read_write> hi_d: array<f32>; // [512]
@group(0) @binding(1) var<storage, read> hi_x: array<f32>;       // [512]
@group(0) @binding(2) var<storage, read> hi_c: array<f32>;       // d1 [512]
@compute @workgroup_size(64)
fn hada_init(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= 512u) { return; }
  hi_d[i] = hi_x[i] * hi_c[i];
}

// ---------------------------------------------------------------- Sinkhorn 4x4
// 20 iters log-domain sobre 16 elems, exp al final. 1 thread.
@group(0) @binding(0) var<storage, read_write> sk_m: array<f32>; // [16]
@compute @workgroup_size(1)
fn sinkhorn() {
  var m: array<f32, 16>;
  for (var i = 0u; i < 16u; i++) { m[i] = sk_m[i]; }
  var it = 0u;
  while (it < 20u) {
    // filas
    for (var r = 0u; r < 4u; r++) {
      var mx: f32 = -1e30;
      for (var c = 0u; c < 4u; c++) {
        if (m[r * 4u + c] > mx) { mx = m[r * 4u + c]; }
      }
      var s: f32 = 0.0;
      for (var c = 0u; c < 4u; c++) { s += exp(m[r * 4u + c] - mx); }
      let lse = mx + log(s);
      for (var c = 0u; c < 4u; c++) { m[r * 4u + c] -= lse; }
    }
    // columnas
    for (var c = 0u; c < 4u; c++) {
      var mx: f32 = -1e30;
      for (var r = 0u; r < 4u; r++) {
        if (m[r * 4u + c] > mx) { mx = m[r * 4u + c]; }
      }
      var s: f32 = 0.0;
      for (var r = 0u; r < 4u; r++) { s += exp(m[r * 4u + c] - mx); }
      let lse = mx + log(s);
      for (var r = 0u; r < 4u; r++) { m[r * 4u + c] -= lse; }
    }
    it += 1u;
  }
  for (var i = 0u; i < 16u; i++) { sk_m[i] = exp(m[i]); }
}

// ---------------------------------------------------------------- Engram
// alpha = sigmoid(dot(rms(u), rms(ek))/sqrt(512)) -> escalar en out[0]
@group(0) @binding(0) var<storage, read> rd_u: array<f32>;  // [512]
@group(0) @binding(1) var<storage, read> rd_e: array<f32>;  // [512] ek
@group(0) @binding(2) var<storage, read_write> rd_o: array<f32>; // [1]
var<workgroup> rd_acc: array<f32, 256>;
@compute @workgroup_size(256)
fn rms_dot(@builtin(local_invocation_id) li: vec3<u32>) {
  let t = li.x;
  var su: f32 = 0.0;
  var se: f32 = 0.0;
  var i = t;
  while (i < 512u) {
    let a = rd_u[i];
    let b = rd_e[i];
    su += a * a;
    se += b * b;
    i += 256u;
  }
  // reduce su y se por separado: dos pasadas con el mismo acc
  rd_acc[t] = su;
  workgroupBarrier();
  var stride = 128u;
  while (stride > 0u) {
    if (t < stride) { rd_acc[t] += rd_acc[t + stride]; }
    workgroupBarrier();
    stride = stride / 2u;
  }
  let nu = inverseSqrt(rd_acc[0u] / 512.0 + 0.000001);
  rd_acc[t] = se;
  workgroupBarrier();
  stride = 128u;
  while (stride > 0u) {
    if (t < stride) { rd_acc[t] += rd_acc[t + stride]; }
    workgroupBarrier();
    stride = stride / 2u;
  }
  let ne = inverseSqrt(rd_acc[0u] / 512.0 + 0.000001);
  // dot normalizado
  var dt: f32 = 0.0;
  i = t;
  while (i < 512u) {
    dt += (rd_u[i] * nu) * (rd_e[i] * ne);
    i += 256u;
  }
  rd_acc[t] = dt;
  workgroupBarrier();
  stride = 128u;
  while (stride > 0u) {
    if (t < stride) { rd_acc[t] += rd_acc[t + stride]; }
    workgroupBarrier();
    stride = stride / 2u;
  }
  if (t == 0u) {
    let z = rd_acc[0u] / 22.627417; // sqrt(512)
    rd_o[0u] = 1.0 / (1.0 + exp(-z));
  }
}

// ev[i] = sum_j taps[j,i]*vring[slot(pos-j*3), i], solo j*3<=pos
struct EvP { pos: u32, _p0: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<storage, read_write> ev_o: array<f32>; // [512]
@group(0) @binding(1) var<storage, read> ev_t: array<f32>;       // [2048] taps
@group(0) @binding(2) var<storage, read> ev_r: array<f32>;       // [6656] vring 13x512
@group(0) @binding(3) var<uniform> ev_p: EvP;
@compute @workgroup_size(64)
fn engram_conv(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= 512u) { return; }
  var acc: f32 = 0.0;
  var j = 0u;
  while (j < 4u) {
    if (j * 3u <= ev_p.pos) {
      let slot = (ev_p.pos - j * 3u) % 13u;
      acc += ev_t[j * 512u + i] * ev_r[slot * 512u + i];
    }
    j += 1u;
  }
  ev_o[i] = acc;
}
