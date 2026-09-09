//! Attention SDPA mono-head: out = softmax(Q·Kᵀ/√d)·V.
//! Q,K,V:(s,d) row-major. Empaquetadas en UN storage [q|k|v] con offsets
//! en el uniform (Adreno 618 tiene pocos slots de storage).
//! f16 real si el device lo soporta; SI NO, cae automático a f32.

const ATTN_F32: &str = r#"
struct P {
    s: u32, d: u32,
    k_off: u32, v_off: u32, pad0: u32, pad1: u32,
}
@group(0) @binding(0) var<storage, read> packed_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x;
    let dd = g.y;
    if (i >= p.s || dd >= p.d) { return; }
    let scale: f32 = inverseSqrt(f32(p.d));
    var mx: f32 = -3.4e38;
    for (var j: u32 = 0u; j < p.s; j = j + 1u) {
        var dot: f32 = 0.0;
        for (var t: u32 = 0u; t < p.d; t = t + 1u) {
            dot = dot + packed_in[i * p.d + t] * packed_in[p.k_off + j * p.d + t];
        }
        mx = max(mx, dot * scale);
    }
    var sum: f32 = 0.0;
    var accv: f32 = 0.0;
    for (var j: u32 = 0u; j < p.s; j = j + 1u) {
        var dot: f32 = 0.0;
        for (var t: u32 = 0u; t < p.d; t = t + 1u) {
            dot = dot + packed_in[i * p.d + t] * packed_in[p.k_off + j * p.d + t];
        }
        let e = exp(dot * scale - mx);
        sum = sum + e;
        accv = accv + e * packed_in[p.v_off + j * p.d + dd];
    }
    out[i * p.d + dd] = accv / sum;
}
"#;

const ATTN_F16: &str = r#"
enable shader-f16;
struct P {
    s: u32, d: u32,
    k_off: u32, v_off: u32, pad0: u32, pad1: u32,
}
@group(0) @binding(0) var<storage, read> packed_in: array<f16>;
@group(0) @binding(1) var<storage, read_write> out: array<f16>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x;
    let dd = g.y;
    if (i >= p.s || dd >= p.d) { return; }
    let scale: f32 = inverseSqrt(f32(p.d));
    var mx: f32 = -3.4e38;
    for (var j: u32 = 0u; j < p.s; j = j + 1u) {
        var dot: f32 = 0.0;
        for (var t: u32 = 0u; t < p.d; t = t + 1u) {
            let qv = f32(packed_in[i * p.d + t]);
            let kv = f32(packed_in[p.k_off + j * p.d + t]);
            dot = dot + qv * kv;
        }
        mx = max(mx, dot * scale);
    }
    var sum: f32 = 0.0;
    var accv: f32 = 0.0;
    for (var j: u32 = 0u; j < p.s; j = j + 1u) {
        var dot: f32 = 0.0;
        for (var t: u32 = 0u; t < p.d; t = t + 1u) {
            let qv = f32(packed_in[i * p.d + t]);
            let kv = f32(packed_in[p.k_off + j * p.d + t]);
            dot = dot + qv * kv;
        }
        let e = exp(dot * scale - mx);
        sum = sum + e;
        accv = accv + e * f32(packed_in[p.v_off + j * p.d + dd]);
    }
    out[i * p.d + dd] = f16(accv / sum);
}
"#;

/// Corre SDPA. `use_f16` es un DESEO: sin soporte del device se usa f32.
pub fn run(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    s: usize,
    d: usize,
    use_f16: bool,
) -> Result<(Vec<f32>, f64), String> {
    let need = s * d;
    if q.len() < need || k.len() < need || v.len() < need {
        return Err(format!(
            "tamaños: q {}, k {}, v {} (necesitan s*d={need})",
            q.len(),
            k.len(),
            v.len()
        ));
    }
    let (device, queue) = super::ctx()?;
    // BANDERA AUTOMÁTICA: f16 solo si el device lo soporta realmente.
    let use_f16 = use_f16 && super::has_f16();

    // Empaquetado [q | k | v] en un único storage.
    let mut packed: Vec<f32> = Vec::with_capacity(need * 3);
    packed.extend_from_slice(&q[..need]);
    packed.extend_from_slice(&k[..need]);
    packed.extend_from_slice(&v[..need]);

    let conv = |arr: &[f32]| -> Vec<u8> {
        if use_f16 {
            super::u16_bytes(&super::to_u16_bits(arr))
        } else {
            super::f32_bytes(arr)
        }
    };
    let elem = if use_f16 { 2usize } else { 4usize };

    let buf_in = super::storage_buf(&device, &conv(&packed), true);
    let buf_out = super::storage_buf(&device, &vec![0u8; need * elem], false);
    let ubo = super::uniform_buf(
        &device,
        &super::params_bytes(&[
            s as u32,
            d as u32,
            need as u32,     // offset de k
            (need * 2) as u32, // offset de v
            0,
            0,
        ]),
    );

    let code = if use_f16 { ATTN_F16 } else { ATTN_F32 };
    let (layout, pipeline) = super::build_pipeline(
        &device,
        code,
        &[
            (0, false, true),  // packed_in: var<storage, read>
            (1, false, false), // out: var<storage, read_write>
            (2, true, false),  // p: uniform
        ],
    )?;

    let start = std::time::Instant::now();
    super::dispatch(
        &device,
        &queue,
        &pipeline,
        &layout,
        vec![
            (0, buf_in.as_entire_binding()),
            (1, buf_out.as_entire_binding()),
            (2, ubo.as_entire_binding()),
        ],
        (((s as u32) + 7) / 8, ((d as u32) + 7) / 8, 1),
    );
    let raw = super::read_back(&device, &queue, &buf_out, (need * elem) as u64)?;
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    Ok((super::bytes_to_f32(&raw, use_f16), ms))
}
