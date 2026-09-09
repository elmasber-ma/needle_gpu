//! Linear Y = X·Wᵀ + b (X:(m,k), W:(n,k) row-major, b:(n)).
//! UN solo storage de entrada empaquetado [a|w|b] con offsets en el
//! uniform: cabe en GPUs con pocos slots de storage (Adreno 618).
//! f16 real si el device lo soporta; SI NO, cae automático a f32.

const LINEAR_F32: &str = r#"
struct Params {
    m: u32, n: u32, k: u32,
    w_off: u32, b_off: u32, pad0: u32, pad1: u32, pad2: u32,
}
@group(0) @binding(0) var<storage, read> packed_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let row = g.x;
    let col = g.y;
    if (row >= p.m || col >= p.n) { return; }
    var acc: f32 = 0.0;
    for (var i: u32 = 0u; i < p.k; i = i + 1u) {
        let x = packed_in[row * p.k + i];
        let w = packed_in[p.w_off + col * p.k + i];
        acc = acc + x * w;
    }
    out[row * p.n + col] = acc + packed_in[p.b_off + col];
}
"#;

const LINEAR_F16: &str = r#"
enable shader-f16;
struct Params {
    m: u32, n: u32, k: u32,
    w_off: u32, b_off: u32, pad0: u32, pad1: u32, pad2: u32,
}
@group(0) @binding(0) var<storage, read> packed_in: array<f16>;
@group(0) @binding(1) var<storage, read_write> out: array<f16>;
@group(0) @binding(2) var<uniform> p: Params;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let row = g.x;
    let col = g.y;
    if (row >= p.m || col >= p.n) { return; }
    var acc: f32 = 0.0;
    for (var i: u32 = 0u; i < p.k; i = i + 1u) {
        let x = f32(packed_in[row * p.k + i]);
        let w = f32(packed_in[p.w_off + col * p.k + i]);
        acc = acc + x * w;
    }
    out[row * p.n + col] = f16(acc + f32(packed_in[p.b_off + col]));
}
"#;

/// Corre Linear. `use_f16` es un DESEO: sin soporte del device se usa f32.
pub fn run(
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    m: usize,
    n: usize,
    k: usize,
    use_f16: bool,
) -> Result<(Vec<f32>, f64), String> {
    if input.len() < m * k || weights.len() < n * k || bias.len() < n {
        return Err(format!(
            "tamaños: input {} (necesita m*k={}), weights {} (n*k={}), bias {} (n={})",
            input.len(), m * k, weights.len(), n * k, bias.len(), n
        ));
    }
    let (device, queue) = super::ctx()?;
    // BANDERA AUTOMÁTICA: f16 solo si el device lo soporta realmente.
    let use_f16 = use_f16 && super::has_f16();

    // Empaquetado [a | w | b] en un único storage.
    let mut packed: Vec<f32> = Vec::with_capacity(m * k + n * k + n);
    packed.extend_from_slice(&input[..m * k]);
    packed.extend_from_slice(&weights[..n * k]);
    packed.extend_from_slice(&bias[..n]);

    let conv = |v: &[f32]| -> Vec<u8> {
        if use_f16 {
            super::u16_bytes(&super::to_u16_bits(v))
        } else {
            super::f32_bytes(v)
        }
    };
    let out_len = m * n;
    let elem = if use_f16 { 2usize } else { 4usize };

    let buf_in = super::storage_buf(&device, &conv(&packed), true);
    let buf_out = super::storage_buf(&device, &vec![0u8; out_len * elem], false);
    let ubo = super::uniform_buf(
        &device,
        &super::params_bytes(&[
            m as u32,
            n as u32,
            k as u32,
            (m * k) as u32,       // offset de w
            (m * k + n * k) as u32, // offset de bias
            0,
            0,
            0,
        ]),
    );

    let code = if use_f16 { LINEAR_F16 } else { LINEAR_F32 };
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
        (((m as u32) + 7) / 8, ((n as u32) + 7) / 8, 1),
    );
    let raw = super::read_back(&device, &queue, &buf_out, (out_len * elem) as u64)?;
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    Ok((super::bytes_to_f32(&raw, use_f16), ms))
}
