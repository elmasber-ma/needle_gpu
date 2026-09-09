//! GELU tanh-approx elementwise, f32 y f16 real.

const GELU_F32: &str = r#"
@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= arrayLength(&data)) { return; }
    let x = data[gid.x];
    let k: f32 = 0.7978845608028654;
    data[gid.x] = select(0.5 * x * (1.0 + tanh(k * (x + 0.044715 * x * x * x))), x, x > 10.0);
}
"#;

const GELU_F16: &str = r#"
enable shader-f16;
@group(0) @binding(0) var<storage, read_write> data: array<f16>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= arrayLength(&data)) { return; }
    let xf = f32(data[gid.x]);
    let k: f32 = 0.7978845608028654;
    let y = select(0.5 * xf * (1.0 + tanh(k * (xf + 0.044715 * xf * xf * xf))), xf, xf > 10.0);
    data[gid.x] = f16(y);
}
"#;

/// Corre GELU sobre [input]; retorna (datos, ms). `use_f16` = kernel f16 real.
pub fn run(input: &[f32], use_f16: bool) -> Result<(Vec<f32>, f64), String> {
    // BANDERA AUTOMÁTICA: f16 solo si el device lo soporta realmente.
    let use_f16 = use_f16 && super::has_f16();
    let (device, queue) = super::ctx()?;
    let bytes = if use_f16 {
        super::u16_bytes(&super::to_u16_bits(input))
    } else {
        super::f32_bytes(input)
    };
    let buffer = super::storage_buf(&device, &bytes, false);
    let code = if use_f16 { GELU_F16 } else { GELU_F32 };
    let (layout, pipeline) =
        super::build_pipeline(&device, code, &[(0, false, false)])?;

    let start = std::time::Instant::now();
    super::dispatch(
        &device,
        &queue,
        &pipeline,
        &layout,
        vec![(0, buffer.as_entire_binding())],
        (((input.len() as u32) + 63) / 64, 1, 1),
    );
    let out = super::read_back(&device, &queue, &buffer, bytes.len() as u64)?;
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    Ok((super::bytes_to_f32(&out, use_f16), ms))
}
