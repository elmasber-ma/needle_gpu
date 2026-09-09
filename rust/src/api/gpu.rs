/// Binding FRB del módulo GPU (`crate::gpu`): compute shaders WGSL nativos.
///
/// Funciones libres sobre el singleton (el codegen expone las clases opacas
/// como abstractas, así que no usamos constructores de clases). Todo retorna
/// f32 hacia Dart; los kernels pueden correr f16 real por dentro.

use crate::gpu;

#[flutter_rust_bridge::frb]
pub struct GpuOpResult {
    pub data: Vec<f32>,
    pub elapsed_ms: f64,
}

#[flutter_rust_bridge::frb]
pub struct ShaderRunResult {
    pub ok: bool,
    pub error: String,
    /// Modo efectivo del Lab: "F32" o "AUTO F32" (normalizado en vuelo).
    pub mode: String,
    pub data: Vec<f32>,
    pub elapsed_ms: f64,
}

/// Inicializa adapter+device pidiendo SHADER_F16. false si no hay GPU.
#[flutter_rust_bridge::frb]
pub async fn gpu_init() -> bool {
    if gpu::ctx().is_ok() {
        return true;
    }
    gpu::init().await
}

/// "nombre | Backend | f16=true/false". Vacío si no init.
#[flutter_rust_bridge::frb]
pub fn gpu_info() -> String {
    gpu::info()
}

#[flutter_rust_bridge::frb]
pub fn gpu_has_f16() -> bool {
    gpu::has_f16()
}

/// Plantilla WGSL para precargar el Shader Lab.
#[flutter_rust_bridge::frb]
pub fn gpu_lab_template() -> String {
    gpu::lab::LAB_TEMPLATE.to_string()
}

/// GELU tanh-approx. useF16 = kernel con array<f16> real.
#[flutter_rust_bridge::frb]
pub fn gpu_gelu(input: Vec<f32>, use_f16: bool) -> Result<GpuOpResult, String> {
    let (data, ms) = gpu::gelu::run(&input, use_f16)?;
    Ok(GpuOpResult { data, elapsed_ms: ms })
}

#[flutter_rust_bridge::frb]
pub struct LinearDims {
    pub m: i64,
    pub n: i64,
    pub k: i64,
}

/// Linear Y = X·Wᵀ + b; W en (n,k) row-major. useF16 = buffers f16.
#[flutter_rust_bridge::frb]
pub fn gpu_linear(
    input: Vec<f32>,
    weights: Vec<f32>,
    bias: Vec<f32>,
    dims: LinearDims,
    use_f16: bool,
) -> Result<GpuOpResult, String> {
    let (data, ms) = gpu::linear::run(
        &input,
        &weights,
        &bias,
        dims.m.max(1) as usize,
        dims.n.max(1) as usize,
        dims.k.max(1) as usize,
        use_f16,
    )?;
    Ok(GpuOpResult { data, elapsed_ms: ms })
}

#[flutter_rust_bridge::frb]
pub struct AttentionDims {
    pub s: i64,
    pub d: i64,
}

/// Attention SDPA mono-head. useF16 = buffers f16.
#[flutter_rust_bridge::frb]
pub fn gpu_attention(
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    dims: AttentionDims,
    use_f16: bool,
) -> Result<GpuOpResult, String> {
    let (data, ms) = gpu::attention::run(
        &q,
        &k,
        &v,
        dims.s.max(1) as usize,
        dims.d.max(1) as usize,
        use_f16,
    )?;
    Ok(GpuOpResult { data, elapsed_ms: ms })
}

/// Corre un shader WGSL arbitrario en caliente (contrato del Shader Lab:
/// binding 0 = data in-place, binding 1 = uniform Params{x,y,z,n}).
#[flutter_rust_bridge::frb]
pub fn gpu_run_wgsl(
    code: String,
    input: Vec<f32>,
    dispatch_x: i64,
    dispatch_y: i64,
    dispatch_z: i64,
    param_x: f64,
    param_y: f64,
    param_z: f64,
) -> ShaderRunResult {
    match gpu::lab::run(
        &code,
        &input,
        [
            dispatch_x.max(1) as u32,
            dispatch_y.max(1) as u32,
            dispatch_z.max(1) as u32,
        ],
        param_x as f32,
        param_y as f32,
        param_z as f32,
    ) {
        Ok((data, ms, modo)) => ShaderRunResult {
            ok: true,
            error: String::new(),
            mode: modo.to_string(),
            data,
            elapsed_ms: ms,
        },
        Err(e) => ShaderRunResult {
            ok: false,
            error: e,
            mode: String::new(),
            data: vec![],
            elapsed_ms: 0.0,
        },
    }
}
