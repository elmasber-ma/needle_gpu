//! Shader Lab: corre WGSL arbitrario "en caliente" estilo celda de Colab.
//!
//! Contrato del shader del usuario:
//!   - binding 0: `var<storage, read_write> data: array<f32>` (in-place)
//!   - binding 1: `var<uniform> params: Params` con struct {x,y,z: f32, n: u32}
//!   - entry point: `fn main(@builtin(global_invocation_id) ...)`
//! El host sube el input en `data`, hace dispatch y lee de vuelta.

pub const LAB_TEMPLATE: &str = r#"
// CONTRATO del Shader Lab:
//   data   : buffer entrada/salida IN-PLACE (f32)
//   params : {x,y,z} libres + n = cantidad de elementos
// Con 'enable shader-f16;' usas f16 real (el device ya tiene el feature).
struct Params { x: f32, y: f32, z: f32, n: u32 }
@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.n) { return; }
    data[gid.x] = data[gid.x] * params.x + params.y;
}
"#;

/// Corre un shader arbitrario. Retorna (datos, ms, modo) o error del compilador.
///
/// AUTOMÁTICO: si el código pide `enable shader-f16` se normaliza a F32 al
/// vuelo (el contrato del lab sube/lee buffers f32, así que es el camino
/// correcto en CUALQUIER chip): nunca más pedimos deshabilitar nada.
pub fn run(
    code: &str,
    input: &[f32],
    workgroups: [u32; 3],
    px: f32,
    py: f32,
    pz: f32,
) -> Result<(Vec<f32>, f64, &'static str), String> {
    let mut code = code.to_string();
    let mut modo = "F32";
    if code.contains("shader-f16") {
        code = code
            .lines()
            .filter(|l| !l.trim_start().starts_with("enable shader-f16"))
            .collect::<Vec<_>>()
            .join("\n");
        code = code.replace("<f16>", "<f32>").replace("f16(", "f32(");
        modo = "AUTO F32";
    }
    let (device, queue) = super::ctx()?;

    let bytes = super::f32_bytes(input);
    let buffer = super::storage_buf(&device, &bytes, false);

    // params: x,y,z como bits f32 + n elementos
    let mut pb = Vec::with_capacity(16);
    pb.extend_from_slice(&px.to_bits().to_ne_bytes());
    pb.extend_from_slice(&py.to_bits().to_ne_bytes());
    pb.extend_from_slice(&pz.to_bits().to_ne_bytes());
    pb.extend_from_slice(&(input.len() as u32).to_ne_bytes());
    let ubo = super::uniform_buf(&device, &pb);

    let (layout, pipeline) =
        super::build_pipeline(&device, &code, &[(0, false, false), (1, true, false)])?;

    let start = std::time::Instant::now();
    super::dispatch(
        &device,
        &queue,
        &pipeline,
        &layout,
        vec![
            (0, buffer.as_entire_binding()),
            (1, ubo.as_entire_binding()),
        ],
        (workgroups[0], workgroups[1], workgroups[2]),
    );
    let raw = super::read_back(&device, &queue, &buffer, bytes.len() as u64)?;
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    Ok((super::bytes_to_f32(&raw, false), ms, modo))
}
