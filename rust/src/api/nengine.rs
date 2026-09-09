//! Needle v2 en WebGPU (fase 1: forward f32 en GPU).
//!
//! Misma superficie que needle CPU pero con el forward en shaders WGSL.
//! Alcance fase 1: greedy/temperatura, sin decoding restringido
//! (constrain) y sin head de confianza. El prompt se arma idéntico
//! (V2Engine::build_prompt) y el parseo de tags es el mismo.

use super::needle::NeedleOut;
use crate::nengine;

/// Carga el .cact, dequantiza a f32 y sube todo a la GPU (~180 MB).
/// Lento una sola vez; después cada token va solo.
#[flutter_rust_bridge::frb]
pub async fn needle_gpu_load(path: String) -> Result<String, String> {
    nengine::load(&path).await
}

/// Libera el motor GPU.
#[flutter_rust_bridge::frb]
pub fn needle_gpu_unload() {
    nengine::unload()
}

#[flutter_rust_bridge::frb]
pub fn needle_gpu_is_loaded() -> bool {
    nengine::is_loaded()
}

/// Diagnóstico: un solo paso cronometrado (cuelgue vs lentitud).
#[flutter_rust_bridge::frb(sync)]
pub fn needle_gpu_diag() -> Result<String, String> {
    nengine::diag_step()
}

/// Genera con el forward en GPU, emitiendo cada pieza en vivo.
/// Sin constrain en fase 1.
#[flutter_rust_bridge::frb]
pub fn needle_gpu_run_stream(
    query: String,
    tools_json: String,
    max_new_tokens: u32,
    temperature: f32,
    seed: u64,
    sink: flutter_rust_bridge::StreamSink<String>,
) -> Result<(), String> {
    nengine::generate_stream(&query, &tools_json, max_new_tokens, temperature, seed, &mut |piece| {
        let _ = sink.add(piece);
    })?;
    Ok(())
}

/// Genera con el forward en GPU. Sin constrain en fase 1.
#[flutter_rust_bridge::frb(sync)]
pub fn needle_gpu_run(
    query: String,
    tools_json: String,
    max_new_tokens: u32,
    temperature: f32,
    seed: u64,
) -> Result<NeedleOut, String> {
    let r = nengine::generate(&query, &tools_json, max_new_tokens, temperature, seed)?;
    Ok(NeedleOut {
        text: r.text,
        tool_call: r.tool_call,
        thinking: r.thinking,
        stop: r.stop,
        prompt_tokens: r.prompt_tokens,
        generated_tokens: r.generated_tokens,
    })
}
