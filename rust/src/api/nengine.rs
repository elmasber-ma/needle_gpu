//! Needle v2 en WebGPU (fase 1: forward f32 en GPU).
//!
//! Misma superficie que needle CPU pero con el forward en shaders WGSL.
//! Alcance fase 1: greedy/temperatura, sin decoding restringido
//! (constrain) y sin head de confianza. El prompt se arma idéntico
//! (V2Engine::build_prompt) y el parseo de tags es el mismo.

use super::needle::NeedleOut;
use crate::frb_generated::StreamSink;
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

/// Paridad CPU vs GPU en 8 tokens greedy (detecta divergencia numérica).
/// Orquesta un motor CPU temporal + el GPU residente: vive en el api,
/// no dentro del módulo GPU.
#[flutter_rust_bridge::frb(sync)]
pub fn needle_gpu_parity(query: String) -> Result<String, String> {
    use needle_infer::v2_engine::{GenerateOptions, V2Engine};
    let cact_path = nengine::cact_path()?;
    let cpu = V2Engine::load(&cact_path).map_err(|e| format!("cpu load: {e}"))?;
    let opts = GenerateOptions {
        max_new_tokens: 8,
        temperature: 0.0,
        seed: 0,
        system: None,
        prefill_chunk: 128,
        constrain: false,
    };
    let rc = cpu.generate(&query, "[]", &opts, |_, _| {});
    let rg = nengine::generate_inner(&query, "[]", 8, 0.0, 0, &mut |_| {})?;
    let n = rc.token_ids.len().min(rg.token_ids.len()).min(8);
    let mut dif = None;
    for k in 0..n {
        if rc.token_ids[k] != rg.token_ids[k] {
            dif = Some(k);
            break;
        }
    }
    let show = |v: &[u32]| {
        v.iter()
            .take(8)
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(",")
    };
    let gaps = rg
        .margins
        .iter()
        .map(|(id, g)| format!("{id}:{g:.3}"))
        .collect::<Vec<_>>()
        .join(" ");
    match dif {
        None if rc.token_ids.len() == rg.token_ids.len() => Ok(format!(
            "MATCH 8/8 · cpu=[{}] gpu=[{}] · gaps=[{}]",
            show(&rc.token_ids),
            show(&rg.token_ids),
            gaps
        )),
        _ => Ok(format!(
            "DIF@{} · cpu=[{}] gpu=[{}] · gaps=[{}] (gap<0.05 en el flip = ruido; gap grande = bug)",
            dif.map(|k| k.to_string()).unwrap_or("len".into()),
            show(&rc.token_ids),
            show(&rg.token_ids),
            gaps
        )),
    }
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
    sink: StreamSink<String>,
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
