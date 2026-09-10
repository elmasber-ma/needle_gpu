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

/// Paridad CPU vs GPU en 8 pasos greedy con logits comparados.
///
/// Corre el CPU (`V2Model::step`, referencia oficial) y la GPU sobre la
/// MISMA secuencia (prompt + 8 greedy del CPU) y compara los vectores de
/// logits completos: `dmax` = max|Δ| por paso. dmax chico = ruido fp32,
/// dmax grande = bug en el port.
#[flutter_rust_bridge::frb(sync)]
pub fn needle_gpu_parity(query: String) -> Result<String, String> {
    use needle_infer::v2_engine::V2Engine;
    use std::cmp::Ordering;
    let cact_path = nengine::cact_path()?;
    let cpu = V2Engine::load(&cact_path).map_err(|e| format!("cpu load: {e}"))?;
    let model = cpu.model();
    let vocab = model.cfg.vocab_size;
    let ids = nengine::prompt_ids(&query, "[]")?;
    if ids.is_empty() {
        return Err("prompt vacío".into());
    }
    let argmax = |lg: &[f32]| {
        lg.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0)
    };
    // CPU stepped greedy: prompt + 8 (generate_sequential == generate).
    // Se guardan TODOS los logits para comparar la trayectoria completa.
    let mut st = model.make_state();
    let mut seq = ids.clone();
    let mut clog: Vec<Vec<f32>> = Vec::new();
    let total = ids.len() + 8;
    for pos in 0..total {
        let tok = *seq.get(pos).ok_or("secuencia corta")?;
        let mut lg = vec![0.0f32; vocab];
        model
            .step(tok, &mut st, &mut lg)
            .map_err(|e| format!("cpu step {pos}: {e}"))?;
        if pos + 1 >= ids.len() && pos + 1 < total {
            seq.push(argmax(&lg));
        }
        clog.push(lg);
    }
    // GPU sobre la misma secuencia, trayectoria completa.
    let glog = nengine::gpu_logits_stepped(&seq)?;
    if glog.len() != clog.len() {
        return Err(format!("gpu devolvió {} logits", glog.len()));
    }
    let vmax = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };
    let p = ids.len();
    let mut pmax = 0.0f32;
    let mut pfirst = Vec::new();
    for k in 0..p {
        let m = vmax(&clog[k], &glog[k]);
        pmax = pmax.max(m);
        if k < 6 {
            pfirst.push(format!("{m:.3}"));
        }
    }
    let show = |v: &[u32]| {
        v.iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(",")
    };
    let ctok: Vec<u32> = clog[p..].iter().map(|lg| argmax(lg)).collect();
    let gtok: Vec<u32> = glog[p..].iter().map(|lg| argmax(lg)).collect();
    let mut dif = None;
    let mut dmax = Vec::with_capacity(8);
    for k in 0..8 {
        if ctok[k] != gtok[k] && dif.is_none() {
            dif = Some(k);
        }
        dmax.push(vmax(&clog[p + k], &glog[p + k]));
    }
    let ds = dmax
        .iter()
        .map(|m| format!("{m:.3}"))
        .collect::<Vec<_>>()
        .join(" ");
    match dif {
        None => Ok(format!(
            "MATCH 8/8 · cpu=[{}] gpu=[{}] · p0=[{}] pmax={pmax:.3} dmax=[{}] (<0.5 = ruido)",
            show(&ctok),
            show(&gtok),
            pfirst.join(" "),
            ds
        )),
        Some(k) => Ok(format!(
            "DIF@{k} · cpu=[{}] gpu=[{}] · p0=[{}] pmax={pmax:.3} dmax=[{}] (dmax<0.5 = ruido; grande = bug)",
            show(&ctok),
            show(&gtok),
            pfirst.join(" "),
            ds
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
