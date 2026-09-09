/// Needle v2 (Cactus Compute) — transformer chico de tool-calling que corre
/// 100% en el dispositivo: mapea (query, lista de tools JSON) → llamada
/// JSON a una de esas tools. Runtime needle-infer, Rust puro sin candle/burn.
///
/// El motor vive en un Mutex global: se carga una vez con el path del
/// archivo .cact (~13.7 MB) y queda residente (~23 MB de trabajo).
use std::path::Path;
use std::sync::Mutex;

use needle_infer::v2_engine::{GenerateOptions, StopReason, V2Engine};

static ENGINE: Mutex<Option<V2Engine>> = Mutex::new(None);

fn take_engine() -> Result<std::sync::MutexGuard<'static, Option<V2Engine>>, String> {
    ENGINE.lock().map_err(|_| "mutex del motor envenenado".to_string())
}

/// Carga (o recarga) el modelo desde un .cact local.
#[flutter_rust_bridge::frb]
pub fn needle_load(path: String) -> Result<String, String> {
    let engine = V2Engine::load(Path::new(&path))
        .map_err(|e| format!("no pude cargar {path}: {e}"))?;
    let mut g = take_engine()?;
    *g = Some(engine);
    Ok("modelo listo".into())
}

/// Libera la memoria del motor.
#[flutter_rust_bridge::frb]
pub fn needle_unload() {
    if let Ok(mut g) = ENGINE.lock() {
        *g = None;
    }
}

#[flutter_rust_bridge::frb]
pub fn needle_is_loaded() -> bool {
    matches!(ENGINE.lock(), Ok(g) if g.is_some())
}

/// Resultado de una generación.
pub struct NeedleOut {
    pub text: String,
    /// Payload entre <tool_call></tool_call> cuando el modelo eligió una tool.
    pub tool_call: Option<String>,
    /// Razonamiento <think></think> si lo hubo.
    pub thinking: Option<String>,
    /// Por qué cortó: eos | im_end | max_tokens | contexto lleno | error.
    pub stop: String,
    pub prompt_tokens: u32,
    /// Tokens generados (para tok/s en benchmark).
    pub generated_tokens: u32,
}

fn stop_str(sr: StopReason) -> &'static str {
    match sr {
        StopReason::Eos => "eos",
        StopReason::ImEnd => "im_end",
        StopReason::MaxTokens => "max_tokens",
        StopReason::ContextFull => "contexto lleno",
        StopReason::Error(e) => e,
    }
}

/// Corre el modelo: (query, tools_json) → tool call.
/// constrain=true activa decoding restringido al trie de tools declaradas.
#[flutter_rust_bridge::frb]
pub fn needle_run(
    query: String,
    tools_json: String,
    constrain: bool,
    max_new_tokens: u32,
    temperature: f32,
    seed: u64,
) -> Result<NeedleOut, String> {
    let g = take_engine()?;
    let eng = g.as_ref().ok_or("modelo no cargado")?;
    let opts = GenerateOptions {
        constrain,
        max_new_tokens: max_new_tokens.clamp(8, 512) as usize,
        temperature,
        seed,
        ..Default::default()
    };
    let r = eng.generate(&query, &tools_json, &opts, |_, _| {});
    Ok(NeedleOut {
        text: r.text,
        tool_call: r.tool_call,
        thinking: r.thinking,
        stop: stop_str(r.stop_reason).to_string(),
        prompt_tokens: r.prompt_tokens as u32,
        generated_tokens: r.token_ids.len() as u32,
    })
}

/// Confianza (0..1) del head propio para esa respuesta dada la query+tools.
#[flutter_rust_bridge::frb]
pub fn needle_confidence(
    query: String,
    tools_json: String,
    completion: String,
) -> Result<f64, String> {
    let g = take_engine()?;
    let eng = g.as_ref().ok_or("modelo no cargado")?;
    eng.confidence_for(&query, &tools_json, &completion)
        .map(|c| c as f64)
        .ok_or_else(|| "este contenedor no trae head de confianza".into())
}

