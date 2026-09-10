//! Motor Needle v2 en WebGPU (fase 1: todo f32).
//!
//! Reusa needle-infer para contenedor/tokenizer/prompt/sampling-idea y
//! ejecuta el forward en WGSL. Paridad objetivo: `generate_sequential`
//! (prefill token por token, decode con KV ring de 256).
//!
//! Lo que queda en CPU: tokenize, hash/fetch Engram, sampling,
//! parseo de tags. Lo que va en GPU: todo el forward por token.

use std::sync::Mutex;

use needle_infer::cact::{Cact, DT_CQ};
use needle_infer::sp_tokenizer::SpTokenizer;
use needle_infer::v2_engine::V2Engine;

pub(crate) mod diag;
pub(crate) mod help;
pub(crate) mod pipes;
pub(crate) mod step;
pub(crate) mod types;
pub(crate) mod weight;

use diag::{sweep_all, sweep_kb};
use help::{at, f32s, sbuf, sbuf_zero, sl};
use pipes::build_all;
use step::step_token;
use types::{Act, EmbSrc, Engine, LayerBuf, SiteBuf};
use weight::{get_f32, load_wmat};

static ENG: Mutex<Option<Engine>> = Mutex::new(None);

// ------------------------------------------------------------------ load

pub async fn load(path: &str) -> Result<String, String> {
    let cact = Cact::load(std::path::Path::new(path))
        .map_err(|e| format!("no pude abrir {path}: {e}"))?;
    let lay = cact
        .layout()
        .map_err(|e| format!("layout inválido: {e}"))?;
    let g = &cact.geom;
    let (d, attn, kv, n_layers, vocab) = (
        g.d_model,
        g.attn_dim(),
        g.kv_dim(),
        g.num_layers,
        g.vocab_size,
    );
    if lay.layers.len() != n_layers {
        return Err("capas del layout != geometría".into());
    }

    let tok = SpTokenizer::from_blob(
        &cact
            .raw_tensor(lay.tokenizer.ok_or("sin tokenizer")?)
            .map_err(|e| format!("tokenizer: {e}"))?,
    )
    .map_err(|e| format!("tokenizer: {e}"))?;
    let eos_id = tok.eos_id;

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .map_err(|_| "sin adapter GPU".to_string())?;
    let aname = adapter.get_info().name.clone();
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("nengine"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        })
        .await
        .map_err(|_| "sin device GPU".to_string())?;

    let (pipes, layouts) = build_all(&device)?;
    let up = |v: Vec<f32>| sbuf(&device, &f32s(&v));

    // bytes crudos del .cact para subir CQ empaquetado tal cual
    let raw = std::fs::read(path).map_err(|e| format!("releer {path}: {e}"))?;
    let codebook = cact.codebook.clone();
    let wm = |idx: usize, o: usize, i: usize, rs: usize, r: usize, et: &str| {
        load_wmat(&device, &cact, &raw, &codebook, idx, o, i, rs, r, et)
    };

    // embedding: Wmat a GPU + fuente de fila en CPU (CQ → dequant 1 fila)
    let emb_rec = cact.record(lay.embedding);
    if emb_rec.ndim != 2 || emb_rec.shape.len() < 2 {
        return Err("embedding no 2-D".into());
    }
    let emb = wm(lay.embedding, vocab, d, 0, vocab, "emb")?;
    let emb_cpu = if emb_rec.dtype == DT_CQ {
        let w = cact
            .cq(lay.embedding)
            .map_err(|e| format!("cq emb: {e}"))?;
        if w.in_feat != d {
            return Err("embedding CQ con in != d".into());
        }
        EmbSrc::Cq(w)
    } else {
        EmbSrc::F32(get_f32(&cact, lay.embedding)?)
    };

    // tabla RoPE cos/sin intercalada [t*64 + 2i]
    let max_seq = g.max_seq_len;
    let mut tab = vec![0.0f32; max_seq * 64];
    for t in 0..max_seq {
        for i in 0..32 {
            let ang = t as f32 / g.rope_theta.powf(2.0 * i as f32 / 64.0);
            tab[t * 64 + 2 * i] = ang.cos();
            tab[t * 64 + 2 * i + 1] = ang.sin();
        }
    }
    let rope = up(tab);

    // capas
    let kv_window = g.kv_window;
    let mut layers = Vec::with_capacity(n_layers);
    for (li, l) in lay.layers.iter().enumerate() {
        let active = li % 4;
        let mut pre_off = vec![0.0f32; 4];
        let mut post_off = vec![0.0f32; 4];
        for lane in 0..4 {
            pre_off[lane] = if lane == active { 4.0 } else { -4.0 };
            post_off[lane] = if lane == active { 0.0 } else { -4.0 };
        }
        let gw = |i: usize| get_f32(&cact, i);
        let a_pre_v = gw(lay.mhc.a_pre)?;
        let a_post_v = gw(lay.mhc.a_post)?;
        let a_res_v = gw(lay.mhc.a_res)?;
        let agate_v = gw(l.attn_gate)?;
        let a_pre = at(&a_pre_v, li, "a_pre")?;
        let a_post = at(&a_post_v, li, "a_post")?;
        let a_res = at(&a_res_v, li, "a_res")?;
        let agate = at(&agate_v, 0, "attn_gate")?;
        let b_pre_v = gw(lay.mhc.b_pre)?;
        let b_post_v = gw(lay.mhc.b_post)?;
        let b_res_v = gw(lay.mhc.b_res)?;
        let phi_pre_v = gw(lay.mhc.phi_pre)?;
        let phi_post_v = gw(lay.mhc.phi_post)?;
        let phi_res_v = gw(lay.mhc.phi_res)?;
        layers.push(LayerBuf {
            q: wm(l.q_proj, attn, d, 0, attn, "q")?,
            k: wm(l.k_proj, kv, d, 0, kv, "k")?,
            v: wm(l.v_proj, kv, d, 0, kv, "v")?,
            g: wm(l.gate_proj, attn, d, 0, attn, "g")?,
            o: wm(l.out_proj, d, attn, 0, d, "o")?,
            qn: up(gw(l.q_norm)?),
            kn: up(gw(l.k_norm)?),
            nin: up(gw(l.norm_in)?),
            pnorm: up(gw(l.post_norm)?),
            phada: up(gw(l.pre_hada)?),
            d1: up(gw(l.d1)?),
            d2: up(gw(l.d2)?),
            d3: up(gw(l.d3)?),
            b_pre: up(sl(&b_pre_v, li * 4, li * 4 + 4, "b_pre")?),
            b_post: up(sl(&b_post_v, li * 4, li * 4 + 4, "b_post")?),
            b_res: up(sl(&b_res_v, li * 16, li * 16 + 16, "b_res")?),
            phi_pre: wm(
                lay.mhc.phi_pre,
                n_layers * 4,
                2048,
                li * 4,
                4,
                "phi_pre",
            )?,
            phi_post: wm(
                lay.mhc.phi_post,
                n_layers * 4,
                2048,
                li * 4,
                4,
                "phi_post",
            )?,
            phi_res: wm(
                lay.mhc.phi_res,
                n_layers * 16,
                2048,
                li * 16,
                16,
                "phi_res",
            )?,
            pre_off: up(pre_off),
            post_off: up(post_off),
            kv_k: sbuf_zero(&device, kv_window * kv),
            kv_v: sbuf_zero(&device, kv_window * kv),
            a_pre,
            a_post,
            a_res,
            agate,
        });
    }

    // engrams: tablas a CPU (gather), projs a GPU (mismo peso), taps f32
    let mut sites = Vec::new();
    let mut tables_cpu = Vec::new();
    for s in lay.engrams.iter() {
        tables_cpu.push(get_f32(&cact, s.tables)?);
        let ring = g.engram_conv_taps * g.engram_conv_dilation + 1;
        sites.push(SiteBuf {
            key: wm(s.key_proj, d, d, 0, d, "ekey")?,
            value: wm(s.value_proj, d, d, 0, d, "evalue")?,
            taps: up(get_f32(&cact, s.taps)?),
            vring: sbuf_zero(&device, ring * d),
        });
    }

    let act = Act {
        h: sbuf_zero(&device, d),
        q: sbuf_zero(&device, attn),
        k: sbuf_zero(&device, kv),
        v: sbuf_zero(&device, kv),
        gt: sbuf_zero(&device, attn),
        o: sbuf_zero(&device, attn),
        ar: sbuf_zero(&device, d),
        y: sbuf_zero(&device, d),
        h2: sbuf_zero(&device, d),
        m: sbuf_zero(&device, 512),
        s4: sbuf_zero(&device, 4),
        s4b: sbuf_zero(&device, 4),
        s16: sbuf_zero(&device, 16),
        t16: sbuf_zero(&device, 16),
        nx: sbuf_zero(&device, 2048),
        lanes_a: sbuf_zero(&device, 2048),
        lanes_b: sbuf_zero(&device, 2048),
        u: sbuf_zero(&device, d),
        bx: sbuf_zero(&device, d),
        e: sbuf_zero(&device, 512),
        ek: sbuf_zero(&device, d),
        ev: sbuf_zero(&device, d),
        lm: sbuf_zero(&device, d),
        logits: sbuf_zero(&device, vocab),
        alpha1: sbuf_zero(&device, 1),
        xh_h: sbuf_zero(&device, d),
        xh_o: sbuf_zero(&device, d),
        xh_e: sbuf_zero(&device, d),
        xh_nx: sbuf_zero(&device, 2048),
        xh_lm: sbuf_zero(&device, d),
    };
    let fnorm = up(get_f32(&cact, lay.final_norm)?);

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("logits_rb"),
        size: (vocab * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Dummy para el slot STORAGE de gamma cuando no hay (tiene que ser
    // STORAGE aunque el kernel no lo lea: si no, validación).
    let dummy1 = sbuf(&device, &[0u8; 4]);
    let eng = Engine {
        cact_path: path.to_string(),
        device,
        queue,
        pipes,
        layouts,
        layers,
        sites,
        act,
        emb,
        rope,
        dummy1,
        fnorm,
        staging,
        tok,
        emb_cpu,
        tables_cpu,
        hist: Vec::new(),
        n_layers,
        d,
        attn,
        kv,
        vocab,
        kv_window,
        max_seq,
        rope_theta: g.rope_theta,
        eos_id,
        engram_sites: g.engram_sites.clone(),
        engram_orders: g.engram_orders.clone(),
        engram_taps: g.engram_conv_taps,
        engram_dil: g.engram_conv_dilation,
        engram_slots: g.engram_slots,
        vring_n: g.engram_conv_taps * g.engram_conv_dilation + 1,
    };

    let info = format!("nengine GPU listo en {aname} · {n_layers} capas");
    *ENG.lock().map_err(|_| "mutex envenenado".to_string())? = Some(eng);
    Ok(info)
}

// ------------------------------------------------------------------ sampling

struct SplitMix(u64);
impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        ((self.next() >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

fn sample(logits: &[f32], temperature: f32, rng: &mut SplitMix) -> u32 {
    if temperature <= 0.0 {
        let mut best = 0u32;
        let mut bv = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > bv {
                bv = v;
                best = i as u32;
            }
        }
        return best;
    }
    let t = temperature.max(1e-6);
    let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut acc = 0.0f64;
    let mut cum = Vec::with_capacity(logits.len());
    for &v in logits {
        acc += (((v - m) / t) as f64).exp();
        cum.push(acc);
    }
    let r = rng.unit() * acc;
    cum.iter()
        .position(|&c| c >= r)
        .map(|i| i as u32)
        .unwrap_or(logits.len() as u32 - 1)
}

// ------------------------------------------------------------------ generate

fn split_tag(s: &str, open: &str, close: &str) -> Option<String> {
    let a = s.find(open)? + open.len();
    let b = s[a..].find(close)? + a;
    Some(s[a..b].to_string())
}

fn strip_tag(s: &str, open: &str, close: &str) -> String {
    if let Some(a) = s.find(open) {
        if let Some(rel) = s[a..].find(close) {
            let b = a + rel + close.len();
            return format!("{}{}", &s[..a], &s[b..]);
        }
    }
    s.to_string()
}

pub struct GenOut {
    pub text: String,
    pub tool_call: Option<String>,
    pub thinking: Option<String>,
    pub stop: String,
    pub prompt_tokens: u32,
    pub generated_tokens: u32,
    pub token_ids: Vec<u32>,
    /// (top1 id, gap top1-top2) de los primeros pasos (diagnóstico).
    pub margins: Vec<(u32, f32)>,
}

fn top2(logits: &[f32]) -> (u32, f32, u32, f32) {
    let mut b1 = 0u32;
    let mut v1 = f32::NEG_INFINITY;
    let mut b2 = 0u32;
    let mut v2 = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > v1 {
            v2 = v1;
            b2 = b1;
            v1 = v;
            b1 = i as u32;
        } else if v > v2 {
            v2 = v;
            b2 = i as u32;
        }
    }
    (b1, v1, b2, v2)
}

/// Un solo paso cronometrado (diagnóstico: cuelga vs lento).
/// Devuelve "step_ms=.. · vocab=.. · capas=..". No toca el historial.
pub fn diag_step() -> Result<String, String> {
    let g = ENG
        .lock()
        .map_err(|_| "mutex envenenado".to_string())?;
    let eng = g.as_ref().ok_or("motor GPU no cargado")?;
    let t0 = std::time::Instant::now();
    let logits = step_token(eng, 2, 0)?;
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    let top = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    let step_s = format!(
        "step_ms={ms:.1} · vocab={} · capas={} · top={top}",
        eng.vocab, eng.n_layers
    );
    let sw = sweep_all(eng)?;
    let kb = sweep_kb(eng)?;
    Ok(format!("{step_s} · {sw} · {kb}"))
}

/// IDs del prompt (diagnóstico de paridad): mismo prompt que generate.
pub fn prompt_ids(query: &str, tools_json: &str) -> Result<Vec<u32>, String> {
    let g = ENG
        .lock()
        .map_err(|_| "mutex envenenado".to_string())?;
    let eng = g.as_ref().ok_or("motor GPU no cargado")?;
    Ok(eng
        .tok
        .encode(&V2Engine::build_prompt(query, tools_json, None)))
}

/// Logits GPU paso a paso sobre `ids` (resetea el estado primero).
/// Devuelve los logits de TODOS los pasos (prefill + decode).
pub fn gpu_logits_stepped(ids: &[u32]) -> Result<Vec<Vec<f32>>, String> {
    let mut g = ENG
        .lock()
        .map_err(|_| "mutex envenenado".to_string())?;
    let eng = g.as_mut().ok_or("motor GPU no cargado")?;
    // reset total: cachés KV, anillos engram, lanes e historial
    eng.hist.clear();
    let z_kv = vec![0u8; eng.kv_window * eng.kv * 4];
    for l in &eng.layers {
        eng.queue.write_buffer(&l.kv_k, 0, &z_kv);
        eng.queue.write_buffer(&l.kv_v, 0, &z_kv);
    }
    let z_vr = vec![0u8; eng.vring_n * eng.d * 4];
    for s in &eng.sites {
        eng.queue.write_buffer(&s.vring, 0, &z_vr);
    }
    let z_l = vec![0u8; 2048 * 4];
    eng.queue.write_buffer(&eng.act.lanes_a, 0, &z_l);
    eng.queue.write_buffer(&eng.act.lanes_b, 0, &z_l);
    eng.hist.extend_from_slice(ids);
    let mut out = Vec::with_capacity(ids.len());
    for (pos, &tok) in ids.iter().enumerate() {
        if pos >= eng.max_seq {
            return Err("contexto lleno".into());
        }
        out.push(step_token(eng, tok, pos)?);
    }
    Ok(out)
}

pub fn generate(
    query: &str,
    tools_json: &str,
    max_new_tokens: u32,
    temperature: f32,
    seed: u64,
) -> Result<GenOut, String> {
    let mut sink = |_: String| {};
    generate_inner(query, tools_json, max_new_tokens, temperature, seed, &mut sink)
}

pub fn generate_stream(
    query: &str,
    tools_json: &str,
    max_new_tokens: u32,
    temperature: f32,
    seed: u64,
    emit: &mut dyn FnMut(String),
) -> Result<GenOut, String> {
    generate_inner(query, tools_json, max_new_tokens, temperature, seed, emit)
}

pub(crate) fn generate_inner(
    query: &str,
    tools_json: &str,
    max_new_tokens: u32,
    temperature: f32,
    seed: u64,
    emit: &mut dyn FnMut(String),
) -> Result<GenOut, String> {
    let mut g = ENG
        .lock()
        .map_err(|_| "mutex envenenado".to_string())?;
    let eng = g.as_mut().ok_or("motor GPU no cargado")?;

    let prompt = V2Engine::build_prompt(query, tools_json, None);
    let ids = eng.tok.encode(&prompt);
    let max_new = max_new_tokens.clamp(8, 512) as usize;
    eng.hist.clear();
    eng.hist.extend_from_slice(&ids);

    // Prefill de todos MENOS el último (el último lo corre el loop de
    // decode para aprovechar sus logits; si no, el token se duplicaría).
    if ids.is_empty() {
        return Err("prompt vacío".into());
    }
    let mut pos = 0usize;
    for &id in ids.iter().take(ids.len() - 1) {
        if pos >= eng.max_seq {
            return Err("prompt más largo que max_seq".into());
        }
        step_token(eng, id, pos)?;
        pos += 1;
    }

    let mut gen: Vec<u32> = Vec::new();
    let mut margins: Vec<(u32, f32)> = Vec::new();
    let mut rng = SplitMix(seed);
    let mut stop = "max_tokens";
    while gen.len() < max_new {
        if pos >= eng.max_seq {
            stop = "contexto lleno";
            break;
        }
        let feed = *gen.last().or(ids.last()).unwrap_or(&0);
        let logits = step_token(eng, feed, pos)?;
        pos += 1;
        let id = sample(&logits, temperature, &mut rng);
        gen.push(id);
        eng.hist.push(id);
        if margins.len() < 8 {
            let (t1, v1, _, v2) = top2(&logits);
            let _ = t1;
            margins.push((id, v1 - v2));
        }
        emit(eng.tok.decode(&[id]));
        if id == eng.eos_id {
            stop = "eos";
            break;
        }
        let txt = eng.tok.decode(&gen);
        if txt.contains("<|im_end|>") {
            stop = "im_end";
            break;
        }
    }

    let mut full = eng.tok.decode(&gen);
    let tool_call = split_tag(&full, "<tool_call>", "</tool_call>");
    let thinking = split_tag(&full, "<think>", "</think>");
    full = strip_tag(&full, "<tool_call>", "</tool_call>");
    full = strip_tag(&full, "<think>", "</think>");
    if stop == "im_end" {
        if let Some(p) = full.find("<|im_end|>") {
            full.truncate(p);
        }
    }
    Ok(GenOut {
        text: full.trim().to_string(),
        tool_call: tool_call.map(|s| s.trim().to_string()),
        thinking: thinking.map(|s| s.trim().to_string()),
        stop: stop.to_string(),
        prompt_tokens: ids.len() as u32,
        generated_tokens: gen.len() as u32,
        token_ids: gen,
        margins,
    })
}



pub fn unload() {
    if let Ok(mut g) = ENG.lock() {
        *g = None;
    }
}

/// Path del .cact cargado (para la paridad del api sin exponer el motor).
pub(crate) fn cact_path() -> Result<String, String> {
    let g = ENG.lock().map_err(|_| "mutex envenenado".to_string())?;
    g.as_ref()
        .ok_or_else(|| "motor GPU no cargado".to_string())
        .map(|e| e.cact_path.clone())
}

pub fn is_loaded() -> bool {
    matches!(ENG.lock(), Ok(g) if g.is_some())
}
