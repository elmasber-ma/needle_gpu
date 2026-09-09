//! Motor Needle v2 en WebGPU (fase 1: todo f32).
//!
//! Reusa needle-infer para contenedor/tokenizer/prompt/sampling-idea y
//! ejecuta el forward en WGSL. Paridad objetivo: `generate_sequential`
//! (prefill token por token, decode con KV ring de 256).
//!
//! Lo que queda en CPU: tokenize, hash/fetch Engram, sampling,
//! parseo de tags. Lo que va en GPU: todo el forward por token.

use std::collections::HashMap;
use std::sync::Mutex;

use needle_infer::cact::{Cact, DT_CQ};
use needle_infer::sp_tokenizer::SpTokenizer;
use needle_infer::v2_engine::V2Engine;
use wgpu::util::DeviceExt;

const SHADERS: &str = include_str!("shaders.wgsl");
const SQRT_D: f32 = 22.627417; // sqrt(512)
const INV_SQRT_HEAD: f32 = 0.125; // 1/sqrt(64)

fn u16f(a: u32, b: u32, c: u32, d: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity(16);
    for x in [a, b, c, d] {
        v.extend_from_slice(&x.to_le_bytes());
    }
    v
}

fn u16fb(s: f32, n: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity(16);
    v.extend_from_slice(&s.to_le_bytes());
    v.extend_from_slice(&n.to_le_bytes());
    v.extend_from_slice(&[0u8; 8]);
    v
}

fn f32s(v: &[f32]) -> Vec<u8> {
    let mut o = Vec::with_capacity(v.len() * 4);
    for x in v {
        o.extend_from_slice(&x.to_le_bytes());
    }
    o
}

// ------------------------------------------------------------------ estado

struct LayerBuf {
    q: wgpu::Buffer,
    k: wgpu::Buffer,
    v: wgpu::Buffer,
    g: wgpu::Buffer,
    o: wgpu::Buffer,
    qn: wgpu::Buffer,
    kn: wgpu::Buffer,
    nin: wgpu::Buffer,
    pnorm: wgpu::Buffer,
    phada: wgpu::Buffer,
    d1: wgpu::Buffer,
    d2: wgpu::Buffer,
    d3: wgpu::Buffer,
    b_pre: wgpu::Buffer,
    b_post: wgpu::Buffer,
    b_res: wgpu::Buffer,
    phi_pre: wgpu::Buffer,
    phi_post: wgpu::Buffer,
    phi_res: wgpu::Buffer,
    pre_off: wgpu::Buffer,
    post_off: wgpu::Buffer,
    kv_k: wgpu::Buffer,
    kv_v: wgpu::Buffer,
    a_pre: f32,
    a_post: f32,
    a_res: f32,
    agate: f32,
}

struct SiteBuf {
    key: wgpu::Buffer,
    value: wgpu::Buffer,
    taps: wgpu::Buffer,
    vring: wgpu::Buffer,
}

struct Act {
    h: wgpu::Buffer,
    q: wgpu::Buffer,
    k: wgpu::Buffer,
    v: wgpu::Buffer,
    gt: wgpu::Buffer,
    o: wgpu::Buffer,
    ar: wgpu::Buffer,
    y: wgpu::Buffer,
    h2: wgpu::Buffer,
    m: wgpu::Buffer,
    s4: wgpu::Buffer,
    s4b: wgpu::Buffer,
    s16: wgpu::Buffer,
    t16: wgpu::Buffer,
    nx: wgpu::Buffer,
    lanes_a: wgpu::Buffer,
    lanes_b: wgpu::Buffer,
    u: wgpu::Buffer,
    bx: wgpu::Buffer,
    e: wgpu::Buffer,
    ek: wgpu::Buffer,
    ev: wgpu::Buffer,
    lm: wgpu::Buffer,
    logits: wgpu::Buffer,
    alpha1: wgpu::Buffer,
}

struct Engine {
    cact_path: String,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipes: HashMap<&'static str, wgpu::ComputePipeline>,
    layouts: HashMap<&'static str, wgpu::BindGroupLayout>,
    layers: Vec<LayerBuf>,
    sites: Vec<SiteBuf>,
    act: Act,
    emb: wgpu::Buffer,
    rope: wgpu::Buffer,
    dummy1: wgpu::Buffer,
    fnorm: wgpu::Buffer,
    staging: wgpu::Buffer,
    // CPU
    tok: SpTokenizer,
    emb_cpu: Vec<f32>,
    tables_cpu: Vec<Vec<f32>>,
    hist: Vec<u32>,
    n_layers: usize,
    d: usize,
    attn: usize,
    kv: usize,
    vocab: usize,
    kv_window: usize,
    max_seq: usize,
    rope_theta: f32,
    eos_id: u32,
    engram_sites: Vec<usize>,
    engram_orders: Vec<usize>,
    engram_taps: usize,
    engram_dil: usize,
    engram_slots: usize,
    vring_n: usize,
}

static ENG: Mutex<Option<Engine>> = Mutex::new(None);

// ------------------------------------------------------------------ helpers gpu

fn sbuf(device: &wgpu::Device, data: &[u8]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: data,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    })
}

fn sbuf_zero(device: &wgpu::Device, floats: usize) -> wgpu::Buffer {
    sbuf(device, &vec![0u8; floats * 4])
}

fn ubuf(device: &wgpu::Device, data: &[u8]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: data,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}

/// (binding, es_uniform, solo_lectura) por entry. DEBE coincidir con el WGSL.
fn spec(entry: &str) -> &'static [(u32, bool, bool)] {
    match entry {
        "rms_norm" => &[(0, false, false), (1, false, true), (2, true, true)],
        "rms_heads" => &[(0, false, false), (1, false, true), (2, true, true)],
        "matvec" => &[
            (0, false, false),
            (1, false, true),
            (2, false, true),
            (3, true, true),
        ],
        "rope" => &[(0, false, false), (1, false, true), (2, true, true)],
        "kv_write" => &[
            (0, false, true),
            (1, false, true),
            (2, false, false),
            (3, false, false),
            (4, true, true),
        ],
        "attn" => &[
            (0, false, true),
            (1, false, true),
            (2, false, true),
            (3, false, false),
            (4, true, true),
        ],
        "elem" => &[(0, false, false), (1, false, true), (2, true, true)],
        "axpy" => &[
            (0, false, false),
            (1, false, true),
            (2, false, true),
            (3, true, true),
        ],
        "axpy_b" => &[
            (0, false, false),
            (1, false, true),
            (2, false, true),
            (3, false, true),
            (4, true, true),
        ],
        "combine" => &[
            (0, false, false),
            (1, false, true),
            (2, false, true),
            (3, true, true),
        ],
        "copy_buf" => &[(0, false, false), (1, false, true), (2, true, true)],
        "sigmoid_affine" => &[
            (0, false, false),
            (1, false, true),
            (2, false, true),
            (3, false, true),
            (4, true, true),
        ],
        "lane_mix" => &[(0, false, false), (1, false, true), (2, false, true)],
        "lanes_new" => &[
            (0, false, false),
            (1, false, true),
            (2, false, true),
            (3, false, true),
            (4, false, true),
        ],
        "init_lanes" => &[(0, false, false), (1, false, true)],
        "mean_lanes" => &[(0, false, false), (1, false, true)],
        "fwht" => &[(0, false, false)],
        "hada_init" => &[(0, false, false), (1, false, true), (2, false, true)],
        "sinkhorn" => &[(0, false, false)],
        "rms_dot" => &[(0, false, true), (1, false, true), (2, false, false)],
        "engram_conv" => &[
            (0, false, false),
            (1, false, true),
            (2, false, true),
            (3, true, true),
        ],
        _ => &[],
    }
}

fn build_all(
    device: &wgpu::Device,
) -> Result<
    (
        HashMap<&'static str, wgpu::ComputePipeline>,
        HashMap<&'static str, wgpu::BindGroupLayout>,
    ),
    String,
> {
    wgpu::naga::front::wgsl::parse_str(SHADERS)
        .map(|_| ())
        .map_err(|e| format!("WGSL inválido:\n{e}"))?;
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("nengine"),
        source: wgpu::ShaderSource::Wgsl(SHADERS.into()),
    });
    let entries = [
        "rms_norm",
        "rms_heads",
        "matvec",
        "rope",
        "kv_write",
        "attn",
        "elem",
        "axpy",
        "axpy_b",
        "combine",
        "copy_buf",
        "sigmoid_affine",
        "lane_mix",
        "lanes_new",
        "init_lanes",
        "mean_lanes",
        "fwht",
        "hada_init",
        "sinkhorn",
        "rms_dot",
        "engram_conv",
    ];
    let mut pipes = HashMap::new();
    let mut layouts = HashMap::new();
    for e in entries {
        let ents: Vec<wgpu::BindGroupLayoutEntry> = spec(e)
            .iter()
            .map(|(i, uni, ro)| wgpu::BindGroupLayoutEntry {
                binding: *i,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: if *uni {
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    }
                } else {
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: *ro,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    }
                },
                count: None,
            })
            .collect();
        let layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &ents,
            });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipe =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(e),
                layout: Some(&pl),
                module: &module,
                entry_point: Some(e),
                compilation_options: Default::default(),
                cache: None,
            });
        pipes.insert(e, pipe);
        layouts.insert(e, layout);
    }
    Ok((pipes, layouts))
}

// ------------------------------------------------------------------ pesos

fn get_f32(cact: &Cact, idx: usize) -> Result<Vec<f32>, String> {
    let rec = cact.record(idx);
    if rec.dtype == DT_CQ {
        let w = cact.cq(idx).map_err(|e| format!("cq {idx}: {e}"))?;
        let mut v = vec![0.0f32; w.out_feat * w.in_feat];
        for o in 0..w.out_feat {
            let base = o * w.in_feat;
            w.dequantize_row(o, &mut v[base..base + w.in_feat]);
        }
        Ok(v)
    } else {
        cact.floats(idx).map_err(|e| format!("floats {idx}: {e}"))
    }
}

/// Rebana con chequeo (los slices fuera de rango son error, no panic).
fn sl(v: &[f32], a: usize, b: usize, name: &str) -> Result<Vec<f32>, String> {
    if b > v.len() || a > b {
        return Err(format!("tensor {name} corto: {} < {b}", v.len()));
    }
    Ok(v[a..b].to_vec())
}

fn at(v: &[f32], i: usize, name: &str) -> Result<f32, String> {
    v.get(i)
        .copied()
        .ok_or_else(|| format!("tensor {name} corto en {i}"))
}

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

    // embedding (GPU + copia CPU para lookup de fila)
    let emb_cpu = get_f32(&cact, lay.embedding)?;
    let emb = up(emb_cpu.clone());

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
            q: up(gw(l.q_proj)?),
            k: up(gw(l.k_proj)?),
            v: up(gw(l.v_proj)?),
            g: up(gw(l.gate_proj)?),
            o: up(gw(l.out_proj)?),
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
            phi_pre: up(sl(
                &phi_pre_v,
                li * 4 * 2048,
                (li + 1) * 4 * 2048,
                "phi_pre",
            )?),
            phi_post: up(sl(
                &phi_post_v,
                li * 4 * 2048,
                (li + 1) * 4 * 2048,
                "phi_post",
            )?),
            phi_res: up(sl(
                &phi_res_v,
                li * 16 * 2048,
                (li + 1) * 16 * 2048,
                "phi_res",
            )?),

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

    // engrams: tablas a CPU (gather), projs+taps a GPU
    let mut sites = Vec::new();
    let mut tables_cpu = Vec::new();
    for s in lay.engrams.iter() {
        tables_cpu.push(get_f32(&cact, s.tables)?);
        let ring = g.engram_conv_taps * g.engram_conv_dilation + 1;
        sites.push(SiteBuf {
            key: up(get_f32(&cact, s.key_proj)?),
            value: up(get_f32(&cact, s.value_proj)?),
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

// ------------------------------------------------------------------ dispatch

fn div64(n: u32) -> u32 {
    (n + 63) / 64
}

impl Engine {
    fn ub(&self, bytes: &[u8]) -> wgpu::Buffer {
        ubuf(&self.device, bytes)
    }

    fn bg(&self, entry: &str, bufs: &[&wgpu::Buffer]) -> wgpu::BindGroup {
        let layout = &self.layouts[entry];
        let ents: Vec<wgpu::BindGroupEntry> = bufs
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        self.device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout,
                entries: &ents,
            })
    }

    fn run(
        &self,
        enc: &mut wgpu::CommandEncoder,
        entry: &'static str,
        bufs: &[&wgpu::Buffer],
        x: u32,
    ) {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipes[entry]);
        pass.set_bind_group(0, &self.bg(entry, bufs), &[]);
        pass.dispatch_workgroups(x, 1, 1);
    }

    fn matvec(
        &self,
        enc: &mut wgpu::CommandEncoder,
        y: &wgpu::Buffer,
        w: &wgpu::Buffer,
        x: &wgpu::Buffer,
        rows: u32,
        cols: u32,
    ) {
        let u = self.ub(&u16f(rows, cols, 0, 0));
        self.run(enc, "matvec", &[y, w, x, &u], div64(rows));
    }

    fn norm(
        &self,
        enc: &mut wgpu::CommandEncoder,
        x: &wgpu::Buffer,
        gamma: Option<&wgpu::Buffer>,
        n: u32,
    ) {
        // OJO: rms_norm usa memoria workgroup → UN solo workgroup (256 hilos).
        let g = gamma.unwrap_or(&self.dummy1);
        let has = if gamma.is_some() { 1 } else { 0 };
        let u = self.ub(&u16f(n, has, 0, 0));
        self.run(enc, "rms_norm", &[x, g, &u], 1);
    }
}

// ------------------------------------------------------------------ un paso

/// Un token en posición `pos`: forward completo, devuelve logits f32.
fn step_token(eng: &Engine, token: u32, pos: usize) -> Result<Vec<f32>, String> {
    let d = eng.d as u32;
    let pos_u = pos as u32;
    let sqrt_d = (eng.d as f32).sqrt();

    // embedding lookup en CPU (1 fila) → upload
    let t = token as usize;
    if t * eng.d + eng.d > eng.emb_cpu.len() {
        return Err(format!("token {token} fuera de vocab"));
    }
    let mut x0 = eng.emb_cpu[t * eng.d..t * eng.d + eng.d].to_vec();
    for v in x0.iter_mut() {
        *v *= sqrt_d;
    }
    eng.queue
        .write_buffer(&eng.act.h, 0, &f32s(&x0));

    let mut enc = eng
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    let a = &eng.act;

    eng.run(&mut enc, "init_lanes", &[&a.lanes_a, &a.h], div64(512));

    // engram fill (antes del loop): hash+gather en CPU
    for si in 0..eng.engram_sites.len() {
        let order_base = &eng.engram_orders;
        let mut e = vec![0.0f32; 512];
        for tab in 0..4 {
            let order = *order_base.get(tab / 2).unwrap_or(&2);
            if order - 1 > pos {
                continue;
            }
            let mut acc = 0x9E3779B9u32.wrapping_mul((tab + 1) as u32);
            for j in 0..order {
                let tok = if j <= pos {
                    eng.hist.get(pos - j).copied().unwrap_or(0)
                } else {
                    0
                };
                acc = (acc ^ tok).wrapping_mul(0x01000193);
            }
            acc ^= acc >> 15;
            let idx = (acc as usize) % eng.engram_slots;
            let base = (tab * eng.engram_slots + idx) * 128;
            let src = &eng.tables_cpu[si];
            // sub_dim = 128
            for k in 0..128 {
                e[tab * 128 + k] = *src.get(base + k).unwrap_or(&0.0);
            }
        }
        eng.queue.write_buffer(&a.e, 0, &f32s(&e));
        let s = &eng.sites[si];
        eng.matvec(&mut enc, &a.ek, &s.key, &a.e, 512, 512);
        eng.matvec(&mut enc, &a.ev, &s.value, &a.e, 512, 512);
        enc.copy_buffer_to_buffer(
            &a.ev,
            0,
            &s.vring,
            ((pos % eng.vring_n) * 512 * 4) as u64,
            512 * 4,
        );
        let u = eng.ub(&u16f(pos_u, 0, 0, 0));
        eng.run(&mut enc, "engram_conv", &[&a.ev, &s.taps, &s.vring, &u], div64(512));
    }

    let lo = pos_u.saturating_sub((eng.kv_window - 1) as u32);
    let slot = (pos % eng.kv_window) as u32;

    for li in 0..eng.n_layers {
        let l = &eng.layers[li];
        let (lin, lout) = if li % 2 == 0 {
            (&a.lanes_a, &a.lanes_b)
        } else {
            (&a.lanes_b, &a.lanes_a)
        };
        // mHC-pre
        eng.run(&mut enc, "copy_buf", &[&a.nx, lin, &eng.ub(&u16f(2048, 0, 0, 0))], div64(2048));
        eng.norm(&mut enc, &a.nx, None, 2048);
        eng.matvec(&mut enc, &a.s4, &l.phi_pre, &a.nx, 4, 2048);
        {
            let u = eng.ub(&{
                let mut v = Vec::with_capacity(16);
                v.extend_from_slice(&l.a_pre.to_le_bytes());
                v.extend_from_slice(&1.0f32.to_le_bytes());
                v.extend_from_slice(&4u32.to_le_bytes());
                v.extend_from_slice(&0u32.to_le_bytes());
                v
            });
            eng.run(
                &mut enc,
                "sigmoid_affine",
                &[&a.s4b, &a.s4, &l.b_pre, &l.pre_off, &u],
                1,
            );
        }
        eng.run(&mut enc, "lane_mix", &[&a.u, &a.s4b, lin], div64(512));
        eng.run(&mut enc, "copy_buf", &[&a.bx, &a.u, &eng.ub(&u16f(512, 0, 0, 0))], div64(512));

        // engram gate en sites
        if eng.engram_sites.contains(&li) {
            // OJO: rms_dot usa memoria workgroup → UN solo workgroup.
            eng.run(&mut enc, "rms_dot", &[&a.u, &a.ek, &a.alpha1], 1);
            eng.run(
                &mut enc,
                "axpy_b",
                &[&a.bx, &a.u, &a.alpha1, &a.ev, &eng.ub(&u16f(512, 0, 0, 0))],
                div64(512),
            );
        }

        // block
        eng.run(&mut enc, "copy_buf", &[&a.h, &a.bx, &eng.ub(&u16f(512, 0, 0, 0))], div64(512));
        eng.norm(&mut enc, &a.h, Some(&l.nin), 512);
        eng.matvec(&mut enc, &a.q, &l.q, &a.h, 512, 512);
        eng.matvec(&mut enc, &a.k, &l.k, &a.h, 256, 512);
        eng.matvec(&mut enc, &a.v, &l.v, &a.h, 256, 512);
        // gate va a su propio buffer: attention pisa a.o después.
        eng.matvec(&mut enc, &a.gt, &l.g, &a.h, 512, 512);
        {
            let u = eng.ub(&u16f(512, 0, 0, 0));
            eng.run(&mut enc, "rms_heads", &[&a.q, &l.qn, &u], div64(512));
        }
        {
            let u = eng.ub(&u16f(256, 0, 0, 0));
            eng.run(&mut enc, "rms_heads", &[&a.k, &l.kn, &u], div64(256));
        }
        {
            let u = eng.ub(&u16f(pos_u, 8, 0, 0));
            eng.run(&mut enc, "rope", &[&a.q, &eng.rope, &u], div64(8 * 32));
        }
        {
            let u = eng.ub(&u16f(pos_u, 4, 0, 0));
            eng.run(&mut enc, "rope", &[&a.k, &eng.rope, &u], div64(4 * 32));
        }
        {
            let u = eng.ub(&u16f(slot, 0, 0, 0));
            eng.run(
                &mut enc,
                "kv_write",
                &[&a.k, &a.v, &l.kv_k, &l.kv_v, &u],
                div64(256),
            );
        }
        {
            let mut v = Vec::with_capacity(16);
            v.extend_from_slice(&pos_u.to_le_bytes());
            v.extend_from_slice(&lo.to_le_bytes());
            v.extend_from_slice(&INV_SQRT_HEAD.to_le_bytes());
            v.extend_from_slice(&0u32.to_le_bytes());
            let u = eng.ub(&v);
            eng.run(
                &mut enc,
                "attn",
                &[&a.q, &l.kv_k, &l.kv_v, &a.o, &u],
                8,
            );
        }
        {
            let u = eng.ub(&u16f(512, 0, 0, 0));
            eng.run(&mut enc, "elem", &[&a.o, &a.gt, &u], div64(512)); // modo 0
        }
        eng.matvec(&mut enc, &a.ar, &l.o, &a.o, 512, 512);
        eng.norm(&mut enc, &a.ar, Some(&l.pnorm), 512);
        {
            let s = 1.0 / (1.0 + (-l.agate).exp());
            let u = eng.ub(&u16fb(s, 512));
            eng.run(&mut enc, "axpy", &[&a.y, &a.bx, &a.ar, &u], div64(512));
        }
        eng.run(&mut enc, "copy_buf", &[&a.h2, &a.y, &eng.ub(&u16f(512, 0, 0, 0))], div64(512));
        eng.norm(&mut enc, &a.h2, Some(&l.phada), 512);
        eng.run(&mut enc, "hada_init", &[&a.m, &a.h2, &l.d1], div64(512));
        // OJO: fwht usa shared mem → UN solo workgroup de 256.
        eng.run(&mut enc, "fwht", &[&a.m], 1);
        {
            let u = eng.ub(&u16f(512, 2, 0, 0));
            eng.run(&mut enc, "elem", &[&a.m, &l.d2, &u], div64(512)); // modo 2
        }
        eng.run(&mut enc, "fwht", &[&a.m], 1);
        {
            let u = eng.ub(&u16f(512, 1, 0, 0));
            eng.run(&mut enc, "elem", &[&a.m, &l.d3, &u], div64(512)); // modo 1
        }
        {
            let u = eng.ub(&u16f(512, 0, 0, 0));
            eng.run(&mut enc, "combine", &[&a.y, &a.m, &a.u, &u], div64(512));
        }

        // mHC-post
        eng.matvec(&mut enc, &a.s16, &l.phi_res, &a.nx, 16, 2048);
        {
            let u = eng.ub(&u16fb(l.a_res, 16));
            eng.run(
                &mut enc,
                "axpy",
                &[&a.t16, &l.b_res, &a.s16, &u],
                1,
            );
        }
        eng.run(&mut enc, "sinkhorn", &[&a.t16], 1);
        eng.matvec(&mut enc, &a.s4, &l.phi_post, &a.nx, 4, 2048);
        {
            let mut v = Vec::with_capacity(16);
            v.extend_from_slice(&l.a_post.to_le_bytes());
            v.extend_from_slice(&2.0f32.to_le_bytes());
            v.extend_from_slice(&4u32.to_le_bytes());
            v.extend_from_slice(&0u32.to_le_bytes());
            let u = eng.ub(&v);
            eng.run(
                &mut enc,
                "sigmoid_affine",
                &[&a.s4b, &a.s4, &l.b_post, &l.post_off, &u],
                1,
            );
        }
        eng.run(
            &mut enc,
            "lanes_new",
            &[lout, lin, &a.t16, &a.s4b, &a.y],
            div64(2048),
        );
    }

    let last = if (eng.n_layers - 1) % 2 == 0 {
        &a.lanes_b
    } else {
        &a.lanes_a
    };
    eng.run(&mut enc, "mean_lanes", &[&a.lm, last], div64(512));
    eng.norm(&mut enc, &a.lm, Some(&eng.fnorm), 512);
    eng.matvec(&mut enc, &a.logits, &eng.emb, &a.lm, eng.vocab as u32, 512);
    enc.copy_buffer_to_buffer(
        &a.logits,
        0,
        &eng.staging,
        0,
        (eng.vocab * 4) as u64,
    );
    let _ = eng.queue.submit(Some(enc.finish()));

    // readback logits
    let slice = eng.staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    eng.device.poll(wgpu::PollType::wait()).expect("poll");
    rx.recv()
        .map_err(|_| "readback roto".to_string())?
        .map_err(|e| format!("map: {e:?}"))?;
    let data = slice.get_mapped_range();
    let mut out = vec![0.0f32; eng.vocab];
    for (i, ch) in data.chunks_exact(4).enumerate() {
        out[i] = f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]);
    }
    drop(data);
    eng.staging.unmap();
    Ok(out)
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
    Ok(format!(
        "step_ms={ms:.1} · vocab={} · capas={} · top={top}",
        eng.vocab, eng.n_layers
    ))
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

fn generate_inner(
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
    })
}

/// Paridad CPU vs GPU: mismos 8 tokens greedy, compara id por id.
/// Carga un motor CPU temporal con el mismo .cact (no toca el GPU).
pub fn parity(query: &str) -> Result<String, String> {
    let cact_path = {
        let g = ENG
            .lock()
            .map_err(|_| "mutex envenenado".to_string())?;
        g.as_ref().ok_or("motor GPU no cargado")?.cact_path.clone()
    };
    use needle_infer::v2_engine::{GenerateOptions, V2Engine};
    let cpu = V2Engine::load(std::path::Path::new(cact_path))
        .map_err(|e| format!("cpu load: {e}"))?;
    let opts = GenerateOptions {
        max_new_tokens: 8,
        temperature: 0.0,
        seed: 0,
        system: None,
        prefill_chunk: 128,
        constrain: false,
    };
    let rc = cpu.generate(query, "[]", &opts, |_, _| {});
    let rg = generate_inner(query, "[]", 8, 0.0, 0, &mut |_| {})?;
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
    match dif {
        None if rc.token_ids.len() == rg.token_ids.len() => Ok(format!(
            "MATCH 8/8 · cpu=[{}] gpu=[{}]",
            show(&rc.token_ids),
            show(&rg.token_ids)
        )),
        _ => Ok(format!(
            "DIF@{} · cpu=[{}] gpu=[{}]",
            dif.map(|k| k.to_string()).unwrap_or("len".into()),
            show(&rc.token_ids),
            show(&rg.token_ids)
        )),
    }
}

pub fn unload() {
    if let Ok(mut g) = ENG.lock() {
        *g = None;
    }
}

pub fn is_loaded() -> bool {
    matches!(ENG.lock(), Ok(g) if g.is_some())
}

