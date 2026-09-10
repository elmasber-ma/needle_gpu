//! Estado del motor: tipos de pesos y buffers residentes.

use std::collections::HashMap;

use needle_infer::sp_tokenizer::SpTokenizer;

// ------------------------------------------------------------------ estado

/// Peso de matmul: o matriz f32 subida, o peso CQ empaquetado
/// (mismos bytes del .cact, desempaquetado al vuelo en el shader).
/// `idx`/`rs` = récord .cact y fila inicial (para referencia CPU en diag).
pub(crate) enum Wmat {
    F32 {
        b: wgpu::Buffer,
        idx: usize,
        rs: usize,
    },
    Cq(CqUp),
}

/// Vista CQ subida a GPU: blob empaquetado + normas + codebook.
pub(crate) struct CqUp {
    pub(crate) packed: wgpu::Buffer, // u32[]
    pub(crate) norms: wgpu::Buffer,  // f32[out*ngroups]
    pub(crate) levels: wgpu::Buffer, // f32 niveles del codebook / ternario
    pub(crate) bits: u32,
    pub(crate) row_u32: u32,
    pub(crate) ngroups: u32,
    pub(crate) idx: usize,
    pub(crate) rs: usize,
    pub(crate) group: u32,
    pub(crate) in_padded: u32,
    pub(crate) out_feat: u32,
    pub(crate) is_ternary: u32,
    pub(crate) nlevels: u32,
}

pub(crate) struct LayerBuf {
    pub(crate) q: Wmat,
    pub(crate) k: Wmat,
    pub(crate) v: Wmat,
    pub(crate) g: Wmat,
    pub(crate) o: Wmat,
    pub(crate) qn: wgpu::Buffer,
    pub(crate) kn: wgpu::Buffer,
    pub(crate) nin: wgpu::Buffer,
    pub(crate) pnorm: wgpu::Buffer,
    pub(crate) phada: wgpu::Buffer,
    pub(crate) d1: wgpu::Buffer,
    pub(crate) d2: wgpu::Buffer,
    pub(crate) d3: wgpu::Buffer,
    pub(crate) b_pre: wgpu::Buffer,
    pub(crate) b_post: wgpu::Buffer,
    pub(crate) b_res: wgpu::Buffer,
    pub(crate) phi_pre: Wmat,
    pub(crate) phi_post: Wmat,
    pub(crate) phi_res: Wmat,
    pub(crate) pre_off: wgpu::Buffer,
    pub(crate) post_off: wgpu::Buffer,
    pub(crate) kv_k: wgpu::Buffer,
    pub(crate) kv_v: wgpu::Buffer,
    pub(crate) a_pre: f32,
    pub(crate) a_post: f32,
    pub(crate) a_res: f32,
    pub(crate) agate: f32,
}

pub(crate) struct SiteBuf {
    pub(crate) key: Wmat,
    pub(crate) value: Wmat,
    pub(crate) taps: wgpu::Buffer,
    pub(crate) vring: wgpu::Buffer,
}

/// Fuente de fila de embedding en CPU (el embedding suele ser CQ-4).
pub(crate) enum EmbSrc {
    F32(Vec<f32>),
    Cq(needle_core::cq::CqWeight),
}

impl EmbSrc {
    pub(crate) fn row(&self, t: usize, d: usize, sqrt_d: f32) -> Result<Vec<f32>, String> {
        match self {
            EmbSrc::F32(v) => {
                if t * d + d > v.len() {
                    return Err(format!("token {t} fuera de vocab"));
                }
                let mut r = v[t * d..t * d + d].to_vec();
                for x in r.iter_mut() {
                    *x *= sqrt_d;
                }
                Ok(r)
            }
            EmbSrc::Cq(w) => {
                if t >= w.out_feat {
                    return Err(format!("token {t} fuera de vocab"));
                }
                let mut r = vec![0.0f32; d];
                w.dequantize_row(t, &mut r);
                for x in r.iter_mut() {
                    *x *= sqrt_d;
                }
                Ok(r)
            }
        }
    }
}

pub(crate) struct Act {
    pub(crate) h: wgpu::Buffer,
    pub(crate) q: wgpu::Buffer,
    pub(crate) k: wgpu::Buffer,
    pub(crate) v: wgpu::Buffer,
    pub(crate) gt: wgpu::Buffer,
    pub(crate) o: wgpu::Buffer,
    pub(crate) ar: wgpu::Buffer,
    pub(crate) y: wgpu::Buffer,
    pub(crate) h2: wgpu::Buffer,
    pub(crate) m: wgpu::Buffer,
    pub(crate) s4: wgpu::Buffer,
    pub(crate) s4b: wgpu::Buffer,
    pub(crate) s16: wgpu::Buffer,
    pub(crate) t16: wgpu::Buffer,
    pub(crate) nx: wgpu::Buffer,
    pub(crate) lanes_a: wgpu::Buffer,
    pub(crate) lanes_b: wgpu::Buffer,
    pub(crate) u: wgpu::Buffer,
    pub(crate) bx: wgpu::Buffer,
    pub(crate) e: wgpu::Buffer,
    pub(crate) ek: wgpu::Buffer,
    pub(crate) ev: wgpu::Buffer,
    pub(crate) lm: wgpu::Buffer,
    pub(crate) logits: wgpu::Buffer,
    pub(crate) alpha1: wgpu::Buffer,
    pub(crate) xh_h: wgpu::Buffer,
    pub(crate) xh_o: wgpu::Buffer,
    pub(crate) xh_e: wgpu::Buffer,
    pub(crate) xh_nx: wgpu::Buffer,
    pub(crate) xh_lm: wgpu::Buffer,
}

pub(crate) struct Engine {
    pub(crate) cact_path: String,
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub(crate) pipes: HashMap<&'static str, wgpu::ComputePipeline>,
    pub(crate) layouts: HashMap<&'static str, wgpu::BindGroupLayout>,
    pub(crate) layers: Vec<LayerBuf>,
    pub(crate) sites: Vec<SiteBuf>,
    pub(crate) act: Act,
    pub(crate) emb: Wmat,
    pub(crate) rope: wgpu::Buffer,
    pub(crate) dummy1: wgpu::Buffer,
    pub(crate) fnorm: wgpu::Buffer,
    pub(crate) staging: wgpu::Buffer,
    // CPU
    pub(crate) tok: SpTokenizer,
    pub(crate) emb_cpu: EmbSrc,
    pub(crate) tables_cpu: Vec<Vec<f32>>,
    pub(crate) hist: Vec<u32>,
    pub(crate) n_layers: usize,
    pub(crate) d: usize,
    pub(crate) attn: usize,
    pub(crate) kv: usize,
    pub(crate) vocab: usize,
    pub(crate) kv_window: usize,
    pub(crate) max_seq: usize,
    pub(crate) rope_theta: f32,
    pub(crate) eos_id: u32,
    pub(crate) engram_sites: Vec<usize>,
    pub(crate) engram_orders: Vec<usize>,
    pub(crate) engram_taps: usize,
    pub(crate) engram_dil: usize,
    pub(crate) engram_slots: usize,
    pub(crate) vring_n: usize,
}
