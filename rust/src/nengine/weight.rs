//! Pesos del .cact: lectura f32 y subida CQ empaquetada a GPU.

use needle_infer::cact::{Cact, DT_CQ};

use super::help::{f32s, sbuf};
use super::types::{CqUp, Wmat};

// ------------------------------------------------------------------ pesos

pub(crate) fn get_f32(cact: &Cact, idx: usize) -> Result<Vec<f32>, String> {
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

fn f16s_to_f32(chunk: &[u8]) -> Vec<f32> {
    chunk
        .chunks_exact(2)
        .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect()
}

/// Sube un tensor de matmul: CQ → empaquetado+normas+codebook (mismo .cact,
/// cero dequant); otro dtype → f32. [rs, rs+rows) banda para mHC.
/// exp_out/exp_in son las dims ESPERADAS ([out,in] row-major): se verifican
/// contra el shape en vez de asumir orientación.
#[allow(clippy::too_many_arguments)]
pub(crate) fn load_wmat(
    device: &wgpu::Device,
    cact: &Cact,
    raw: &[u8],
    codebook: &[f32],
    idx: usize,
    exp_out: usize,
    exp_in: usize,
    rs: usize,
    rows: usize,
    etiqueta: &str,
) -> Result<Wmat, String> {
    let rec = cact.record(idx);
    if rec.dtype != DT_CQ {
        let v = cact.floats(idx).map_err(|e| format!("floats {idx}: {e}"))?;
        if v.len() < exp_out * exp_in {
            return Err(format!("tensor {etiqueta} muy chico"));
        }
        // Banda para mHC aunque sea FP (rara vez pasa: mhc es CQ-4).
        let band = if rows != exp_out {
            if (rs + rows) * exp_in > v.len() {
                return Err(format!("tensor {etiqueta}: banda FP fuera de rango"));
            }
            v[rs * exp_in..(rs + rows) * exp_in].to_vec()
        } else {
            v[..exp_out * exp_in].to_vec()
        };
        return Ok(Wmat::F32 {
            b: sbuf(device, &f32s(&band)),
            idx,
            rs,
        });
    }
    if rec.ndim != 2 || rec.shape.len() < 2 {
        return Err(format!("tensor {etiqueta} no 2-D"));
    }
    if rec.shape[0] != exp_out || rec.shape[1] != exp_in {
        return Err(format!(
            "tensor {etiqueta}: shape {:?} != esperado [{exp_out},{exp_in}]",
            &rec.shape[..2]
        ));
    }
    let (out, inn) = (exp_out, exp_in);
    let (group, bits) = (rec.group, rec.bits);
    if group != 128 {
        return Err(format!("tensor {etiqueta}: grupo {group} != 128 (GPU fase 2)"));
    }
    if bits != 2 && bits != 4 && bits != 5 {
        return Err(format!("tensor {etiqueta}: bits {bits} sin kernel GPU"));
    }
    if rs + rows > out {
        return Err(format!("tensor {etiqueta}: banda fuera de rango"));
    }
    let eff = if bits == 5 { 2 } else { bits as usize };
    let in_padded = inn.div_ceil(group) * group;
    let ngroups = in_padded / group;
    let row_bytes = in_padded * eff / 8;
    let n_packed = out * row_bytes;
    let n_norms_b = out * ngroups * 2;
    let off = rec.offset as usize;
    let nbytes = rec.nbytes as usize;
    if off + nbytes > raw.len() || n_packed + n_norms_b != nbytes {
        return Err(format!(
            "tensor {etiqueta}: blob inconsistente (off {off}, len {nbytes})"
        ));
    }
    // banda de filas
    let p0 = off + rs * row_bytes;
    let packed_b = &raw[p0..p0 + rows * row_bytes];
    let mut packed = Vec::with_capacity(rows * row_bytes / 4);
    for ch in packed_b.chunks_exact(4) {
        packed.push(u32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]));
    }
    let nb0 = off + n_packed + rs * ngroups * 2;
    let norms = f16s_to_f32(&raw[nb0..nb0 + rows * ngroups * 2]);
    let levels: Vec<f32> = if bits == 5 {
        let c = 1.2240064 / (group as f32).sqrt();
        vec![-c, 0.0, c]
    } else if bits == 2 {
        codebook
            .get(0..4)
            .ok_or_else(|| format!("tensor {etiqueta}: codebook corto"))?
            .to_vec()
    } else {
        codebook
            .get(12..28)
            .ok_or_else(|| format!("tensor {etiqueta}: codebook corto"))?
            .to_vec()
    };
    let nlevels = levels.len() as u32;
    Ok(Wmat::Cq(CqUp {
        packed: sbuf(device, & {
            let mut b = Vec::with_capacity(packed.len() * 4);
            for w in &packed {
                b.extend_from_slice(&w.to_le_bytes());
            }
            b
        }),
        norms: sbuf(device, &f32s(&norms)),
        levels: sbuf(device, &f32s(&levels)),
        bits: bits as u32,
        row_u32: (row_bytes / 4) as u32,
        ngroups: ngroups as u32,
        idx,
        rs,
        group: group as u32,
        in_padded: in_padded as u32,
        out_feat: rows as u32,
        is_ternary: if bits == 5 { 1 } else { 0 },
        nlevels,
    }))
}
