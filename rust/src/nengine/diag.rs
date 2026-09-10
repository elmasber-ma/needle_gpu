//! Diagnóstico: barridos GPU vs referencia (proyecciones, prepare, KV).

use needle_infer::cact::{Cact, DT_CQ};

use super::help::{div64, f32s, read_storage, u16f, INV_SQRT_HEAD};
use super::types::{Engine, Wmat};
use super::weight::get_f32;

// --------------------------------------- barrido por proyección (diag CQ)

// Una proyección GPU vs pesos reales: entrada determinista, prepare+proj en
// GPU, matvec f32 ingenuo en CPU. Devuelve Δmax.
fn sweep_one(
    eng: &Engine,
    cact: &Cact,
    name: &str,
    w: &Wmat,
    rows: u32,
    in_feat: u32,
    x: &[f32],
) -> Result<(String, f32), String> {
    let (idx, rs) = match w {
        Wmat::F32 { idx, rs, .. } => (*idx, *rs),
        Wmat::Cq(c) => (c.idx, c.rs),
    };
    // referencia CPU: pesos reales × x
    let full = get_f32(cact, idx)?;
    let base = rs * in_feat as usize;
    let need = base + rows as usize * in_feat as usize;
    if need > full.len() {
        return Err(format!("{name}: pesos cortos"));
    }
    let mut y_cpu = vec![0.0f32; rows as usize];
    for o in 0..rows as usize {
        let mut acc = 0.0f32;
        let row = &full[base + o * in_feat as usize..base + (o + 1) * in_feat as usize];
        for (a, b) in row.iter().zip(x.iter()) {
            acc += a * b;
        }
        y_cpu[o] = acc;
    }
    // referencia needle-core oficial (mismo x): aísla mi slice/shader.
    // d_nc = |nc - gpu|; si d_nc≈0 pero d_cpu grande, miente mi get_f32.
    let mut d_nc = -1.0f32;
    if cact.record(idx).dtype == DT_CQ {
        if let Ok(w) = cact.cq(idx).map_err(|_| "cq".to_string()) {
            if w.in_feat == x.len() {
                let mut xh_nc = vec![0.0f32; w.in_padded];
                w.prepare_input(x, &mut xh_nc);
                let mut y_nc = vec![0.0f32; rows as usize];
                w.matvec_rows_prepared(&xh_nc, rs, &mut y_nc);
                // se compara contra y_gpu más abajo; guardo en y_cpu2 vía dm extra
                d_nc = 0.0;
                for (a, b) in y_nc.iter().zip(y_cpu.iter()) {
                    d_nc = d_nc.max((a - b).abs());
                }
                // d_nc aquí = |nc - cpu| (chequeo de mi referencia)
                let _ = &y_nc;
            }
        }
    }
    let _ = d_nc;
    // GPU: misma x por prepare+proj, salida a logits, readback
    let a = &eng.act;
    let (xbuf, xh) = if in_feat == 2048 {
        (&a.nx, &a.xh_nx)
    } else {
        (&a.h, &a.xh_h)
    };
    eng.queue.write_buffer(xbuf, 0, &f32s(x));
    let mut enc = eng
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    eng.prepare(&mut enc, xh, xbuf, in_feat);
    eng.proj(&mut enc, &a.logits, w, xbuf, xh, rows, in_feat);
    enc.copy_buffer_to_buffer(&a.logits, 0, &eng.staging, 0, (rows * 4) as u64);
    eng.queue.submit(Some(enc.finish()));
    let slice = eng.staging.slice(..(rows * 4) as u64);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    eng.device.poll(wgpu::PollType::wait()).expect("poll");
    rx.recv()
        .map_err(|_| "readback roto".to_string())?
        .map_err(|e| format!("map: {e:?}"))?;
    let data = slice.get_mapped_range();
    let mut dm = 0.0f32;
    for (i, ch) in data.chunks_exact(4).enumerate() {
        if i >= y_cpu.len() {
            break;
        }
        let v = f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]);
        dm = dm.max((v - y_cpu[i]).abs());
    }
    drop(data);
    eng.staging.unmap();
    Ok((name.to_string(), dm))
}

/// Compara la rotación Hadamard (prepare) GPU vs la oficial de needle-core.
/// Devuelve Δmax. Si esto da ~0, el prepare está bien y el bug es decode.
fn sweep_prep(eng: &Engine, cact: &Cact, idx: usize, x: &[f32]) -> Result<f32, String> {
    let w: needle_core::cq::CqWeight = cact.cq(idx).map_err(|e| format!("cq: {e}"))?;
    let mut xh_cpu = vec![0.0f32; w.in_padded];
    w.prepare_input(x, &mut xh_cpu);
    let a = &eng.act;
    let (xbuf, xh) = if x.len() == 2048 {
        (&a.nx, &a.xh_nx)
    } else {
        (&a.h, &a.xh_h)
    };
    eng.queue.write_buffer(xbuf, 0, &f32s(x));
    let mut enc = eng
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    eng.prepare(&mut enc, xh, xbuf, x.len() as u32);
    let n = xh_cpu.len();
    enc.copy_buffer_to_buffer(xh, 0, &eng.staging, 0, (n * 4) as u64);
    eng.queue.submit(Some(enc.finish()));
    let slice = eng.staging.slice(..(n * 4) as u64);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    eng.device.poll(wgpu::PollType::wait()).expect("poll");
    rx.recv()
        .map_err(|_| "readback roto".to_string())?
        .map_err(|e| format!("map: {e:?}"))?;
    let data = slice.get_mapped_range();
    let mut dm = 0.0f32;
    for (i, ch) in data.chunks_exact(4).enumerate() {
        if i >= n {
            break;
        }
        let v = f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]);
        dm = dm.max((v - xh_cpu[i]).abs());
    }
    drop(data);
    eng.staging.unmap();
    Ok(dm)
}

/// Tests unitarios de lo que el sweep no cubre: rms_heads, rope y
/// kv_write+attention. El paso 0 es exacto y el 1 diverge: lo único que
/// viaja entre pasos es el KV-cache, así que el bug vive acá.
pub(crate) fn sweep_kb(eng: &Engine) -> Result<String, String> {
    let mut s = 0xABCDEF01u32;
    let mut rnd = || {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        (s >> 8) as f32 / 16777216.0 - 0.5
    };
    let a = &eng.act;
    let l = &eng.layers[0];
    let mut enc = eng
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    // A: rms_heads sobre a.q con gamma=q_norm de l0
    let x: Vec<f32> = (0..512).map(|_| rnd()).collect();
    eng.queue.write_buffer(&a.q, 0, &f32s(&x));
    let g = read_storage(eng, &l.qn, 64)?;
    let u = eng.ub(&u16f(512, 0, 0, 0));
    eng.run(&mut enc, "rms_heads", &[&a.q, &l.qn, &u], div64(512));
    eng.queue.submit(Some(enc.finish()));
    let yq = read_storage(eng, &a.q, 512)?;
    let mut d_rms = 0.0f32;
    for i in 0..512 {
        let base = (i / 64) * 64;
        let mut ss = 0.0f32;
        for j in 0..64 {
            ss += x[base + j] * x[base + j];
        }
        let want = (1.0 + g[i % 64]) * x[i] / (ss / 64.0 + 1e-6).sqrt();
        d_rms = d_rms.max((yq[i] - want).abs());
    }

    // B: rope sobre a.q (8 heads), pos=3
    let x2: Vec<f32> = (0..512).map(|_| rnd()).collect();
    eng.queue.write_buffer(&a.q, 0, &f32s(&x2));
    let mut enc = eng
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    let u = eng.ub(&u16f(3, 8, 0, 0));
    eng.run(&mut enc, "rope", &[&a.q, &eng.rope, &u], div64(8 * 32));
    eng.queue.submit(Some(enc.finish()));
    let yq2 = read_storage(eng, &a.q, 512)?;
    let mut d_rope = 0.0f32;
    for h in 0..8 {
        for i in 0..32 {
            let ang = 3.0 / eng.rope_theta.powf(2.0 * i as f32 / 64.0);
            let (c, sn) = (ang.cos(), ang.sin());
            let (a0, b0) = (x2[h * 64 + i], x2[h * 64 + 32 + i]);
            d_rope = d_rope.max((yq2[h * 64 + i] - (a0 * c - b0 * sn)).abs());
            d_rope = d_rope.max((yq2[h * 64 + 32 + i] - (b0 * c + a0 * sn)).abs());
        }
    }

    // C: kv_write (slot 1 vía kernel) + attention pos=1, lo=0
    let k0: Vec<f32> = (0..256).map(|_| rnd()).collect();
    let k1: Vec<f32> = (0..256).map(|_| rnd()).collect();
    let v0: Vec<f32> = (0..256).map(|_| rnd()).collect();
    let v1: Vec<f32> = (0..256).map(|_| rnd()).collect();
    let q: Vec<f32> = (0..512).map(|_| rnd()).collect();
    eng.queue.write_buffer(&l.kv_k, 0, &f32s(&k0));
    eng.queue.write_buffer(&l.kv_v, 0, &f32s(&v0));
    eng.queue.write_buffer(&l.kv_v, (256 * 4) as u64, &f32s(&v1));
    eng.queue.write_buffer(&a.q, 0, &f32s(&q));
    eng.queue.write_buffer(&a.k, 0, &f32s(&k1));
    let mut enc = eng
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    let u = eng.ub(&u16f(1, 0, 0, 0));
    eng.run(
        &mut enc,
        "kv_write",
        &[&a.k, &a.v, &l.kv_k, &l.kv_v, &u],
        div64(256),
    );
    // OJO: kv_write también escribe V desde a.v (basura de tests previos);
    // reescribo el slot 1 de V directo para aislar attention.
    eng.queue.submit(Some(enc.finish()));
    eng.queue.write_buffer(&l.kv_v, (256 * 4) as u64, &f32s(&v1));
    let mut enc = eng
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    let mut v = Vec::with_capacity(16);
    v.extend_from_slice(&1u32.to_le_bytes());
    v.extend_from_slice(&0u32.to_le_bytes());
    v.extend_from_slice(&INV_SQRT_HEAD.to_le_bytes());
    v.extend_from_slice(&0u32.to_le_bytes());
    let u = eng.ub(&v);
    eng.run(&mut enc, "attn", &[&a.q, &l.kv_k, &l.kv_v, &a.o, &u], 8);
    eng.queue.submit(Some(enc.finish()));
    let yo = read_storage(eng, &a.o, 512)?;
    // kv_write check: slot 1 de K debe ser k1
    let kk = read_storage(eng, &l.kv_k, 512)?;
    let mut d_kv = 0.0f32;
    for i in 0..256 {
        d_kv = d_kv.max((kk[i] - k0[i]).abs());
        d_kv = d_kv.max((kk[256 + i] - k1[i]).abs());
    }
    // referencia attention CPU
    let mut d_attn = 0.0f32;
    for h in 0..8 {
        let kv = h / 2;
        let mut sc = [0.0f32; 2];
        for t in 0..2 {
            let k = if t == 0 { &k0 } else { &k1 };
            let mut acc = 0.0f32;
            for d in 0..64 {
                acc += q[h * 64 + d] * k[kv * 64 + d];
            }
            sc[t] = acc * 0.125;
        }
        let m = sc[0].max(sc[1]);
        let e0 = (sc[0] - m).exp();
        let e1 = (sc[1] - m).exp();
        let (w0, w1) = (e0 / (e0 + e1), e1 / (e0 + e1));
        for d in 0..64 {
            let want = w0 * v0[kv * 64 + d] + w1 * v1[kv * 64 + d];
            d_attn = d_attn.max((yo[h * 64 + d] - want).abs());
        }
    }
    Ok(format(
        "kb=[rms:{d_rms:.5} rope:{d_rope:.5} kv:{d_kv:.5} attn:{d_attn:.5}]"
    ))
}

/// Extrae (idx .cact) de un Wmat.
fn widx(w: &Wmat) -> usize {
    match w {
        Wmat::F32 { idx, .. } => *idx,
        Wmat::Cq(c) => c.idx,
    }
}

/// Corre todas las proyecciones del motor contra los pesos reales.
/// Aísla un bug de prepare/matvec del resto del forward.
pub(crate) fn sweep_all(eng: &Engine) -> Result<String, String> {
    let cact = Cact::load(std::path::Path::new(&eng.cact_path))
        .map_err(|e| format!("reabrir cact: {e}"))?;
    let mut s = 0x12345678u32;
    let mut rnd = || {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        (s >> 8) as f32 / 16777216.0 - 0.5
    };
    let x512: Vec<f32> = (0..512).map(|_| rnd()).collect();
    let x2048: Vec<f32> = (0..2048).map(|_| rnd()).collect();
    let mut worst: Vec<(String, f32)> = Vec::new();
    let mut n = 0u32;
    let mut gmax = 0.0f32;
    let mut run = |name: String, w: &Wmat, rows: u32, inf: u32, x: &[f32]| -> Result<(), String> {
        let (_, dm) = sweep_one(eng, &cact, &name, w, rows, inf, x)?;
        n += 1;
        gmax = gmax.max(dm);
        if dm > 0.02 {
            worst.push((name, dm));
        }
        Ok(())
    };
    run("emb".into(), &eng.emb, eng.vocab as u32, 512, &x512)?;
    for li in 0..eng.n_layers {
        let l = &eng.layers[li];
        run(format!("q{li}"), &l.q, 512, 512, &x512)?;
        run(format!("k{li}"), &l.k, 256, 512, &x512)?;
        run(format!("v{li}"), &l.v, 256, 512, &x512)?;
        run(format!("g{li}"), &l.g, 512, 512, &x512)?;
        run(format!("o{li}"), &l.o, 512, 512, &x512)?;
        run(format!("pp{li}"), &l.phi_pre, 4, 2048, &x2048)?;
        run(format!("po{li}"), &l.phi_post, 4, 2048, &x2048)?;
        run(format!("pr{li}"), &l.phi_res, 16, 2048, &x2048)?;
    }
    for (si, st) in eng.sites.iter().enumerate() {
        run(format!("ek{si}"), &st.key, 512, 512, &x512)?;
        run(format!("ev{si}"), &st.value, 512, 512, &x512)?;
    }
    worst.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mal: Vec<String> = worst
        .iter()
        .take(10)
        .map(|(nm, dm)| format!("{nm}:{dm:.2}"))
        .collect();
    // máximo por tipo de proyección (cabe en el chat y dice QUÉ clase falla)
    let mut kind: std::collections::HashMap<String, f32> = [
        "emb", "q", "k", "v", "g", "o", "pp", "po", "pr", "ek", "ev",
    ]
    .iter()
    .map(|k| (k.to_string(), 0.0f32))
    .collect();
    for (nm, dm) in worst.iter() {
        let key = if nm.starts_with("pp") {
            "pp"
        } else if nm.starts_with("po") {
            "po"
        } else if nm.starts_with("pr") {
            "pr"
        } else if nm.starts_with("ek") {
            "ek"
        } else if nm.starts_with("ev") {
            "ev"
        } else if nm == "emb" {
            "emb"
        } else {
            &nm[..1]
        };
        if let Some(e) = kind.get_mut(key) {
            *e = e.max(*dm);
        }
    }
    let mut ks: Vec<String> = kind
        .iter()
        .map(|(k, v)| format!("{k}:{v:.2}"))
        .collect();
    ks.sort();
    // prepare 512 (q0) y 2048 (pp0) contra needle-core
    let p512 = sweep_prep(eng, &cact, widx(&eng.layers[0].q), &x512).unwrap_or(-1.0);
    let p2048 =
        sweep_prep(eng, &cact, widx(&eng.layers[0].phi_pre), &x2048).unwrap_or(-1.0);
    Ok(format!(
        "sweep n={n} maxΔ={gmax:.5} mal=[{}] kind=[{}] prep=[512:{p512:.5} 2048:{p2048:.5}]",
        mal.join(" "),
        ks.join(" ")
    ))
}
