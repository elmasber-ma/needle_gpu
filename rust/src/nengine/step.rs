//! Dispatch de kernels y el forward de un token.

use super::help::{div64, f32s, u16f, u16fb, INV_SQRT_HEAD};
use super::types::{Engine, Wmat};

// ------------------------------------------------------------------ dispatch

impl Engine {
    pub(crate) fn ub(&self, bytes: &[u8]) -> wgpu::Buffer {
        super::help::ubuf(&self.device, bytes)
    }

    pub(crate) fn bg(&self, entry: &str, bufs: &[&wgpu::Buffer]) -> wgpu::BindGroup {
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

    pub(crate) fn run(
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

    /// Rotación Hadamard de la activación (una vez por activación; la
    /// comparten todas las projs que la leen, como prepare_input del CPU).
    pub(crate) fn prepare(
        &self,
        enc: &mut wgpu::CommandEncoder,
        xh: &wgpu::Buffer,
        x: &wgpu::Buffer,
        in_feat: u32,
    ) {
        let ng = in_feat / 128;
        let u = self.ub(&u16f(in_feat, ng, 0, 0));
        self.run(enc, "cq_prepare", &[xh, x, &u], ng);
    }

    /// Proyección y = W @ x: ruta CQ empaquetada o f32 según el tensor.
    /// Para CQ, xh debe venir de prepare() sobre la misma activación.
    pub(crate) fn proj(
        &self,
        enc: &mut wgpu::CommandEncoder,
        y: &wgpu::Buffer,
        w: &Wmat,
        x: &wgpu::Buffer,
        xh: &wgpu::Buffer,
        rows: u32,
        in_feat: u32,
    ) {
        match w {
            Wmat::F32 { b, .. } => self.matvec(enc, y, b, x, rows, in_feat),
            Wmat::Cq(c) => {
                let mut v = Vec::with_capacity(32);
                v.extend_from_slice(&rows.to_le_bytes());
                v.extend_from_slice(&0u32.to_le_bytes());
                v.extend_from_slice(&c.ngroups.to_le_bytes());
                v.extend_from_slice(&c.group.to_le_bytes());
                v.extend_from_slice(&c.bits.to_le_bytes());
                v.extend_from_slice(&c.row_u32.to_le_bytes());
                v.extend_from_slice(&c.is_ternary.to_le_bytes());
                v.extend_from_slice(&c.nlevels.to_le_bytes());
                let u = self.ub(&v);
                self.run(
                    enc,
                    "cq_matvec",
                    &[y, &c.packed, &c.norms, xh, &c.levels, &u],
                    div64(rows),
                );
            }
        }
    }

    pub(crate) fn norm(
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
pub(crate) fn step_token(eng: &Engine, token: u32, pos: usize) -> Result<Vec<f32>, String> {
    let d = eng.d as u32;
    let pos_u = pos as u32;
    let sqrt_d = (eng.d as f32).sqrt();

    // embedding lookup en CPU (1 fila, CQ → dequant solo esa fila)
    let x0 = eng.emb_cpu.row(token as usize, eng.d, sqrt_d)?;
    eng.queue.write_buffer(&eng.act.h, 0, &f32s(&x0));

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
        eng.prepare(&mut enc, &a.xh_e, &a.e, 512);
        eng.proj(&mut enc, &a.ek, &s.key, &a.e, &a.xh_e, 512, 512);
        eng.proj(&mut enc, &a.ev, &s.value, &a.e, &a.xh_e, 512, 512);
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
        eng.prepare(&mut enc, &a.xh_nx, &a.nx, 2048);
        eng.proj(&mut enc, &a.s4, &l.phi_pre, &a.nx, &a.xh_nx, 4, 2048);
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
        eng.prepare(&mut enc, &a.xh_h, &a.h, 512);
        eng.proj(&mut enc, &a.q, &l.q, &a.h, &a.xh_h, 512, 512);
        eng.proj(&mut enc, &a.k, &l.k, &a.h, &a.xh_h, 256, 512);
        eng.proj(&mut enc, &a.v, &l.v, &a.h, &a.xh_h, 256, 512);
        // gate va a su propio buffer: attention pisa a.o después.
        eng.proj(&mut enc, &a.gt, &l.g, &a.h, &a.xh_h, 512, 512);
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
        eng.prepare(&mut enc, &a.xh_o, &a.o, 512);
        eng.proj(&mut enc, &a.ar, &l.o, &a.o, &a.xh_o, 512, 512);
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

        // mHC-post (reusa xh_nx ya preparado)
        eng.proj(&mut enc, &a.s16, &l.phi_res, &a.nx, &a.xh_nx, 16, 2048);
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
        eng.proj(&mut enc, &a.s4, &l.phi_post, &a.nx, &a.xh_nx, 4, 2048);
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
    eng.prepare(&mut enc, &a.xh_lm, &a.lm, 512);
    eng.proj(
        &mut enc,
        &a.logits,
        &eng.emb,
        &a.lm,
        &a.xh_lm,
        eng.vocab as u32,
        512,
    );
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
