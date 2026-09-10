//! Ayudas chicas del motor: consts, encoders, buffers, slices, readback.

use wgpu::util::DeviceExt;

use super::types::Engine;

pub(crate) const SQRT_D: f32 = 22.627417; // sqrt(512)
pub(crate) const INV_SQRT_HEAD: f32 = 0.125; // 1/sqrt(64)

pub(crate) fn u16f(a: u32, b: u32, c: u32, d: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity(16);
    for x in [a, b, c, d] {
        v.extend_from_slice(&x.to_le_bytes());
    }
    v
}

pub(crate) fn u16fb(s: f32, n: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity(16);
    v.extend_from_slice(&s.to_le_bytes());
    v.extend_from_slice(&n.to_le_bytes());
    v.extend_from_slice(&[0u8; 8]);
    v
}

pub(crate) fn f32s(v: &[f32]) -> Vec<u8> {
    let mut o = Vec::with_capacity(v.len() * 4);
    for x in v {
        o.extend_from_slice(&x.to_le_bytes());
    }
    o
}

// ------------------------------------------------------------------ helpers gpu

pub(crate) fn sbuf(device: &wgpu::Device, data: &[u8]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: data,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    })
}

pub(crate) fn sbuf_zero(device: &wgpu::Device, floats: usize) -> wgpu::Buffer {
    sbuf(device, &vec![0u8; floats * 4])
}

pub(crate) fn ubuf(device: &wgpu::Device, data: &[u8]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: data,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}

/// Rebana con chequeo (los slices fuera de rango son error, no panic).
pub(crate) fn sl(v: &[f32], a: usize, b: usize, name: &str) -> Result<Vec<f32>, String> {
    if b > v.len() || a > b {
        return Err(format!("tensor {name} corto: {} < {b}", v.len()));
    }
    Ok(v[a..b].to_vec())
}

pub(crate) fn at(v: &[f32], i: usize, name: &str) -> Result<f32, String> {
    v.get(i)
        .copied()
        .ok_or_else(|| format!("tensor {name} corto en {i}"))
}

pub(crate) fn div64(n: u32) -> u32 {
    (n + 63) / 64
}

/// Lee `n` floats de un buffer storage vía staging (1 submit).
pub(crate) fn read_storage(
    eng: &Engine,
    buf: &wgpu::Buffer,
    n: usize,
) -> Result<Vec<f32>, String> {
    let mut enc = eng
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    enc.copy_buffer_to_buffer(buf, 0, &eng.staging, 0, (n * 4) as u64);
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
    let mut out = vec![0.0f32; n];
    for (i, ch) in data.chunks_exact(4).enumerate() {
        if i >= n {
            break;
        }
        out[i] = f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]);
    }
    drop(data);
    eng.staging.unmap();
    Ok(out)
}
