//! Infraestructura GPU compartida: singleton device/queue, helpers de
//! buffers/pipelines y lectura de vuelta al CPU.

pub mod attention;
pub mod gelu;
pub mod lab;
pub mod linear;

use wgpu::util::DeviceExt;

pub struct GpuState {
    pub device: Option<wgpu::Device>,
    pub queue: Option<wgpu::Queue>,
    pub has_f16: bool,
    pub info: String,
}

static STATE: std::sync::Mutex<GpuState> = std::sync::Mutex::new(GpuState {
    device: None,
    queue: None,
    has_f16: false,
    info: String::new(),
});

/// Adapter + device con SHADER_F16 si está disponible. false si no hay GPU.
pub async fn init() -> bool {
    {
        let st = STATE.lock().unwrap();
        if st.device.is_some() {
            return true;
        }
    }
    let instance = wgpu::Instance::default();
    let adapter = match instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
    {
        Ok(a) => a,
        Err(_) => return false,
    };

    let wanted = wgpu::Features::SHADER_F16;
    let has_f16 = adapter.features().contains(wanted);

    let desc = wgpu::DeviceDescriptor {
        label: Some("needle_gpu"),
        required_features: if has_f16 { wanted } else { wgpu::Features::empty() },
        required_limits: wgpu::Limits::default(),
        memory_hints: wgpu::MemoryHints::default(),
        trace: wgpu::Trace::Off,
    };
    let (device, queue) = match adapter.request_device(&desc).await {
        Ok(d) => d,
        Err(_) => return false,
    };

    let info = adapter.get_info();
    let mut st = STATE.lock().unwrap();
    st.device = Some(device);
    st.queue = Some(queue);
    st.has_f16 = has_f16;
    st.info = format!("{} | {:?} | f16={}", info.name, info.backend, has_f16);
    true
}

pub fn has_f16() -> bool {
    STATE.lock().unwrap().has_f16
}

pub fn info() -> String {
    STATE.lock().unwrap().info.clone()
}

/// (device, queue) o error legible si no se inicializó.
pub fn ctx() -> Result<(wgpu::Device, wgpu::Queue), String> {
    let st = STATE.lock().unwrap();
    match (&st.device, &st.queue) {
        (Some(d), Some(q)) => Ok((d.clone(), q.clone())),
        _ => Err("GPU sin inicializar (llamá gpuInit primero)".to_string()),
    }
}

/// Valida WGSL con naga y retorna el diagnóstico completo si falla.
pub fn check_wgsl(code: &str) -> Result<(), String> {
    wgpu::naga::front::wgsl::parse_str(code)
        .map(|_| ())
        .map_err(|e| format!("error compilando WGSL:\n{e}"))
}

// ---------------- helpers de construcción ----------------

pub fn storage_buf(
    device: &wgpu::Device,
    bytes: &[u8],
    read_only: bool,
) -> wgpu::Buffer {
    let mut usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
    if !read_only {
        usage |= wgpu::BufferUsages::COPY_DST;
    }
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytes,
        usage,
    })
}

pub fn uniform_buf(device: &wgpu::Device, bytes: &[u8]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"),
        contents: bytes,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}

/// Pipeline compute desde código WGSL con los bindings indicados.
/// Cada entrada: (binding, es_uniform, storage_solo_lectura).
/// El flag tiene que COINCIDIR con la declaración del WGSL (`var<storage,
/// read>` vs `read_write`): si difiere, wgpu rechaza el pipeline.
pub fn build_pipeline(
    device: &wgpu::Device,
    code: &str,
    bindings: &[(u32, bool, bool)],
) -> Result<(wgpu::BindGroupLayout, wgpu::ComputePipeline), String> {
    check_wgsl(code)?;
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("kernel"),
        source: wgpu::ShaderSource::Wgsl(code.into()),
    });
    let entries: Vec<wgpu::BindGroupLayoutEntry> = bindings
        .iter()
        .map(|(i, is_uniform, read_only)| wgpu::BindGroupLayoutEntry {
            binding: *i,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: if *is_uniform {
                wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                }
            } else {
                wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: *read_only },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                }
            },
            count: None,
        })
        .collect();
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &entries,
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    Ok((layout, pipeline))
}

/// Dispatch + submit. Retorna el instante para medir afuera si hace falta.
pub fn dispatch(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    layout: &wgpu::BindGroupLayout,
    resources: Vec<(u32, wgpu::BindingResource<'_>)>,
    workgroups: (u32, u32, u32),
) {
    let entries: Vec<wgpu::BindGroupEntry> = resources
        .into_iter()
        .map(|(binding, resource)| wgpu::BindGroupEntry { binding, resource })
        .collect();
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout,
        entries: &entries,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        cpass.set_pipeline(pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(workgroups.0.max(1), workgroups.1.max(1), workgroups.2.max(1));
    }
    queue.submit(Some(encoder.finish()));
}

/// Lee un buffer completo al CPU vía staging map.
pub fn read_back(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &wgpu::Buffer,
    len: u64,
) -> Result<Vec<u8>, String> {
    let stage = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: len,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(src, 0, &stage, 0, len);
    queue.submit(Some(encoder.finish()));

    let (tx, rx) = std::sync::mpsc::channel();
    stage.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::PollType::Wait);
    rx.recv()
        .map_err(|_| "readback cancelado".to_string())?
        .map_err(|e| format!("map error: {e:?}"))?;
    let data = stage.get_mapped_range(..).to_vec();
    stage.unmap();
    Ok(data)
}

// ---------------- conversión f16 ----------------

pub fn to_u16_bits(v: &[f32]) -> Vec<u16> {
    v.iter().map(|x| half::f16::from_f32(*x).to_bits()).collect()
}

pub fn from_u16_bits(v: &[u16]) -> Vec<f32> {
    v.iter()
        .map(|x| half::f16::from_bits(*x).to_f32())
        .collect()
}

pub fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_ne_bytes()).collect()
}

pub fn u16_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_ne_bytes()).collect()
}

pub fn bytes_to_f32(bytes: &[u8], as_f16: bool) -> Vec<f32> {
    if as_f16 {
        let bits: Vec<u16> = bytes
            .chunks_exact(2)
            .map(|c| u16::from_ne_bytes([c[0], c[1]]))
            .collect();
        from_u16_bits(&bits)
    } else {
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
}

/// Params uniform de N×u32 (padded a múltiplo de 16 bytes).
pub fn params_bytes(p: &[u32]) -> Vec<u8> {
    let mut v: Vec<u8> = p.iter().flat_map(|x| x.to_ne_bytes()).collect();
    while v.len() % 16 != 0 {
        v.push(0);
    }
    v
}
