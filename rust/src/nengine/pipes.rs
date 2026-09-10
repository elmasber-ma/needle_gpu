//! Pipelines WGSL: layouts por entry y construcción de todos los kernels.

use std::collections::HashMap;

pub(crate) const SHADERS: &str = include_str!("shaders.wgsl");

/// (binding, es_uniform, solo_lectura) por entry. DEBE coincidir con el WGSL.
fn spec(entry: &str) -> &'static [(u32, bool, bool)] {
    match entry {
        "rms_norm" => &[(0, false, false), (1, false, true), (2, true, true)],
        "rms_heads" => &[(0, false, false), (1, false, true), (2, true, true)],
        "cq_prepare" => &[(0, false, false), (1, false, true), (2, true, true)],
        "cq_matvec" => &[
            (0, false, false),
            (1, false, true),
            (2, false, true),
            (3, false, true),
            (4, false, true),
            (5, true, true),
        ],
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

pub(crate) fn build_all(
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
        "cq_prepare",
        "cq_matvec",
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
