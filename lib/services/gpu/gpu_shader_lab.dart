import '../../src/rust/api/gpu.dart' as rust;

/// Shader Lab: corre WGSL arbitrario en caliente estilo celda de Colab.
///
/// Contrato del shader:
///   binding 0: `var<storage, read_write> data: array<f32>` in-place
///   binding 1: `var<uniform> params` struct {x,y,z: f32, n: u32}
///   entry point `main`
class GpuShaderLab {
  Future<String> template() => rust.gpuLabTemplate();

  Future<rust.ShaderRunResult> run({
    required String code,
    required List<double> input,
    required int dispatchX,
    required int dispatchY,
    required int dispatchZ,
    double paramX = 0,
    double paramY = 0,
    double paramZ = 0,
  }) {
    return rust.gpuRunWgsl(
      code: code,
      input: input,
      dispatchX: dispatchX,
      dispatchY: dispatchY,
      dispatchZ: dispatchZ,
      paramX: paramX,
      paramY: paramY,
      paramZ: paramZ,
    );
  }
}
