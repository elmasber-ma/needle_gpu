import '../../src/rust/api/gpu.dart' as rust;
import 'gpu_context.dart';

/// Linear Y = X·Wᵀ + b en GPU: X:(m,k), W:(n,k), bias:(n).
/// f16 real en buffers (mitad de memoria), acumulación f32.
/// Sin SHADER_F16 en el chip cae automático a F32.
class GpuLinear {
  Future<rust.GpuOpResult> run({
    required List<double> input,
    required List<double> weights,
    required List<double> bias,
    required int m,
    required int n,
    required int k,
    bool f16 = false,
  }) {
    final useF16 = f16 && GpuContext.instance.hasF16;
    return rust.gpuLinear(
      input: input,
      weights: weights,
      bias: bias,
      dims: rust.LinearDims(m: m, n: n, k: k),
      useF16: useF16,
    );
  }
}
