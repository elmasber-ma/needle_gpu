import '../../src/rust/api/gpu.dart' as rust;
import 'gpu_context.dart';

/// Attention SDPA mono-head en GPU: softmax(Q·Kᵀ/√d)·V con Q,K,V:(s,d).
/// f16 real en buffers; sin SHADER_F16 en el chip cae automático a F32.
class GpuAttention {
  Future<rust.GpuOpResult> run({
    required List<double> q,
    required List<double> k,
    required List<double> v,
    required int seq,
    required int dim,
    bool f16 = false,
  }) {
    final useF16 = f16 && GpuContext.instance.hasF16;
    return rust.gpuAttention(
      q: q,
      k: k,
      v: v,
      dims: rust.AttentionDims(s: seq, d: dim),
      useF16: useF16,
    );
  }
}
