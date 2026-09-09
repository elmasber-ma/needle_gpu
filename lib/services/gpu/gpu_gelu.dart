import '../../src/rust/api/gpu.dart' as rust;
import 'gpu_context.dart';

/// GELU tanh-approx en GPU. Con [f16]=true corre el kernel con array<f16>
/// real; si el chip no soporta SHADER_F16 cae automático a F32
/// (clamp local, el switch de la UI nunca se bloquea).
class GpuGelu {
  Future<rust.GpuOpResult> run(List<double> input, {bool f16 = false}) {
    final useF16 = f16 && GpuContext.instance.hasF16;
    return rust.gpuGelu(input: input, useF16: useF16);
  }
}
