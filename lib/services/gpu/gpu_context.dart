import '../../src/rust/api/gpu.dart' as rust;

/// Contexto GPU global: init una sola vez, expone info del adaptador y
/// si el chip soporta f16 en shader (Features::SHADER_F16).
class GpuContext {
  static final GpuContext instance = GpuContext._();

  GpuContext._();

  bool ready = false;
  String info = '';
  bool hasF16 = false;

  /// true si hay device disponible. false = sin backend GPU.
  Future<bool> init() async {
    if (ready) return true;
    try {
      ready = await rust.gpuInit();
      if (ready) {
        info = await rust.gpuInfo();
        hasF16 = await rust.gpuHasF16();
      }
    } catch (_) {
      ready = false;
    }
    return ready;
  }

  /// Re-lee info tras un init exitoso.
  Future<void> refresh() async {
    if (!ready) return;
    info = await rust.gpuInfo();
    hasF16 = await rust.gpuHasF16();
  }
}
