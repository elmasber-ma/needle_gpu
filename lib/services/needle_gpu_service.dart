import 'package:flutter/foundation.dart';

import '../src/rust/api/nengine.dart' as ng;
import 'needle_service.dart';

/// Motor GPU (WebGPU/WGSL) separado del CPU.
/// El modelo (.cact) y la descarga viven en [NeedleService]; acá solo
/// carga GPU, generación GPU (bloqueante y stream) y diagnósticos.
class NeedleGpuService extends ChangeNotifier {
  NeedleGpuService._();
  static final NeedleGpuService instance = NeedleGpuService._();

  bool _loadedGpu = false;
  bool get loadedGpu => _loadedGpu;

  /// Carga el mismo .cact en el motor GPU (forward WGSL, ~180 MB).
  Future<String> load() async {
    final cpu = NeedleService.instance;
    await cpu.refresh();
    final p = cpu.modelPath;
    if (p == null) throw 'primero descargá el modelo (13.7 MB)';
    final r = await ng.needleGpuLoad(path: p);
    _loadedGpu = true;
    notifyListeners();
    return r;
  }

  /// Genera con el forward en GPU. Sin constrain en fase 1.
  Future<NeedleOut> run({
    required String query,
    required String toolsJson,
    int maxNewTokens = 128,
    double temperature = 0.0,
    int seed = 0,
  }) async {
    final r = await ng.needleGpuRun(
      query: query,
      toolsJson: toolsJson,
      maxNewTokens: maxNewTokens,
      temperature: temperature,
      seed: BigInt.from(seed),
    );
    return NeedleOut.fromRust(r);
  }

  /// Generación GPU en vivo: emite cada pieza de texto por el stream.
  Stream<String> runStream({
    required String query,
    required String toolsJson,
    int maxNewTokens = 256,
    double temperature = 0.0,
    int seed = 0,
  }) {
    return ng.needleGpuRunStream(
      query: query,
      toolsJson: toolsJson,
      maxNewTokens: maxNewTokens,
      temperature: temperature,
      seed: BigInt.from(seed),
    );
  }

  void unload() {
    ng.needleGpuUnload();
    _loadedGpu = false;
    notifyListeners();
  }

  /// Un solo paso cronometrado (diagnóstico cuelgue vs lentitud).
  Future<String> diag() async => ng.needleGpuDiag();

  /// Paridad CPU vs GPU en 8 tokens (detecta divergencia numérica).
  Future<String> parity(String query) async =>
      ng.needleGpuParity(query: query);
}
