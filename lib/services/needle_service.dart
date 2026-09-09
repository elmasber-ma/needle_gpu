import 'dart:convert';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';

import '../src/rust/api/needle.dart' as rust;
import '../src/rust/api/nengine.dart' as ng;

/// Needle v2 on-device: descarga del .cact (13.7 MB), carga del motor
/// y tool-calling local (query + tools JSON → llamada JSON).
class NeedleOut {
  final String text;
  final String? toolCall;
  final String? thinking;
  final String stop;
  final int promptTokens;
  final int generatedTokens;
  NeedleOut({
    required this.text,
    required this.toolCall,
    required this.thinking,
    required this.stop,
    required this.promptTokens,
    required this.generatedTokens,
  });

  static NeedleOut _fromRust(rust.NeedleOut o) => NeedleOut(
        text: o.text,
        toolCall: o.toolCall,
        thinking: o.thinking,
        stop: o.stop,
        promptTokens: o.promptTokens,
        generatedTokens: o.generatedTokens,
      );
}

class NeedleService extends ChangeNotifier {
  NeedleService._();
  static final NeedleService instance = NeedleService._();

  static const modelUrl =
      'https://huggingface.co/Cactus-Compute/needle2/resolve/main/needle2.cact';
  static const modelBytes = 13.7 * 1024 * 1024;

  String? _modelPath;
  bool _busy = false;
  bool _loadedV2 = false;
  double _progress = 0;

  bool get busy => _busy;
  double get progress => _progress;
  bool get downloaded => _modelPath != null;

  /// El motor Rust arranca sin modelo tras cada reinicio de la app,
  /// así que el estado real vive acá y se actualiza con load/unload.
  bool get loaded => _loadedV2;
  bool _loadedGpu = false;
  bool get loadedGpu => _loadedGpu;
  String? get modelPath => _modelPath;

  Future<String> get dirPath async {
    final d = await getApplicationSupportDirectory();
    final dir = Directory('${d.path}/needle');
    if (!dir.existsSync()) dir.createSync(recursive: true);
    return dir.path;
  }

  Future<void> refresh() async {
    final dir = await dirPath;
    final f = File('$dir/needle2.cact');
    _modelPath = f.existsSync() ? f.path : null;
    notifyListeners();
  }

  /// Baja un archivo remoto a [dest] reportando progreso (0..1).
  Future<void> _fetch(String url, String dest, int approxBytes) async {
    final client = http.Client();
    try {
      final res = await client.send(http.Request('GET', Uri.parse(url)));
      if (res.statusCode != 200) throw 'HTTP ${res.statusCode} en $url';
      final total = res.contentLength ?? approxBytes;
      final sink = File(dest).openWrite();
      var got = 0;
      await for (final chunk in res.stream) {
        got += chunk.length;
        sink.add(chunk);
        _progress = total > 0 ? got / total : 0;
        notifyListeners();
      }
      await sink.close();
    } finally {
      client.close();
    }
  }

  /// Descarga el .cact de v2 desde HuggingFace.
  Future<String> downloadModel() async {
    if (_busy) throw 'ya hay una descarga en curso';
    _busy = true;
    _progress = 0;
    notifyListeners();
    try {
      final dir = await dirPath;
      final tmp = '$dir/needle2.cact.tmp';
      await _fetch(modelUrl, tmp, modelBytes.toInt());
      final finalPath = '$dir/needle2.cact';
      File(tmp).renameSync(finalPath);
      _modelPath = finalPath;
      return finalPath;
    } finally {
      _busy = false;
      notifyListeners();
    }
  }

  /// Carga el modelo ya descargado en el motor Rust.
  Future<String> load() async {
    await refresh();
    final p = _modelPath;
    if (p == null) throw 'primero descargá el modelo (13.7 MB)';
    final r = await rust.needleLoad(path: p);
    _loadedV2 = true;
    notifyListeners();
    return r;
  }

  /// Carga el mismo .cact en el motor GPU (forward WGSL, ~180 MB).
  Future<String> loadGpu() async {
    await refresh();
    final p = _modelPath;
    if (p == null) throw 'primero descargá el modelo (13.7 MB)';
    final r = await ng.needleGpuLoad(path: p);
    _loadedGpu = true;
    notifyListeners();
    return r;
  }

  /// Genera con el forward en GPU. Sin constrain en fase 1.
  Future<NeedleOut> runGpu({
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
    return NeedleOut._fromRust(r);
  }

  void unloadGpu() {
    ng.needleGpuUnload();
    _loadedGpu = false;
    notifyListeners();
  }

  Future<NeedleOut> run({
    required String query,
    required String toolsJson,
    bool constrain = true,
    int maxNewTokens = 128,
    double temperature = 0.0,
    int seed = 0,
  }) async {
    final r = await rust.needleRun(
      query: query,
      toolsJson: toolsJson,
      constrain: constrain,
      maxNewTokens: maxNewTokens,
      temperature: temperature,
      seed: BigInt.from(seed),
    );
    return NeedleOut._fromRust(r);
  }

  /// Confianza del head propio para la última respuesta.
  Future<double> confidence({
    required String query,
    required String toolsJson,
    required String completion,
  }) =>
      rust.needleConfidence(
          query: query, toolsJson: toolsJson, completion: completion);

  void unload() {
    rust.needleUnload();
    _loadedV2 = false;
    notifyListeners();
  }

  /// Pretty-print defensivo del payload JSON de la tool call.
  static String pretty(String raw) {
    try {
      return const JsonEncoder.withIndent('  ').convert(jsonDecode(raw));
    } catch (_) {
      return raw;
    }
  }
}
