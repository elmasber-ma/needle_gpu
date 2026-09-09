import 'package:flutter/material.dart';

import '../services/gpu/gpu_context.dart';
import '../services/gpu/gpu_linear.dart';
import '../services/needle_service.dart';
import '../services/needle_gpu_service.dart';
import '../services/tools.dart';
import 'copy_btn.dart';
import 'model_card.dart';

class _Fila {
  final String nombre;
  final String valor;
  _Fila(this.nombre, this.valor);
}

/// Tab 3 · Benchmark: tokens totales + tiempos, CPU vs GPU.
/// - Needle: prompt/gen tokens, ms totales, tok/s.
/// - Matmul 96×96: Dart ingenuo (CPU) vs wgpu (GPU), ms lado a lado.
class BenchScreen extends StatefulWidget {
  const BenchScreen({super.key});

  @override
  State<BenchScreen> createState() => _BenchScreenState();
}

class _BenchScreenState extends State<BenchScreen> {
  final _svc = NeedleService.instance;
  final _gsvc = NeedleGpuService.instance;
  final _filas = <_Fila>[];
  bool _busy = false;

  static const _benchPrompt =
      'Explicá en dos frases qué es un token en un modelo de lenguaje.';

  Future<void> _benchNeedle(
      Future<NeedleOut> Function() fn, String tag) async {
    final sw = Stopwatch()..start();
    final out = await fn();
    sw.stop();
    final ms = sw.elapsedMilliseconds;
    final tps = out.generatedTokens > 0 && ms > 0
        ? out.generatedTokens * 1000 / ms
        : 0.0;
    _filas.add(_Fila('$tag prompt tok', '${out.promptTokens}'));
    _filas.add(_Fila('$tag gen tok', '${out.generatedTokens}'));
    _filas.add(_Fila(
        '$tag total tok', '${out.promptTokens + out.generatedTokens}'));
    _filas.add(_Fila('$tag tiempo', '$ms ms (${out.stop})'));
    _filas.add(
        _Fila('$tag velocidad', '${tps.toStringAsFixed(1)} tok/s'));
  }

  /// Matmul ingenua en Dart (CPU) para comparar contra wgpu.
  double _matmulCpu(List<double> a, List<double> w, List<double> b,
      List<double> out, int m, int n, int k) {
    final sw = Stopwatch()..start();
    for (var i = 0; i < m; i++) {
      for (var j = 0; j < n; j++) {
        var acc = b[j];
        for (var p = 0; p < k; p++) {
          acc += a[i * k + p] * w[j * k + p];
        }
        out[i * n + j] = acc;
      }
    }
    sw.stop();
    return sw.elapsedMicroseconds / 1000.0;
  }

  Future<void> _correr() async {
    if (_busy) return;
    setState(() {
      _busy = true;
      _filas.clear();
    });
    try {
      // ---- 1. Needle CPU vs GPU: tokens + tiempo ----
      if (!_svc.loaded) {
        _filas.add(_Fila('Needle CPU', 'modelo no cargado'));
      } else {
        await _benchNeedle(
            () => _svc.run(
                  query: _benchPrompt,
                  toolsJson: '[]',
                  constrain: false,
                  maxNewTokens: 128,
                ),
            'Needle CPU');
      }
      if (!_gsvc.loadedGpu) {
        _filas.add(_Fila('Needle GPU', 'motor GPU sin cargar (tab Chat)'));
      } else {
        await _benchNeedle(
            () => _gsvc.run(
                  query: _benchPrompt,
                  toolsJson: '[]',
                  maxNewTokens: 128,
                ),
            'Needle GPU');
      }
      // ---- 2. Matmul CPU (Dart) vs GPU (wgpu) ----
      const m = 96, k = 96, n = 96;
      final a = List<double>.generate(m * k, (i) => (i % 13) / 13.0);
      final w = List<double>.generate(n * k, (i) => (i % 7) / 7.0);
      final b = List<double>.filled(n, 0.1);
      final out = List<double>.filled(m * n, 0);
      final msCpu = _matmulCpu(a, w, b, out, m, n, k);
      _filas.add(_Fila('Matmul CPU 96×96', '${msCpu.toStringAsFixed(2)} ms'));
      if (GpuContext.instance.ready) {
        try {
          final r = await GpuLinear().run(
            input: a,
            weights: w,
            bias: b,
            m: m,
            n: n,
            k: k,
            f16: GpuContext.instance.hasF16,
          );
          _filas.add(_Fila(
              'Matmul GPU 96×96 ${GpuContext.instance.hasF16 ? "F16" : "F32"}',
              '${r.elapsedMs.toStringAsFixed(2)} ms'));
          final speed = r.elapsedMs > 0 ? msCpu / r.elapsedMs : 0.0;
          _filas.add(
              _Fila('Speedup GPU/CPU', '${speed.toStringAsFixed(1)}×'));
        } catch (e) {
          _filas.add(_Fila('Matmul GPU', 'ERROR: $e'));
        }
      } else {
        _filas.add(_Fila('Matmul GPU', 'GPU no inicializada (ver tab GPU)'));
      }
    } catch (e) {
      _filas.add(_Fila('ERROR', '$e'));
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        const ModelCard(),
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 10),
          child: Wrap(
            spacing: 8,
            children: [
              FilledButton.icon(
                onPressed: _busy ? null : _correr,
                icon: const Icon(Icons.speed_rounded, size: 18),
                label: const Text('Correr benchmark'),
              ),
              OutlinedButton.icon(
                onPressed: (_busy || !_gsvc.loadedGpu)
                    ? null
                    : () async {
                        setState(() => _busy = true);
                        try {
                          final r =
                              await _gsvc.parity('Hola');
                          if (mounted) {
                            setState(() =>
                                _filas.insert(0, _Fila('Paridad', r)));
                          }
                        } catch (e) {
                          if (mounted) {
                            setState(() => _filas.insert(
                                0, _Fila('Paridad ERROR', '$e')));
                          }
                        } finally {
                          if (mounted) setState(() => _busy = false);
                        }
                      },
                icon: const Icon(Icons.compare_arrows_rounded, size: 18),
                label: const Text('Paridad CPU/GPU',
                    style: TextStyle(fontSize: 12)),
              ),
            ],
          ),
        ),
        if (_busy)
          const Padding(
            padding: EdgeInsets.all(12),
            child: CircularProgressIndicator(),
          ),
        Expanded(
          child: ListView.builder(
            padding: const EdgeInsets.all(10),
            itemCount: _filas.length,
            itemBuilder: (_, i) {
              final f = _filas[i];
              return Card(
                color: const Color(0xFF0B1220),
                margin: const EdgeInsets.only(bottom: 6),
                child: ListTile(
                  dense: true,
                  title: Text(f.nombre,
                      style: const TextStyle(fontSize: 12)),
                  subtitle: Text(f.valor,
                      style: const TextStyle(
                          fontSize: 12,
                          fontFamily: 'monospace',
                          color: Colors.greenAccent)),
                  trailing:
                      CopyBtn(texto: () => '${f.nombre}: ${f.valor}'),
                ),
              );
            },
          ),
        ),
      ],
    );
  }
}
