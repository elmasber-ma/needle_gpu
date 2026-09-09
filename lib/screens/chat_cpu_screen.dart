import 'package:flutter/material.dart';

import '../services/needle_service.dart';
import '../services/tools.dart';
import 'copy_btn.dart';
import 'model_card.dart';

class _Msg {
  final bool user;
  final String text;
  final String meta;
  _Msg(this.user, this.text, [this.meta = '']);
}

/// Tab 1 · Chat: conversa con Needle on-device en CPU o GPU (WebGPU).
/// Cada respuesta muestra tokens totales, tiempo y tok/s.
class ChatCpuScreen extends StatefulWidget {
  const ChatCpuScreen({super.key});

  @override
  State<ChatCpuScreen> createState() => _ChatCpuScreenState();
}

class _ChatCpuScreenState extends State<ChatCpuScreen> {
  final _svc = NeedleService.instance;
  final _ctrl = TextEditingController();
  final _msgs = <_Msg>[];
  bool _busy = false;
  bool _gpu = false;

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  Future<void> _enviar() async {
    final q = _ctrl.text.trim();
    if (q.isEmpty || _busy) return;
    final listo = _gpu ? _svc.loadedGpu : _svc.loaded;
    if (!listo) {
      setState(() => _msgs.add(_Msg(
          false, _gpu ? 'Cargá el modelo GPU primero.' : 'Cargá el modelo primero.')));
      return;
    }
    setState(() {
      _msgs.add(_Msg(true, q));
      _ctrl.clear();
      _busy = true;
    });
    if (_gpu) {
      // GPU en vivo: cada pieza se pinta al llegar (si se cuelga, se ve
      // exactamente dónde: sin piezas = trabado en prefill/paso).
      final buf = StringBuffer();
      setState(() => _msgs.add(_Msg(false, '', 'GPU · generando…')));
      final sw = Stopwatch()..start();
      try {
        await for (final piece in _svc.runGpuStream(
            query: q, toolsJson: '[]', maxNewTokens: 256)) {
          buf.write(piece);
          if (mounted) {
            setState(() =>
                _msgs[_msgs.length - 1] = _Msg(false, buf.toString(), 'GPU · generando…'));
          }
        }
        sw.stop();
        final ms = sw.elapsedMilliseconds;
        final aprox = buf.length ~/ 4;
        if (mounted) {
          setState(() => _msgs[_msgs.length - 1] = _Msg(
              false,
              buf.toString().trim().isEmpty ? '(vacío)' : buf.toString().trim(),
              'GPU · ~$aprox tok · ${ms}ms · listo'));
        }
      } catch (e) {
        if (mounted) {
          setState(() => _msgs[_msgs.length - 1] =
              _Msg(false, buf.toString(), 'GPU ERROR: $e'));
        }
      } finally {
        if (mounted) setState(() => _busy = false);
      }
      return;
    }
    final sw = Stopwatch()..start();
    try {
      final out = await _svc.run(
        query: q,
        toolsJson: '[]',
        constrain: false,
        maxNewTokens: 256,
      );
      sw.stop();
      final ms = sw.elapsedMilliseconds;
      final tps = out.generatedTokens > 0 && ms > 0
          ? (out.generatedTokens * 1000 / ms).toStringAsFixed(1)
          : '—';
      final texto = out.text.trim().isEmpty ? '(vacío)' : out.text.trim();
      if (mounted) {
        setState(() => _msgs.add(_Msg(false, texto,
            'CPU · tok ${out.promptTokens}+${out.generatedTokens} · ${ms}ms · $tps tok/s · ${out.stop}')));
      }
    } catch (e) {
      if (mounted) setState(() => _msgs.add(_Msg(false, 'ERROR: $e')));
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
          child: Row(
            children: [
              const Text('Motor:', style: TextStyle(fontSize: 12)),
              const SizedBox(width: 8),
              SegmentedButton<bool>(
                segments: const [
                  ButtonSegment(value: false, label: Text('CPU')),
                  ButtonSegment(value: true, label: Text('GPU')),
                ],
                selected: {_gpu},
                onSelectionChanged: (s) =>
                    setState(() => _gpu = s.first),
              ),
              const SizedBox(width: 8),
              if (_gpu)
                Text(_svc.loadedGpu ? 'GPU lista' : 'GPU sin cargar',
                    style: TextStyle(
                        fontSize: 11,
                        color: _svc.loadedGpu
                            ? Colors.greenAccent
                            : Colors.orangeAccent)),
            ],
          ),
        ),
        if (_gpu)
          Padding(
            padding: const EdgeInsets.fromLTRB(10, 6, 10, 0),
            child: Row(
              children: [
                if (!_svc.loadedGpu)
                  FilledButton.icon(
                    onPressed: _busy
                        ? null
                        : () async {
                            setState(() => _busy = true);
                            try {
                              final r = await _svc.loadGpu();
                              if (mounted) {
                                setState(() => _msgs.add(_Msg(false, r)));
                              }
                            } catch (e) {
                              if (mounted) {
                                setState(() =>
                                    _msgs.add(_Msg(false, 'ERROR GPU: $e')));
                              }
                            } finally {
                              if (mounted) setState(() => _busy = false);
                            }
                          },
                    icon: const Icon(Icons.memory_rounded, size: 18),
                    label: const Text('Cargar GPU (~180 MB)',
                        style: TextStyle(fontSize: 12)),
                  ),
                if (_svc.loadedGpu)
                  OutlinedButton.icon(
                    onPressed: () {
                      _svc.unloadGpu();
                      setState(() {});
                    },
                    icon: const Icon(Icons.eject_rounded, size: 18),
                    label: const Text('Liberar GPU',
                        style: TextStyle(fontSize: 12)),
                  ),
                if (_svc.loadedGpu)
                  OutlinedButton.icon(
                    onPressed: _busy
                        ? null
                        : () async {
                            setState(() => _busy = true);
                            try {
                              final r = await _svc.gpuDiag();
                              if (mounted) {
                                setState(() =>
                                    _msgs.add(_Msg(false, r, 'GPU diag')));
                              }
                            } catch (e) {
                              if (mounted) {
                                setState(() => _msgs.add(
                                    _Msg(false, 'DIAG ERROR: $e')));
                              }
                            } finally {
                              if (mounted) setState(() => _busy = false);
                            }
                          },
                    icon: const Icon(Icons.monitor_heart_rounded,
                        size: 18),
                    label: const Text('Diag 1 paso',
                        style: TextStyle(fontSize: 12)),
                  ),
              ],
            ),
          ),
        Expanded(
          child: _msgs.isEmpty
              ? const Center(
                  child: Text('Preguntá algo — corre 100% en CPU',
                      style: TextStyle(color: Colors.grey, fontSize: 12)))
              : ListView.builder(
                  padding: const EdgeInsets.all(10),
                  itemCount: _msgs.length,
                  itemBuilder: (_, i) {
                    final m = _msgs[i];
                    return Align(
                      alignment: m.user
                          ? Alignment.centerRight
                          : Alignment.centerLeft,
                      child: Container(
                        margin: const EdgeInsets.only(bottom: 8),
                        padding: const EdgeInsets.all(10),
                        constraints: BoxConstraints(
                            maxWidth:
                                MediaQuery.of(context).size.width * .85),
                        decoration: BoxDecoration(
                          color: m.user
                              ? Colors.cyanAccent.withValues(alpha: .15)
                              : const Color(0xFF0B1220),
                          borderRadius: BorderRadius.circular(12),
                          border: Border.all(color: Colors.white12),
                        ),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            SelectableText(m.text,
                                style: const TextStyle(fontSize: 13)),
                            if (m.meta.isNotEmpty)
                              Padding(
                                padding: const EdgeInsets.only(top: 4),
                                child: Text(m.meta,
                                    style: const TextStyle(
                                        fontSize: 9,
                                        fontFamily: 'monospace',
                                        color: Colors.grey)),
                              ),
                            if (!m.user)
                              Align(
                                alignment: Alignment.centerRight,
                                child: CopyBtn(
                                    size: 14,
                                    texto: () => m.meta.isEmpty
                                        ? m.text
                                        : '${m.text}\n${m.meta}'),
                              ),
                          ],
                        ),
                      ),
                    );
                  },
                ),
        ),
        if (_busy) const LinearProgressIndicator(minHeight: 2),
        Padding(
          padding: const EdgeInsets.fromLTRB(10, 4, 10, 12),
          child: Row(
            children: [
              Expanded(
                child: TextField(
                  controller: _ctrl,
                  minLines: 1,
                  maxLines: 3,
                  onSubmitted: (_) => _enviar(),
                  decoration: const InputDecoration(
                    hintText: 'Escribí…',
                    border: OutlineInputBorder(),
                    isDense: true,
                  ),
                ),
              ),
              const SizedBox(width: 8),
              FilledButton(
                onPressed: _busy ? null : _enviar,
                child: const Text('➤'),
              ),
            ],
          ),
        ),
      ],
    );
  }
}
