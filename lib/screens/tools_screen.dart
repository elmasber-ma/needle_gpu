import 'package:flutter/material.dart';

import '../services/needle_service.dart';
import '../services/tools.dart';
import 'copy_btn.dart';
import 'model_card.dart';

/// Tab 4 · Tools: loop agéntico con Needle on-device.
/// query → tool_call → ejecuta → realimenta, hasta respuesta final.
/// Tools: leer/escribir/listar archivos, buscar web, usar GPU.
class ToolsScreen extends StatefulWidget {
  const ToolsScreen({super.key});

  @override
  State<ToolsScreen> createState() => _ToolsScreenState();
}

class _ToolsScreenState extends State<ToolsScreen> {
  final _ctrl = TextEditingController();
  final _log = <String>[];
  String _respuesta = '';
  bool _busy = false;

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  void _say(String s) {
    if (mounted) {
      setState(() {
        _log.add(s);
        if (_log.length > 60) _log.removeAt(0);
      });
    }
  }

  Future<void> _ejecutar() async {
    final q = _ctrl.text.trim();
    if (q.isEmpty || _busy) return;
    if (!NeedleService.instance.loaded) {
      _say('ERROR: cargá el modelo primero.');
      return;
    }
    setState(() {
      _busy = true;
      _respuesta = '';
    });
    _say('» $q');
    try {
      final resp = await agentLoop(
        query: q,
        maxIters: 6,
        onStep: (s) => _say(s.kind == 'tool' ? '[tool] ${s.text}' : s.text),
      );
      if (mounted) setState(() => _respuesta = resp);
    } catch (e) {
      _say('ERROR: $e');
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
          child: Text(
            'Tools: ${kTools.map((t) => t.name).join(' · ')}',
            style: const TextStyle(fontSize: 10, color: Colors.grey),
          ),
        ),
        const SizedBox(height: 6),
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 10),
          child: Row(
            children: [
              Expanded(
                child: TextField(
                  controller: _ctrl,
                  minLines: 1,
                  maxLines: 3,
                  onSubmitted: (_) => _ejecutar(),
                  decoration: const InputDecoration(
                    hintText: 'Pedí algo (ej: listá mis documentos)…',
                    border: OutlineInputBorder(),
                    isDense: true,
                  ),
                ),
              ),
              const SizedBox(width: 8),
              FilledButton(
                onPressed: _busy ? null : _ejecutar,
                child: const Text('➤'),
              ),
            ],
          ),
        ),
        if (_respuesta.isNotEmpty)
          Container(
            width: double.infinity,
            margin: const EdgeInsets.fromLTRB(10, 8, 10, 0),
            padding: const EdgeInsets.all(10),
            decoration: BoxDecoration(
              color: Colors.greenAccent.withValues(alpha: .07),
              borderRadius: BorderRadius.circular(10),
              border:
                  Border.all(color: Colors.greenAccent.withValues(alpha: .4)),
            ),
            child: SelectableText(_respuesta,
                style: const TextStyle(fontSize: 13)),
          ),
        if (_busy) const LinearProgressIndicator(minHeight: 2),
        Expanded(
          child: Container(
            width: double.infinity,
            margin: const EdgeInsets.fromLTRB(10, 8, 10, 12),
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: Colors.black,
              borderRadius: BorderRadius.circular(8),
              border: Border.all(color: Colors.grey[800]!),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Align(
                  alignment: Alignment.centerRight,
                  child: CopyBtn(
                      texto: () => _log.isEmpty
                          ? '· log del agente ·'
                          : _log.join('\n')),
                ),
                Expanded(
                  child: SingleChildScrollView(
                    reverse: true,
                    child: SelectableText(
                      _log.isEmpty
                          ? '· log del agente ·'
                          : _log.join('\n'),
                      style: const TextStyle(
                          fontSize: 11, fontFamily: 'monospace'),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }
}
