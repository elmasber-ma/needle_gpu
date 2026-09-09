import 'package:flutter/material.dart';

import '../services/needle_service.dart';

/// Tarjeta de estado del modelo: descargar → cargar → listo.
/// Reusada en Chat, Benchmark y Tools.
class ModelCard extends StatefulWidget {
  const ModelCard({super.key});

  @override
  State<ModelCard> createState() => _ModelCardState();
}

class _ModelCardState extends State<ModelCard> {
  final _svc = NeedleService.instance;
  String _msg = '';

  @override
  void initState() {
    super.initState();
    _svc.addListener(_refresh);
    _svc.refresh();
  }

  @override
  void dispose() {
    _svc.removeListener(_refresh);
    super.dispose();
  }

  void _refresh() {
    if (mounted) setState(() {});
  }

  Future<void> _do(Future<String> Function() work) async {
    try {
      setState(() => _msg = '…');
      final r = await work();
      if (mounted) setState(() => _msg = r);
    } catch (e) {
      if (mounted) setState(() => _msg = 'ERROR: $e');
    }
  }

  @override
  Widget build(BuildContext context) {
    final ok = _svc.loaded;
    return Container(
      width: double.infinity,
      margin: const EdgeInsets.all(10),
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: (ok ? Colors.greenAccent : Colors.orangeAccent)
            .withValues(alpha: .07),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(
            color: (ok ? Colors.greenAccent : Colors.orangeAccent)
                .withValues(alpha: .4)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            ok
                ? 'Needle 45M listo (CPU)'
                : _svc.downloaded
                    ? 'Modelo descargado, sin cargar'
                    : 'Modelo no descargado (13.7 MB)',
            style: const TextStyle(fontSize: 12),
          ),
          if (_svc.busy)
            Padding(
              padding: const EdgeInsets.only(top: 6),
              child: LinearProgressIndicator(value: _svc.progress),
            ),
          if (_msg.isNotEmpty)
            Padding(
              padding: const EdgeInsets.only(top: 4),
              child: Text(_msg,
                  style:
                      const TextStyle(fontSize: 10, color: Colors.grey)),
            ),
          const SizedBox(height: 6),
          Wrap(
            spacing: 8,
            children: [
              if (!_svc.downloaded)
                FilledButton.icon(
                  onPressed: _svc.busy ? null : () => _do(_svc.downloadModel),
                  icon: const Icon(Icons.download_rounded, size: 18),
                  label: const Text('Descargar',
                      style: TextStyle(fontSize: 12)),
                ),
              if (_svc.downloaded && !ok)
                FilledButton.icon(
                  onPressed: _svc.busy ? null : () => _do(_svc.load),
                  icon: const Icon(Icons.memory_rounded, size: 18),
                  label: const Text('Cargar',
                      style: TextStyle(fontSize: 12)),
                ),
              if (ok)
                OutlinedButton.icon(
                  onPressed: () {
                    _svc.unload();
                    setState(() => _msg = 'motor liberado');
                  },
                  icon: const Icon(Icons.eject_rounded, size: 18),
                  label: const Text('Liberar',
                      style: TextStyle(fontSize: 12)),
                ),
            ],
          ),
        ],
      ),
    );
  }
}
