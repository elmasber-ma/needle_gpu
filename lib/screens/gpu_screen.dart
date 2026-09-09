import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../services/gpu/gpu_attention.dart';
import '../services/gpu/gpu_context.dart';
import '../services/gpu/gpu_gelu.dart';
import '../services/gpu/gpu_linear.dart';
import '../services/gpu/gpu_shader_lab.dart';
import 'copy_btn.dart';

class _OpRow {
  final String nombre;
  final bool ok;
  final String detalle;
  _OpRow(this.nombre, this.ok, this.detalle);
}

/// Tab 2 · GPU Web: bloques del transformer en wgpu + mini lab WGSL.
/// Todo con ms medidos en el dispositivo.
class GpuScreen extends StatefulWidget {
  const GpuScreen({super.key});

  @override
  State<GpuScreen> createState() => _GpuScreenState();
}

class _GpuScreenState extends State<GpuScreen> {
  final _ctx = GpuContext.instance;
  final _code = TextEditingController();
  final _rows = <_OpRow>[];
  bool _initing = false;
  bool _busy = false;

  @override
  void initState() {
    super.initState();
    _init();
  }

  @override
  void dispose() {
    _code.dispose();
    super.dispose();
  }

  Future<void> _init() async {
    setState(() => _initing = true);
    await _ctx.init();
    try {
      _code.text = await GpuShaderLab().template();
    } catch (_) {}
    if (mounted) setState(() => _initing = false);
  }

  void _add(String n, bool ok, String d) {
    if (mounted) setState(() => _rows.insert(0, _OpRow(n, ok, d)));
  }

  List<double> _rand(int n) {
    final r = math.Random(42);
    return List.generate(n, (_) => r.nextDouble() * 4 - 2);
  }

  Future<void> _gelu() async {
    if (_busy || !_ctx.ready) return;
    setState(() => _busy = true);
    try {
      const n = 65536;
      final r = await GpuGelu().run(_rand(n), f16: _ctx.hasF16);
      _add('GELU ${_ctx.hasF16 ? "F16" : "F32"}', true,
          'n=$n · ${r.elapsedMs.toStringAsFixed(2)} ms · ${r.data.take(4).map((v) => v.toStringAsFixed(3)).join(' ')}');
    } catch (e) {
      _add('GELU', false, '$e');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _linear() async {
    if (_busy || !_ctx.ready) return;
    setState(() => _busy = true);
    try {
      const m = 128, k = 128, n = 128;
      final r = await GpuLinear().run(
        input: _rand(m * k),
        weights: _rand(n * k),
        bias: _rand(n),
        m: m,
        n: n,
        k: k,
        f16: _ctx.hasF16,
      );
      _add('LINEAR ${_ctx.hasF16 ? "F16" : "F32"}', true,
          '($m×$k)·($k×$n) · ${r.elapsedMs.toStringAsFixed(2)} ms');
    } catch (e) {
      _add('LINEAR', false, '$e');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _attn() async {
    if (_busy || !_ctx.ready) return;
    setState(() => _busy = true);
    try {
      const s = 64, d = 32;
      final r = await GpuAttention().run(
        q: _rand(s * d),
        k: _rand(s * d),
        v: _rand(s * d),
        seq: s,
        dim: d,
        f16: _ctx.hasF16,
      );
      _add('ATTN ${_ctx.hasF16 ? "F16" : "F32"}', true,
          'seq=$s dim=$d · ${r.elapsedMs.toStringAsFixed(2)} ms');
    } catch (e) {
      _add('ATTN', false, '$e');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _lab() async {
    if (_busy || !_ctx.ready) return;
    setState(() => _busy = true);
    try {
      const n = 1024;
      final r = await GpuShaderLab().run(
        code: _code.text,
        input: List.generate(n, (i) => (i % 17).toDouble() - 8),
        dispatchX: (n + 63) ~/ 64,
        dispatchY: 1,
        dispatchZ: 1,
        paramX: 2.0,
      );
      _add('LAB ${r.mode}', r.ok,
          r.ok ? '${r.elapsedMs.toStringAsFixed(2)} ms · ${r.data.take(4).map((v) => v.toStringAsFixed(3)).join(' ')}' : r.error);
    } catch (e) {
      _add('LAB', false, '$e');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final color = !_ctx.ready
        ? Colors.redAccent
        : (_ctx.hasF16 ? Colors.greenAccent : Colors.orangeAccent);
    return Column(
      children: [
        Container(
          width: double.infinity,
          margin: const EdgeInsets.all(10),
          padding: const EdgeInsets.all(10),
          decoration: BoxDecoration(
            color: color.withValues(alpha: .07),
            borderRadius: BorderRadius.circular(10),
            border: Border.all(color: color.withValues(alpha: .4)),
          ),
          child: _initing
              ? const Text('inicializando GPU…',
                  style: TextStyle(fontSize: 11))
              : Text(_ctx.ready ? _ctx.info : 'GPU no disponible',
                  style: TextStyle(
                      fontSize: 11, fontFamily: 'monospace', color: color)),
        ),
        Wrap(
          spacing: 8,
          children: [
            FilledButton(
                onPressed: (_busy || !_ctx.ready) ? null : _gelu,
                child: const Text('GELU', style: TextStyle(fontSize: 12))),
            FilledButton(
                onPressed: (_busy || !_ctx.ready) ? null : _linear,
                child: const Text('LINEAR', style: TextStyle(fontSize: 12))),
            FilledButton(
                onPressed: (_busy || !_ctx.ready) ? null : _attn,
                child: const Text('ATTN', style: TextStyle(fontSize: 12))),
            FilledButton.tonal(
                onPressed: (_busy || !_ctx.ready) ? null : _lab,
                child: const Text('RUN WGSL', style: TextStyle(fontSize: 12))),
          ],
        ),
        Padding(
          padding: const EdgeInsets.fromLTRB(10, 8, 10, 0),
          child: TextField(
            controller: _code,
            maxLines: 6,
            style: const TextStyle(
                fontSize: 10, fontFamily: 'monospace'),
            decoration: const InputDecoration(
              border: OutlineInputBorder(),
              isDense: true,
              hintText: 'kernel WGSL (binding0 f32 in-place, binding1 params)…',
            ),
          ),
        ),
        if (_busy) const LinearProgressIndicator(minHeight: 2),
        Expanded(
          child: ListView.builder(
            padding: const EdgeInsets.all(10),
            itemCount: _rows.length,
            itemBuilder: (_, i) {
              final r = _rows[i];
              final c = r.ok ? Colors.greenAccent : Colors.redAccent;
              return Card(
                color: const Color(0xFF0B1220),
                margin: const EdgeInsets.only(bottom: 6),
                child: ListTile(
                  dense: true,
                  leading: Icon(
                      r.ok ? Icons.check_circle : Icons.cancel,
                      color: c,
                      size: 18),
                  title: Text(r.nombre,
                      style: const TextStyle(fontSize: 12)),
                  subtitle: Text(r.detalle,
                      style: const TextStyle(
                          fontSize: 10, fontFamily: 'monospace')),
                  trailing:
                      CopyBtn(texto: () => '${r.nombre}\n${r.detalle}'),
                ),
              );
            },
          ),
        ),
      ],
    );
  }
}
