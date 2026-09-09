import 'dart:convert';
import 'dart:io';

import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';

import 'gpu/gpu_shader_lab.dart';
import 'needle_service.dart';

/// Tools locales para el loop agéntico con Needle (on-device).
/// Formato de schema: el que entrena needle (compacto, sin pretty-print).
class ToolDef {
  final String name;
  final String description;
  final Map<String, dynamic> parameters;
  const ToolDef(this.name, this.description, this.parameters);
}

const kTools = [
  ToolDef(
    'leer_archivo',
    'Lee un archivo de texto dentro de las carpetas de la app.',
    {
      'type': 'object',
      'properties': {
        'path': {'type': 'string', 'description': 'ruta relativa o absoluta'}
      },
      'required': ['path'],
    },
  ),
  ToolDef(
    'escribir_archivo',
    'Crea o sobrescribe un archivo de texto en las carpetas de la app.',
    {
      'type': 'object',
      'properties': {
        'path': {'type': 'string'},
        'content': {'type': 'string'},
      },
      'required': ['path', 'content'],
    },
  ),
  ToolDef(
    'listar_dir',
    'Lista los archivos de una carpeta de la app.',
    {
      'type': 'object',
      'properties': {
        'path': {'type': 'string', 'description': "'.' = raíz de Documentos"}
      },
      'required': ['path'],
    },
  ),
  ToolDef(
    'buscar_web',
    'Descarga una URL http(s) y devuelve el texto (HTML sin etiquetas).',
    {
      'type': 'object',
      'properties': {
        'url': {'type': 'string'}
      },
      'required': ['url'],
    },
  ),
  ToolDef(
    'usar_gpu',
    'Ejecuta un kernel WGSL @compute sobre N elementos en la GPU y devuelve los primeros valores y el tiempo.',
    {
      'type': 'object',
      'properties': {
        'wgsl': {'type': 'string', 'description': 'código WGSL con @compute'},
        'n': {'type': 'integer', 'description': 'cantidad de elementos'},
      },
      'required': ['wgsl', 'n'],
    },
  ),
];

/// JSON compacto para needle (el pretty-print rompe la decisión).
String toolsJson() => jsonEncode([
      for (final t in kTools)
        {'name': t.name, 'description': t.description, 'parameters': t.parameters},
    ]);

Future<Directory> _docsDir() async =>
    Directory((await getApplicationDocumentsDirectory()).path);

/// Resuelve ruta y garantiza que quede DENTRO de las carpetas de la app.
Future<String?> _resolveSandboxed(String path) async {
  final docs = await _docsDir();
  final sup = await getApplicationSupportDirectory();
  String p = path.trim();
  if (p.startsWith('./')) p = p.substring(2);
  final f = p.startsWith('/')
      ? File(p)
      : File('${docs.path}${Platform.pathSeparator}$p');
  final canonical = f.parent.resolveSymbolicLinksSync();
  if (!canonical.startsWith(docs.path) && !canonical.startsWith(sup.path)) {
    return null;
  }
  return f.path;
}

String _clip(String s, int max) =>
    s.length <= max ? s : '${s.substring(0, max)}\n…[truncado ${s.length} chars]';

String stripHtml(String html) {
  var t = html.replaceAll(
      RegExp(r'<script[^>]*>.*?</script>', dotAll: true, multiLine: true), '');
  t = t.replaceAll(
      RegExp(r'<style[^>]*>.*?</style>', dotAll: true, multiLine: true), '');
  t = t.replaceAll(RegExp(r'<[^>]+>'), ' ');
  t = t.replaceAll('&nbsp;', ' ').replaceAll('&amp;', '&');
  return t.replaceAll(RegExp(r'\s+'), ' ').trim();
}

/// Ejecuta una tool YA autorizada. Nunca lanza: devuelve texto.
Future<String> executeTool(String name, Map<String, dynamic> args) async {
  try {
    switch (name) {
      case 'leer_archivo':
        final path = await _resolveSandboxed(args['path'] ?? '.');
        if (path == null) return 'ERROR: ruta fuera del sandbox de la app';
        final f = File(path);
        if (!f.existsSync()) return 'ERROR: no existe $path';
        return _clip(f.readAsStringSync(), 8000);

      case 'escribir_archivo':
        final path = await _resolveSandboxed(args['path'] ?? '');
        if (path == null) return 'ERROR: ruta fuera del sandbox de la app';
        File(path).parent.createSync(recursive: true);
        File(path).writeAsStringSync(args['content'] ?? '');
        return 'OK: escrito ${File(path).lengthSync()} bytes en $path';

      case 'listar_dir':
        final path = await _resolveSandboxed(args['path'] ?? '.');
        if (path == null) return 'ERROR: fuera del sandbox';
        final d =
            FileSystemEntity.isDirectorySync(path) ? Directory(path) : null;
        if (d == null || !d.existsSync()) return 'ERROR: carpeta inexistente';
        return _clip(d.listSync().map((e) {
          final tag = e is Directory ? '[d] ' : '[f] ';
          return '$tag${e.path.split(Platform.pathSeparator).last}';
        }).join('\n'), 8000);

      case 'buscar_web':
        final url = args['url'] ?? '';
        final res =
            await http.get(Uri.parse(url)).timeout(const Duration(seconds: 15));
        if (res.statusCode != 200) return 'ERROR: HTTP ${res.statusCode}';
        final ct = res.headers['content-type'] ?? '';
        final text = ct.contains('html') ? stripHtml(res.body) : res.body;
        return _clip(text, 8000);

      case 'usar_gpu':
        final lab = GpuShaderLab();
        final n = ((args['n'] ?? 64) as num).toInt().clamp(8, 65536);
        final r = await lab.run(
          code: args['wgsl'] ?? '',
          input: List.generate(n, (i) => i.toDouble()),
          dispatchX: ((n + 63) ~/ 64).clamp(1, 65535),
          dispatchY: 1,
          dispatchZ: 1,
          paramX: 2.0,
          paramY: 0,
          paramZ: 0,
        );
        if (!r.ok) return 'ERROR GPU: ${r.error}';
        return 'OK GPU (${r.elapsedMs.toStringAsFixed(2)} ms): '
            '${r.data.take(8).map((v) => v.toStringAsFixed(3)).join('  ')}';

      default:
        return 'ERROR: herramienta desconocida $name';
    }
  } catch (e) {
    return 'ERROR ejecutando $name: $e';
  }
}

Map<String, dynamic> parseToolArgs(String raw) {
  try {
    final v = jsonDecode(raw);
    if (v is Map<String, dynamic>) {
      if (v['arguments'] is Map) {
        return Map<String, dynamic>.from(v['arguments']);
      }
      if (v['parameters'] is Map) {
        return Map<String, dynamic>.from(v['parameters']);
      }
      return v;
    }
    if (v is List && v.isNotEmpty && v.first is Map) {
      return Map<String, dynamic>.from(v.first);
    }
  } catch (_) {}
  return {};
}

String toolNameOf(String raw) {
  try {
    final v = jsonDecode(raw);
    if (v is Map<String, dynamic>) {
      for (final k in ['name', 'tool', 'function']) {
        if (v[k] is String) return v[k];
      }
    }
  } catch (_) {}
  return '';
}

/// Un paso del loop agéntico (para pintar el log en la UI).
class AgentStep {
  final String kind; // 'tool' | 'respuesta' | 'error'
  final String text;
  AgentStep(this.kind, this.text);
}

/// Loop agéntico con Needle on-device: query → tool_call → ejecuta →
/// realimenta, hasta respuesta final o límite de iteraciones.
/// Llama [onStep] por cada paso para la UI. Retorna la respuesta final.
Future<String> agentLoop({
  required String query,
  int maxIters = 6,
  int maxNewTokens = 160,
  void Function(AgentStep step)? onStep,
}) async {
  final svc = NeedleService.instance;
  if (!svc.loaded) throw 'modelo no cargado';
  final tools = toolsJson();
  var contexto = query;
  for (var i = 0; i < maxIters; i++) {
    final out = await svc.run(
      query: contexto,
      toolsJson: tools,
      constrain: true,
      maxNewTokens: maxNewTokens,
    );
    final tc = (out.toolCall ?? '').trim();
    if (tc.isEmpty) {
      final resp = out.text.trim().isEmpty ? '(sin texto)' : out.text.trim();
      onStep?.call(AgentStep('respuesta', resp));
      return resp;
    }
    final nombre = toolNameOf(tc);
    final args = parseToolArgs(tc);
    onStep?.call(AgentStep('tool', '$nombre ${jsonEncode(args)}'));
    final def = kTools.where((t) => t.name == nombre);
    final res = def.isEmpty
        ? 'ERROR: herramienta desconocida $nombre'
        : await executeTool(nombre, args);
    onStep?.call(AgentStep('tool', '→ ${_clip(res, 500)}'));
    contexto = '$contexto\n[Resultado de $nombre: $res]\nRespondé al usuario.';
  }
  return '(límite de $maxIters pasos alcanzado)';
}
