import 'package:flutter/material.dart';
import 'package:needle_gpu/src/rust/frb_generated.dart';

/// Inicializa lo mínimo: binding + puente Rust (el resto es bajo demanda).
Future<void> initApp() async {
  WidgetsFlutterBinding.ensureInitialized();
  try {
    await RustLib.init();
  } catch (e) {
    debugPrint('Rust init error: $e');
  }
}
