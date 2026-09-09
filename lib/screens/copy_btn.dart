import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

/// Botón copiar estándar para logs y errores: copia [texto] y avisa.
class CopyBtn extends StatelessWidget {
  final String Function() texto;
  final double size;
  const CopyBtn({super.key, required this.texto, this.size = 16});

  @override
  Widget build(BuildContext context) {
    return IconButton(
      visualDensity: VisualDensity.compact,
      tooltip: 'Copiar log',
      icon: Icon(Icons.copy_rounded, size: size),
      onPressed: () {
        Clipboard.setData(ClipboardData(text: texto()));
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
              content: Text('Log copiado'),
              duration: Duration(seconds: 1)),
        );
      },
    );
  }
}
