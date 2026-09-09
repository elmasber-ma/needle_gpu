import 'package:flutter/material.dart';

import 'app/app.dart';
import 'app/bootstrap.dart';

Future<void> main() async {
  await initApp();
  runApp(const NeedleGpuApp());
}
