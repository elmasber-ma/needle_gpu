import 'package:flutter/material.dart';

import '../screens/bench_screen.dart';
import '../screens/chat_cpu_screen.dart';
import '../screens/gpu_screen.dart';
import '../screens/tools_screen.dart';

/// Needle CPU vs GPU Web: 4 pestañas (chat CPU, GPU, benchmark, tools).
class NeedleGpuApp extends StatelessWidget {
  const NeedleGpuApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Needle GPU',
      theme: ThemeData.dark().copyWith(
        scaffoldBackgroundColor: const Color(0xFF020617),
      ),
      home: const HomeTabs(),
    );
  }
}

class HomeTabs extends StatelessWidget {
  const HomeTabs({super.key});

  @override
  Widget build(BuildContext context) {
    return DefaultTabController(
      length: 4,
      child: Scaffold(
        appBar: AppBar(
          title: const Text('Needle CPU vs GPU'),
          bottom: const TabBar(
            isScrollable: true,
            tabs: [
              Tab(icon: Icon(Icons.chat_rounded), text: 'Chat'),
              Tab(icon: Icon(Icons.memory_rounded), text: 'GPU Web'),
              Tab(icon: Icon(Icons.speed_rounded), text: 'Benchmark'),
              Tab(icon: Icon(Icons.handyman_rounded), text: 'Tools'),
            ],
          ),
        ),
        body: const TabBarView(
          children [
            ChatCpuScreen(),
            GpuScreen(),
            BenchScreen(),
            ToolsScreen(),
          ],
        ),
      ),
    );
  }
}
