import 'package:flutter/material.dart';
import 'screens/assessment_screen.dart';

void main() {
  runApp(const NeuroMotionApp());
}

class NeuroMotionApp extends StatelessWidget {
  const NeuroMotionApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'NeuroMotion-ADS',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.blueAccent,
          brightness: Brightness.light,
        ),
        useMaterial3: true,
      ),
      // Bind the root to our new Assessment capture screen
      home: AssessmentScreen(),
    );
  }
}
