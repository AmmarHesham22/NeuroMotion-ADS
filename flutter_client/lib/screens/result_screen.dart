import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import '../models/assessment_result.dart';

class ResultScreen extends StatelessWidget {
  final AssessmentResult result;

  ResultScreen({required this.result});

  @override
  Widget build(BuildContext context) {
    // Handle processing failure or null score edges
    if (result.status == 'Failed' || result.adosScore == null) {
      return Scaffold(
        appBar: AppBar(title: Text('Analysis Results')),
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(Icons.error_outline, size: 64, color: Colors.redAccent),
              SizedBox(height: 16),
              Text(
                'Analysis Failed.\nThe AI service could not process the video.',
                textAlign: TextAlign.center,
                style: TextStyle(fontSize: 18, color: Colors.redAccent),
              ),
              SizedBox(height: 24),
              ElevatedButton(
                onPressed: () => Navigator.pop(context),
                child: Text('Go Back'),
              ),
            ],
          ),
        ),
      );
    }

    // Determine severity baseline dynamically (Assuming ADOS scale max ~ 10.0 for this prototype)
    double adosVal = result.adosScore!;
    double anomVal = result.anomalyScore ?? 0.0;
    
    String severityLabel = adosVal > 7.0 ? "High Severity" : (adosVal > 4.0 ? "Moderate" : "Low / Typical");
    Color severityColor = adosVal > 7.0 ? Colors.red : (adosVal > 4.0 ? Colors.orange : Colors.green);

    return Scaffold(
      appBar: AppBar(
        title: Text('Diagnostic Report'),
        centerTitle: true,
      ),
      body: Padding(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Card(
              elevation: 4,
              child: Padding(
                padding: const EdgeInsets.all(24.0),
                child: Column(
                  children: [
                    Text(
                      'Predicted ADOS Metric',
                      style: TextStyle(fontSize: 16, color: Colors.grey[700]),
                    ),
                    SizedBox(height: 8),
                    Text(
                      adosVal.toStringAsFixed(2),
                      style: TextStyle(fontSize: 48, fontWeight: FontWeight.bold, color: severityColor),
                    ),
                    SizedBox(height: 8),
                    Container(
                      padding: EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                      decoration: BoxDecoration(
                        color: severityColor.withOpacity(0.1),
                        borderRadius: BorderRadius.circular(20),
                      ),
                      child: Text(
                        severityLabel,
                        style: TextStyle(color: severityColor, fontWeight: FontWeight.w600),
                      ),
                    ),
                  ],
                ),
              ),
            ),
            SizedBox(height: 32),
            Text(
              'Behavioral Manifold Distribution',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.w600),
              textAlign: TextAlign.center,
            ),
            SizedBox(height: 24),
            Expanded(
              child: BarChart(
                BarChartData(
                  alignment: BarChartAlignment.spaceAround,
                  maxY: 10,
                  barGroups: [
                    BarChartGroupData(
                      x: 0,
                      barRods: [
                        BarChartRodData(
                          toY: adosVal,
                          color: Colors.blueAccent,
                          width: 40,
                          borderRadius: BorderRadius.circular(4),
                        ),
                      ],
                    ),
                    BarChartGroupData(
                      x: 1,
                      barRods: [
                        BarChartRodData(
                          toY: anomVal,
                          color: Colors.deepPurpleAccent,
                          width: 40,
                          borderRadius: BorderRadius.circular(4),
                        ),
                      ],
                    ),
                  ],
                  titlesData: FlTitlesData(
                    show: true,
                    bottomTitles: AxisTitles(
                      sideTitles: SideTitles(
                        showTitles: true,
                        getTitlesWidget: (value, meta) {
                          switch (value.toInt()) {
                            case 0:
                              return Padding(padding: EdgeInsets.only(top: 8), child: Text('ADOS\nScore', textAlign: TextAlign.center, style: TextStyle(fontWeight: FontWeight.bold)));
                            case 1:
                              return Padding(padding: EdgeInsets.only(top: 8), child: Text('Anomaly\nIndex', textAlign: TextAlign.center, style: TextStyle(fontWeight: FontWeight.bold)));
                            default:
                              return Text('');
                          }
                        },
                      ),
                    ),
                    leftTitles: AxisTitles(
                      sideTitles: SideTitles(showTitles: true, reservedSize: 32),
                    ),
                    topTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
                    rightTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
                  ),
                  borderData: FlBorderData(show: false),
                  gridData: FlGridData(show: true, drawVerticalLine: false, horizontalInterval: 2.0),
                ),
              ),
            ),
            SizedBox(height: 24),
            ElevatedButton(
              onPressed: () => Navigator.pop(context),
              child: Text('Process Another Video'),
              style: ElevatedButton.styleFrom(
                minimumSize: Size(double.infinity, 50),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
