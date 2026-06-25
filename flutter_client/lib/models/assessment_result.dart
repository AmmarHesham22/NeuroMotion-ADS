class AssessmentResult {
  final String id;
  final String status;
  final double? adosScore;
  final double? anomalyScore;
  final String? videoFilePath;

  AssessmentResult({
    required this.id,
    required this.status,
    this.adosScore,
    this.anomalyScore,
    this.videoFilePath,
  });

  factory AssessmentResult.fromJson(Map<String, dynamic> json) {
    return AssessmentResult(
      id: json['id'] ?? '',
      status: json['status'] ?? 'Pending',
      adosScore: json['adosScore']?.toDouble(),
      anomalyScore: json['anomalyScore']?.toDouble(),
      videoFilePath: json['videoFilePath'],
    );
  }
}
