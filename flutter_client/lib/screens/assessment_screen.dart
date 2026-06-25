import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import '../services/api_service.dart';
import '../models/assessment_result.dart';
import 'result_screen.dart';

class AssessmentScreen extends StatefulWidget {
  @override
  _AssessmentScreenState createState() => _AssessmentScreenState();
}

class _AssessmentScreenState extends State<AssessmentScreen> {
  final ImagePicker _picker = ImagePicker();
  final ApiService _apiService = ApiService();
  
  File? _selectedVideo;
  bool _isProcessing = false;

  Future<void> _pickVideo() async {
    final XFile? video = await _picker.pickVideo(source: ImageSource.gallery);
    if (video != null) {
      setState(() {
        _selectedVideo = File(video.path);
      });
    }
  }

  Future<void> _uploadAndAnalyze() async {
    if (_selectedVideo == null) return;

    setState(() {
      _isProcessing = true;
    });

    // Generate a dummy user ID to satisfy the .NET Foreign Key logic
    String dummyUserId = "00000000-0000-0000-0000-000000000000";
    
    AssessmentResult? result = await _apiService.uploadVideo(_selectedVideo!, dummyUserId);

    setState(() {
      _isProcessing = false;
    });

    if (result != null) {
      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (context) => ResultScreen(result: result),
        ),
      );
    } else {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Upload failed or returned invalid data. Check server connection.')),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('NeuroMotion-ADS', style: TextStyle(fontWeight: FontWeight.bold)),
        centerTitle: true,
        elevation: 0,
      ),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              if (_selectedVideo != null) ...[
                Icon(Icons.video_file, size: 64, color: Theme.of(context).colorScheme.primary),
                SizedBox(height: 8),
                Text(
                  _selectedVideo!.path.split('/').last,
                  textAlign: TextAlign.center,
                  style: TextStyle(fontWeight: FontWeight.w600, fontSize: 16),
                ),
              ] else ...[
                Icon(Icons.movie_creation_outlined, size: 64, color: Colors.grey),
                SizedBox(height: 8),
                Text('No video selected.', style: TextStyle(color: Colors.grey, fontSize: 16)),
              ],
              SizedBox(height: 32),
              ElevatedButton.icon(
                onPressed: _isProcessing ? null : _pickVideo,
                icon: Icon(Icons.photo_library),
                label: Text('Select from Gallery'),
                style: ElevatedButton.styleFrom(
                  minimumSize: Size(200, 50),
                ),
              ),
              SizedBox(height: 24),
              if (_isProcessing)
                Column(
                  children: [
                    CircularProgressIndicator(),
                    SizedBox(height: 16),
                    Text('Analyzing behavioral manifold...', style: TextStyle(color: Colors.grey[700])),
                  ],
                )
              else
                ElevatedButton.icon(
                  onPressed: _selectedVideo == null ? null : _uploadAndAnalyze,
                  icon: Icon(Icons.auto_graph),
                  label: Text('Upload & Analyze'),
                  style: ElevatedButton.styleFrom(
                    minimumSize: Size(200, 50),
                    backgroundColor: Theme.of(context).colorScheme.secondaryContainer,
                    foregroundColor: Theme.of(context).colorScheme.onSecondaryContainer,
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }
}
