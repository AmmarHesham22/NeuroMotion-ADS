import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;
import '../models/assessment_result.dart';

class ApiService {
  // Use 10.0.2.2 for Android emulator to access the local machine's localhost, 
  // or use localhost/127.0.0.1 for iOS simulator and web.
  // We assume the default ASP.NET Core HTTP port is 5000.
  static const String baseUrl = 'http://10.0.2.2:5000/api/Assessment';

  Future<AssessmentResult?> uploadVideo(File videoFile, String userId) async {
    try {
      var request = http.MultipartRequest('POST', Uri.parse('$baseUrl/upload'));
      
      // Attach the user ID parameter
      request.fields['userId'] = userId;
      
      // Attach the physical video file as a multipart binary stream
      var multipartFile = await http.MultipartFile.fromPath(
        'video',
        videoFile.path,
      );
      request.files.add(multipartFile);

      // Execute the request
      var streamedResponse = await request.send();
      var response = await http.Response.fromStream(streamedResponse);

      // Parse the output
      if (response.statusCode == 200 || response.statusCode == 201) {
        var jsonResponse = jsonDecode(response.body);
        return AssessmentResult.fromJson(jsonResponse);
      } else {
        print('Upload failed. Status: ${response.statusCode}');
        print('Response body: ${response.body}');
        return null;
      }
    } catch (e) {
      print('Critical error during video upload: $e');
      return null;
    }
  }
}
