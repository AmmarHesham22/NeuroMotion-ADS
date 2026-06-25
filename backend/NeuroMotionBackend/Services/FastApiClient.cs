using System;
using System.IO;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text.Json;
using System.Threading.Tasks;

namespace NeuroMotionBackend.Services
{
    public interface IAiInferenceService
    {
        Task<AiPredictionResult> PredictAsync(string absoluteVideoPath);
    }

    public class AiPredictionResult
    {
        public double? AdosScore { get; set; }
        public double? AnomalyScore { get; set; }
        public int? ChunksAnalyzed { get; set; }
        public string? Message { get; set; }
    }

    public class FastApiClient : IAiInferenceService
    {
        private readonly HttpClient _httpClient;

        public FastApiClient(HttpClient httpClient)
        {
            _httpClient = httpClient;
        }

        public async Task<AiPredictionResult> PredictAsync(string absoluteVideoPath)
        {
            if (!File.Exists(absoluteVideoPath))
            {
                throw new FileNotFoundException($"Video file not found at {absoluteVideoPath}");
            }

            using var form = new MultipartFormDataContent();
            using var fileStream = new FileStream(absoluteVideoPath, FileMode.Open, FileAccess.Read);
            using var streamContent = new StreamContent(fileStream);

            // Determine content type based on extension
            var ext = Path.GetExtension(absoluteVideoPath).ToLower();
            string contentType = ext == ".avi" ? "video/avi" : "video/mp4";
            streamContent.Headers.ContentType = new MediaTypeHeaderValue(contentType);

            form.Add(streamContent, "video", Path.GetFileName(absoluteVideoPath));

            var response = await _httpClient.PostAsync("api/v1/predict", form);
            response.EnsureSuccessStatusCode();

            var responseString = await response.Content.ReadAsStringAsync();
            var jsonDoc = JsonDocument.Parse(responseString);
            var root = jsonDoc.RootElement;

            if (root.TryGetProperty("status", out var status) && status.GetString() == "success")
            {
                var data = root.GetProperty("data");
                
                double? ados = null;
                double? anomaly = null;
                int? chunks = null;
                string? msg = null;

                if (data.TryGetProperty("ados_score", out var adosProp) && adosProp.ValueKind != JsonValueKind.Null)
                    ados = adosProp.GetDouble();

                if (data.TryGetProperty("anomaly_score", out var anomProp) && anomProp.ValueKind != JsonValueKind.Null)
                    anomaly = anomProp.GetDouble();

                if (data.TryGetProperty("chunks_analyzed", out var chunksProp) && chunksProp.ValueKind != JsonValueKind.Null)
                    chunks = chunksProp.GetInt32();
                    
                if (data.TryGetProperty("message", out var msgProp) && msgProp.ValueKind != JsonValueKind.Null)
                    msg = msgProp.GetString();

                return new AiPredictionResult
                {
                    AdosScore = ados,
                    AnomalyScore = anomaly,
                    ChunksAnalyzed = chunks,
                    Message = msg
                };
            }

            throw new Exception($"FastAPI returned error or unexpected format: {responseString}");
        }
    }
}
