using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using NeuroMotionBackend.Data;
using NeuroMotionBackend.Models;
using NeuroMotionBackend.Services;
using System;
using System.IO;
using System.Threading.Tasks;

namespace NeuroMotionBackend.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class AssessmentController : ControllerBase
    {
        private readonly ApplicationDbContext _context;
        private readonly IFileStorageService _fileStorage;
        private readonly IAiInferenceService _aiInference;

        public AssessmentController(
            ApplicationDbContext context, 
            IFileStorageService fileStorage, 
            IAiInferenceService aiInference)
        {
            _context = context;
            _fileStorage = fileStorage;
            _aiInference = aiInference;
        }

        [HttpPost("upload")]
        public async Task<IActionResult> UploadVideo([FromForm] IFormFile video, [FromForm] Guid? userId)
        {
            if (video == null || video.Length == 0)
            {
                return BadRequest("No video file uploaded.");
            }

            // Ensure we have a user to satisfy Foreign Key constraint
            User targetUser = null;
            if (userId.HasValue)
            {
                targetUser = await _context.Users.FindAsync(userId.Value);
            }
            
            if (targetUser == null)
            {
                // Create dummy user for testing if none exists
                targetUser = new User { Name = "Test User", Email = "test@example.com" };
                _context.Users.Add(targetUser);
                await _context.SaveChangesAsync();
            }

            // a. Instantiate new Assessment record in DB with status Processing
            var assessment = new Assessment
            {
                UserId = targetUser.Id,
                Status = "Processing"
            };
            _context.Assessments.Add(assessment);
            await _context.SaveChangesAsync();

            try
            {
                // b. Persist the uploaded file
                var relativePath = await _fileStorage.SaveVideoAsync(video, assessment.Id);
                assessment.VideoFilePath = relativePath;
                await _context.SaveChangesAsync();

                // Build absolute path to pass to FastAPI
                // LocalFileStorageService puts it in "uploads/videos/..." relative to CurrentDirectory
                var absolutePath = Path.Combine(Directory.GetCurrentDirectory(), relativePath);

                // c & d. Pass absolute path to FastApiClient and get scores
                var prediction = await _aiInference.PredictAsync(absolutePath);

                if (prediction.AdosScore.HasValue)
                {
                    assessment.AdosScore = prediction.AdosScore.Value;
                    assessment.AnomalyScore = prediction.AnomalyScore.Value;
                    assessment.Status = "Completed";
                }
                else
                {
                    assessment.Status = "Failed"; // e.g., video too short
                }
                
                // e. Save changes and return updated record
                await _context.SaveChangesAsync();
                return Ok(assessment);
            }
            catch (Exception ex)
            {
                assessment.Status = "Failed";
                await _context.SaveChangesAsync();
                return StatusCode(500, new { error = ex.Message });
            }
        }
    }
}
