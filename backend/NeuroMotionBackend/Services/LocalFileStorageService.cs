using Microsoft.AspNetCore.Http;
using System;
using System.IO;
using System.Threading.Tasks;

namespace NeuroMotionBackend.Services
{
    public class LocalFileStorageService : IFileStorageService
    {
        private readonly string _storagePath;

        public LocalFileStorageService()
        {
            _storagePath = Path.Combine(Directory.GetCurrentDirectory(), "uploads", "videos");
            if (!Directory.Exists(_storagePath))
            {
                Directory.CreateDirectory(_storagePath);
            }
        }

        public async Task<string> SaveVideoAsync(IFormFile file, Guid assessmentId)
        {
            if (file == null || file.Length == 0)
            {
                throw new ArgumentException("File is empty or null.", nameof(file));
            }

            var extension = Path.GetExtension(file.FileName);
            var uniqueFileName = $"{assessmentId}{extension}";
            var absolutePath = Path.Combine(_storagePath, uniqueFileName);

            using (var stream = new FileStream(absolutePath, FileMode.Create))
            {
                await file.CopyToAsync(stream);
            }

            // Return relative path for database storage
            return Path.Combine("uploads", "videos", uniqueFileName).Replace("\\", "/");
        }
    }
}
