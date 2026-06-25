using Microsoft.AspNetCore.Http;
using System;
using System.Threading.Tasks;

namespace NeuroMotionBackend.Services
{
    public interface IFileStorageService
    {
        Task<string> SaveVideoAsync(IFormFile file, Guid assessmentId);
    }
}
