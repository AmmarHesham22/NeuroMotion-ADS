using System;

namespace NeuroMotionBackend.Models
{
    public class Assessment
    {
        public Guid Id { get; set; } = Guid.NewGuid();
        public Guid UserId { get; set; }
        public User User { get; set; } = null!;
        public string VideoFilePath { get; set; } = string.Empty;
        public double? AdosScore { get; set; }
        public double? AnomalyScore { get; set; }
        public string Status { get; set; } = "Pending"; // Pending, Processing, Completed, Failed
        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    }
}
