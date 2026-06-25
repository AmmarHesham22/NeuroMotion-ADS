using Microsoft.EntityFrameworkCore;
using NeuroMotionBackend.Data;
using NeuroMotionBackend.Services;

var builder = WebApplication.CreateBuilder(args);

// Configure Structured Logging
builder.Logging.ClearProviders();
builder.Logging.AddConsole();
builder.Logging.AddDebug();

// Configure CORS for Flutter Client
builder.Services.AddCors(options =>
{
    options.AddPolicy("AllowFlutterClient", policy =>
    {
        policy.AllowAnyOrigin()
              .AllowAnyHeader()
              .AllowAnyMethod();
    });
});

// Configure Entity Framework Core with SQLite (for development simplicity)
builder.Services.AddDbContext<ApplicationDbContext>(options =>
    options.UseSqlite(builder.Configuration.GetConnectionString("DefaultConnection") ?? "Data Source=NeuroMotion.db"));

// Register File Storage Service
builder.Services.AddSingleton<IFileStorageService, LocalFileStorageService>();

// Register AI Inference HTTP Client
builder.Services.AddHttpClient<IAiInferenceService, FastApiClient>(client =>
{
    client.BaseAddress = new Uri("http://localhost:8000/");
});

builder.Services.AddControllers();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

var app = builder.Build();

if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseHttpsRedirection();

app.UseCors("AllowFlutterClient");

app.UseAuthorization();

app.MapControllers();

app.Run();
