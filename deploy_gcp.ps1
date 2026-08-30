# deploy_gcp.ps1
# Automates the deployment of the Construction Safety AI app to Google Cloud Run

$PROJECT_ID = "constructionsafetyai"
$SERVICE_NAME = "construction-safety-ai"
$REGION = "us-central1"

# Read Environment Variables from environment or .env if available
$AZURE_ENDPOINT = if ($env:AZURE_OPENAI_ENDPOINT) { $env:AZURE_OPENAI_ENDPOINT } else { "https://your-resource.cognitiveservices.azure.com/" }
$AZURE_KEY = if ($env:AZURE_OPENAI_API_KEY) { $env:AZURE_OPENAI_API_KEY } else { "YOUR_API_KEY_HERE" }
$AZURE_VERSION = if ($env:AZURE_OPENAI_API_VERSION) { $env:AZURE_OPENAI_API_VERSION } else { "2024-12-01-preview" }
$AZURE_DEPLOYMENT = if ($env:AZURE_OPENAI_DEPLOYMENT) { $env:AZURE_OPENAI_DEPLOYMENT } else { "gpt-4o" }

$ENV_VARS = "AZURE_OPENAI_ENDPOINT=$AZURE_ENDPOINT," +
            "AZURE_OPENAI_API_KEY=$AZURE_KEY," +
            "AZURE_OPENAI_API_VERSION=$AZURE_VERSION," +
            "AZURE_OPENAI_DEPLOYMENT=$AZURE_DEPLOYMENT"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Deploying Construction Safety AI to Google Cloud Run" -ForegroundColor Cyan
Write-Host "Project ID: $PROJECT_ID" -ForegroundColor Cyan
Write-Host "Service Name: $SERVICE_NAME" -ForegroundColor Cyan
Write-Host "Region: $REGION" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# Find gcloud command path
$GCLOUD_PATH = "gcloud"
if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    $fallback = "$env:USERPROFILE\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
    if (Test-Path $fallback) {
        $GCLOUD_PATH = $fallback
    } else {
        Write-Error "gcloud CLI is not found on your system PATH. Please install Google Cloud SDK."
        Write-Host "Visit: https://cloud.google.com/sdk/docs/install" -ForegroundColor Yellow
        exit 1
    }
}

# Set the active project
Write-Host "Configuring active GCP project..." -ForegroundColor Green
& $GCLOUD_PATH config set project $PROJECT_ID

# Deploy using Google Cloud Build (remote container building)
Write-Host "Submitting build and deploying to Cloud Run..." -ForegroundColor Green
& $GCLOUD_PATH run deploy $SERVICE_NAME `
    --source . `
    --region $REGION `
    --allow-unauthenticated `
    --set-env-vars=$ENV_VARS

if ($LASTEXITCODE -eq 0) {
    Write-Host "Deployment completed successfully!" -ForegroundColor Green
} else {
    Write-Host "Deployment failed." -ForegroundColor Red
}
