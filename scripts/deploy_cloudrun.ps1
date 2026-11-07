# Publish the web analyzer to Google Cloud Run, from this folder.
#
# Before the first run (once):
#   1. A Google Cloud project with an active billing account (console).
#   2. The Groq key stored in Secret Manager, typed by you, never in a file:
#        gcloud secrets create groq-api-key --replication-policy=automatic --project <PROJECT>
#        (then:)  Read-Host "Groq key" | gcloud secrets versions add groq-api-key --data-file=- --project <PROJECT>
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\deploy_cloudrun.ps1 -Project <PROJECT>
#
# What it sets, and why:
#   us-central1          tier-1 region, inside Cloud Run's free monthly allowance
#   --no-cpu-throttling  analyses run in a background thread after /analyze
#                        answers; without it Cloud Run takes the CPU away
#   --max-instances 1    jobs and rate limits live in memory; also a cost ceiling
#   --min-instances 0    nothing runs (or costs) while nobody uses it
#   2 CPU / 2 GiB        OpenCV, OCR and a downloaded video in /tmp (which is RAM)

param(
    [Parameter(Mandatory = $true)][string]$Project,
    [string]$Region = "us-central1",
    [string]$Service = "ai-video-observatory"
)
$ErrorActionPreference = "Stop"

gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com --project $Project

# The service account Cloud Run uses must be able to read the secret.
$number = gcloud projects describe $Project --format="value(projectNumber)"
gcloud secrets add-iam-policy-binding groq-api-key --project $Project `
    --member "serviceAccount:$number-compute@developer.gserviceaccount.com" `
    --role roles/secretmanager.secretAccessor | Out-Null

gcloud run deploy $Service --project $Project --region $Region --source . `
    --allow-unauthenticated `
    --cpu 2 --memory 2Gi --no-cpu-throttling `
    --min-instances 0 --max-instances 1 --concurrency 20 --timeout 900 `
    --set-secrets "GROQ_API_KEY=groq-api-key:latest" `
    --set-env-vars "TRUST_PROXY=1,PROXY_HOPS=1"

$url = gcloud run services describe $Service --project $Project --region $Region --format="value(status.url)"
Write-Host ""
Write-Host "Live at $url"
Write-Host "Check the rate-limit address: $url/api/ip  (it should show your own IP)"
Write-Host "Then set the GitHub variable ANALYZER_URL to $url and re-run the pages workflow."
