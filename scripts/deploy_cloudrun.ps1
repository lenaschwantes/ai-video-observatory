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

# Windows PowerShell does not stop when a native command fails: check each one.
function Step([string]$what) {
    if ($LASTEXITCODE -ne 0) { Write-Host "Stopped: $what failed (see the error above)." -ForegroundColor Red; exit 1 }
}

$billing = gcloud billing projects describe $Project --format="value(billingEnabled)"
Step "reading the project's billing"
if ($billing -ne "True") {
    Write-Host "Billing is not active on $Project. Link it to an open billing account first:" -ForegroundColor Red
    Write-Host "  gcloud billing accounts list"
    Write-Host "  gcloud billing projects link $Project --billing-account=<ID with OPEN True>"
    exit 1
}

gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com --project $Project
Step "enabling the APIs"

# The service account Cloud Run uses must be able to read the secret.
$number = gcloud projects describe $Project --format="value(projectNumber)"
Step "reading the project number"
gcloud secrets add-iam-policy-binding groq-api-key --project $Project `
    --member "serviceAccount:$number-compute@developer.gserviceaccount.com" `
    --role roles/secretmanager.secretAccessor | Out-Null
Step "letting Cloud Run read the groq-api-key secret (does it exist?)"

# Optional: cookies of a secondary YouTube account, so YouTube does not block
# downloads from Google's servers. Mounted as a read-only file.
$secrets = "GROQ_API_KEY=groq-api-key:latest"
$envs = "TRUST_PROXY=1,PROXY_HOPS=1"
$ErrorActionPreference = "Continue"  # a missing secret writes to stderr; that is fine here
gcloud secrets describe youtube-cookies --project $Project --format="value(name)" 2>$null | Out-Null
$hasCookies = ($LASTEXITCODE -eq 0)
$ErrorActionPreference = "Stop"
if ($hasCookies) {
    gcloud secrets add-iam-policy-binding youtube-cookies --project $Project `
        --member "serviceAccount:$number-compute@developer.gserviceaccount.com" `
        --role roles/secretmanager.secretAccessor | Out-Null
    Step "letting Cloud Run read the youtube-cookies secret"
    $secrets += ",/secrets/youtube/cookies.txt=youtube-cookies:latest"
    $envs += ",YTDLP_COOKIES=/secrets/youtube/cookies.txt"
    Write-Host "Using YouTube cookies from the youtube-cookies secret."
}
$global:LASTEXITCODE = 0

gcloud run deploy $Service --project $Project --region $Region --source . --quiet `
    --allow-unauthenticated `
    --cpu 2 --memory 2Gi --no-cpu-throttling `
    --min-instances 0 --max-instances 1 --concurrency 20 --timeout 900 `
    --set-secrets $secrets `
    --set-env-vars $envs
Step "the deploy"

$url = gcloud run services describe $Service --project $Project --region $Region --format="value(status.url)"
Write-Host ""
Write-Host "Live at $url"
Write-Host "Check the rate-limit address: $url/api/ip  (it should show your own IP)"
Write-Host "Then set the GitHub variable ANALYZER_URL to $url and re-run the pages workflow."
