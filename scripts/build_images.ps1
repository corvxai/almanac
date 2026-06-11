<#
.SYNOPSIS
  Build arcratio Docker images for an explicit target CPU architecture (Windows).

.DESCRIPTION
  The validator spawns the agent-runner image as sibling containers at runtime.
  A container image is architecture-specific, so the agent-runner image must be
  built for the architecture of the host that will RUN the validator (typically
  an amd64 Linux server). This wrapper makes the target platform explicit and
  warns when cross-building (emulated via buildx/QEMU).

  This is the PowerShell equivalent of scripts/build_images.sh for Windows
  users. On WSL or Git Bash you can use the .sh script instead.

.PARAMETER Target
  agent-runner (default), validator, or all.

.PARAMETER Platform
  Target platform, default linux/amd64. Override with -Platform linux/arm64.

.EXAMPLE
  ./scripts/build_images.ps1
  ./scripts/build_images.ps1 -Target all -Platform linux/amd64
#>
param(
  [ValidateSet("agent-runner", "validator", "all")]
  [string]$Target = "agent-runner",
  [string]$Platform = $(if ($env:PLATFORM) { $env:PLATFORM } else { "linux/amd64" })
)

$ErrorActionPreference = "Stop"

# On Windows the Docker host is Linux (WSL2 / Docker Desktop), so the native
# build platform tracks the Docker engine, normally linux/amd64.
$hostArch = $env:PROCESSOR_ARCHITECTURE
$hostPlat = if ($hostArch -match "ARM64") { "linux/arm64" } else { "linux/amd64" }

Write-Host "Host arch            : $hostArch (engine native: $hostPlat)"
Write-Host "Target build platform: $Platform"

if ($Platform -ne $hostPlat) {
  Write-Warning "Cross-building for $Platform on a $hostPlat engine (QEMU emulation, slower). Needs buildx + binfmt; Docker Desktop bundles these."
}

docker buildx version *> $null
if ($LASTEXITCODE -ne 0) {
  Write-Error "'docker buildx' is required (Docker Desktop / Docker 19.03+)."
  exit 1
}

function Build-Image($dockerfile, $tag) {
  Write-Host ">>> Building $tag from $dockerfile for $Platform"
  docker buildx build --platform $Platform --file $dockerfile --tag $tag --load .
  if ($LASTEXITCODE -ne 0) { throw "build failed for $tag" }
}

switch ($Target) {
  "agent-runner" { Build-Image "docker/agent-runner.Dockerfile" "arcratio/agent-runner:latest" }
  "validator"    { Build-Image "docker/validator.Dockerfile"    "arcratio/validator:latest" }
  "all" {
    Build-Image "docker/agent-runner.Dockerfile" "arcratio/agent-runner:latest"
    Build-Image "docker/validator.Dockerfile"    "arcratio/validator:latest"
  }
}

Write-Host "Done. Built $Target for $Platform."
