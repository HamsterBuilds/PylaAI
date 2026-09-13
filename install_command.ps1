$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$profileDir = Split-Path -Parent $PROFILE
New-Item -ItemType Directory -Force -Path $profileDir | Out-Null
$line = "function run([string]`$command) { & '$root\\run.ps1' `$command }"
$profileText = if (Test-Path $PROFILE) { Get-Content -Raw $PROFILE } else { "" }
if ($profileText -notmatch 'function run\(') { Add-Content -Path $PROFILE -Value "`n$line`n" }
Write-Host "PowerShell command installed: run hamster_bot"
