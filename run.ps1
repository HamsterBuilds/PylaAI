param([string]$Command = "setup")
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
if ($Command -in @("hamster_bot", "hamster-bot", "start")) {
    & py -3.11 (Join-Path $root "hamster_bot.py")
} elseif ($Command -in @("setup", "install", "wizard", "ui")) {
    & py -3.11 (Join-Path $root "setup.py")
} else {
    Write-Host "Unknown command '$Command'. Use: .\run.ps1 setup or .\run.ps1 hamster_bot"
    exit 2
}
