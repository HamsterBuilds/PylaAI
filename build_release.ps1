$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$version = if ($args.Count) { $args[0] } else { "0.1.0" }
$out = Join-Path $root "dist\\HamsterBOT-$version"
if (Test-Path $out) { Remove-Item -Recurse -Force $out }
New-Item -ItemType Directory -Force -Path $out | Out-Null
foreach ($item in @("api","cfg","images","models","playstyles","static","templates","scrcpy","main.py","play.py","detect.py","pathfinding.py","perception.py","state_finder.py","stage_manager.py","trophy_observer.py","brawl_api_credentials.py","utils.py","window_controller.py","setup.py","run.ps1","setup.cmd","install_command.ps1","requirements.txt","README.md","LICENSE")) {
  $source = Join-Path $root $item; if (Test-Path $source) { Copy-Item -Recurse -Force $source (Join-Path $out $item) }
}
$zip = Join-Path $root "dist\\HamsterBOT-$version.zip"
if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path (Join-Path $out "*") -DestinationPath $zip
Write-Host "Created $zip"
