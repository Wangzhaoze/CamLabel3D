param(
    [string]$CondaEnvName = "camlabel3d",
    [string]$ReleaseDate = "2026-07-16",
    [string]$ReleaseName = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not $ReleaseName) {
    $ReleaseName = "CamLabel3D-windows-$ReleaseDate"
}

function Assert-WorkspacePath {
    param([string]$PathToCheck, [string]$WorkspaceRoot)
    $resolved = [System.IO.Path]::GetFullPath($PathToCheck)
    $workspace = [System.IO.Path]::GetFullPath($WorkspaceRoot)
    if (-not $resolved.StartsWith($workspace, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to operate outside workspace: $resolved"
    }
}

function Reset-WorkspaceDirectory {
    param([string]$TargetPath, [string]$WorkspaceRoot)
    Assert-WorkspacePath -PathToCheck $TargetPath -WorkspaceRoot $WorkspaceRoot
    if (Test-Path -LiteralPath $TargetPath) {
        Remove-Item -LiteralPath $TargetPath -Recurse -Force
    }
    New-Item -ItemType Directory -Path $TargetPath -Force | Out-Null
}

function Copy-ReleaseItem {
    param([string]$Source, [string]$DestinationParent)
    if (-not (Test-Path -LiteralPath $Source)) {
        throw "Required release item is missing: $Source"
    }
    Copy-Item -LiteralPath $Source -Destination $DestinationParent -Recurse -Force
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [string[]]$ArgumentList = @()
    )
    & $FilePath @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $($LASTEXITCODE): $FilePath $($ArgumentList -join ' ')"
    }
}

$RepoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$BuildRoot = Join-Path $RepoRoot "build"
$ReleaseRoot = Join-Path $BuildRoot "release"
$StageRoot = Join-Path $ReleaseRoot $ReleaseName
$RuntimeRoot = Join-Path $StageRoot "runtime"
$LauncherDistRoot = Join-Path $BuildRoot "launcher-dist"
$PyInstallerWorkRoot = Join-Path $BuildRoot "pyinstaller-work"
$PyInstallerSpecRoot = Join-Path $BuildRoot "pyinstaller-spec"
$PackedEnvZip = Join-Path $BuildRoot "$CondaEnvName-packed.zip"
$ReleaseZip = Join-Path $ReleaseRoot "$ReleaseName.zip"
$HelperRoot = Join-Path $BuildRoot "helpers"

New-Item -ItemType Directory -Path $BuildRoot -Force | Out-Null
New-Item -ItemType Directory -Path $ReleaseRoot -Force | Out-Null
New-Item -ItemType Directory -Path $HelperRoot -Force | Out-Null

$envInfo = & conda env list --json | ConvertFrom-Json
$envPath = $envInfo.envs | Where-Object { (Split-Path $_ -Leaf) -eq $CondaEnvName } | Select-Object -First 1
if (-not $envPath) {
    throw "Conda environment '$CondaEnvName' was not found."
}

$EnvPython = Join-Path $envPath "python.exe"
if (-not (Test-Path -LiteralPath $EnvPython)) {
    throw "Environment python was not found: $EnvPython"
}
$CondaPackExe = Join-Path $envPath "Scripts\conda-pack.exe"
if (-not (Test-Path -LiteralPath $CondaPackExe)) {
    $CondaPackExe = ""
}

$requiredCheckpoints = @(
    "wilddet3d_alldata_all_prompt_v1.0.pt",
    "lingbot_depth_model.pt"
)
foreach ($checkpoint in $requiredCheckpoints) {
    $checkpointPath = Join-Path $RepoRoot "ckpts\$checkpoint"
    if (-not (Test-Path -LiteralPath $checkpointPath)) {
        throw "Required checkpoint is missing: $checkpointPath"
    }
}

Write-Host "Installing packaging tools into $CondaEnvName ..."
Invoke-Checked -FilePath $EnvPython -ArgumentList @("-m", "pip", "install", "--disable-pip-version-check", "pyinstaller", "conda-pack")

$CondaPackExe = Join-Path $envPath "Scripts\conda-pack.exe"
if (-not (Test-Path -LiteralPath $CondaPackExe)) {
    throw "conda-pack.exe was not installed into the environment."
}

Reset-WorkspaceDirectory -TargetPath $LauncherDistRoot -WorkspaceRoot $RepoRoot
Reset-WorkspaceDirectory -TargetPath $PyInstallerWorkRoot -WorkspaceRoot $RepoRoot
Reset-WorkspaceDirectory -TargetPath $PyInstallerSpecRoot -WorkspaceRoot $RepoRoot
Reset-WorkspaceDirectory -TargetPath $StageRoot -WorkspaceRoot $RepoRoot

Write-Host "Building launcher executables ..."
Invoke-Checked -FilePath $EnvPython -ArgumentList @(
    "-m", "PyInstaller",
    "--noconfirm",
    "--clean",
    "--onefile",
    "--windowed",
    "--name", "CamLabel3D",
    "--distpath", $LauncherDistRoot,
    "--workpath", $PyInstallerWorkRoot,
    "--specpath", $PyInstallerSpecRoot,
    (Join-Path $RepoRoot "packaging\launch_camlabel3d.py")
)

Invoke-Checked -FilePath $EnvPython -ArgumentList @(
    "-m", "PyInstaller",
    "--noconfirm",
    "--clean",
    "--onefile",
    "--console",
    "--name", "CamLabel3D-Postprocess",
    "--distpath", $LauncherDistRoot,
    "--workpath", $PyInstallerWorkRoot,
    "--specpath", $PyInstallerSpecRoot,
    (Join-Path $RepoRoot "packaging\launch_camlabel3d_postprocess.py")
)

Write-Host "Packing Conda runtime ..."
if (Test-Path -LiteralPath $PackedEnvZip) {
    Remove-Item -LiteralPath $PackedEnvZip -Force
}
Invoke-Checked -FilePath $CondaPackExe -ArgumentList @(
    "-p", $envPath,
    "-o", $PackedEnvZip,
    "--format", "zip",
    "--compress-level", "0",
    "--ignore-missing-files",
    "--force"
)

$extractZipScript = Join-Path $HelperRoot "extract_zip.py"
@"
from pathlib import Path
import sys
import zipfile

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
destination.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(source) as archive:
    archive.extractall(destination)
"@ | Set-Content -LiteralPath $extractZipScript -Encoding UTF8

$createZipScript = Join-Path $HelperRoot "create_zip.py"
@"
from pathlib import Path
import sys
import zipfile

source = Path(sys.argv[1]).resolve()
target = Path(sys.argv[2]).resolve()
target.parent.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
    for path in sorted(source.rglob("*")):
        if path.is_dir():
            continue
        archive.write(path, path.relative_to(source))
"@ | Set-Content -LiteralPath $createZipScript -Encoding UTF8

Write-Host "Expanding runtime into staged release ..."
Invoke-Checked -FilePath $EnvPython -ArgumentList @($extractZipScript, $PackedEnvZip, $RuntimeRoot)
if (Test-Path -LiteralPath $PackedEnvZip) {
    Remove-Item -LiteralPath $PackedEnvZip -Force
}

Write-Host "Copying application sources and assets ..."
Copy-ReleaseItem -Source (Join-Path $RepoRoot "camlabel3d") -DestinationParent $StageRoot
Copy-ReleaseItem -Source (Join-Path $RepoRoot "workers") -DestinationParent $StageRoot
Copy-ReleaseItem -Source (Join-Path $RepoRoot "configs") -DestinationParent $StageRoot

$stageCkptRoot = Join-Path $StageRoot "ckpts"
New-Item -ItemType Directory -Path $stageCkptRoot -Force | Out-Null
foreach ($checkpoint in $requiredCheckpoints) {
    Copy-Item -LiteralPath (Join-Path $RepoRoot "ckpts\$checkpoint") -Destination $stageCkptRoot -Force
}

Copy-Item -LiteralPath (Join-Path $RepoRoot "README.md") -Destination $StageRoot -Force
Copy-Item -LiteralPath (Join-Path $LauncherDistRoot "CamLabel3D.exe") -Destination $StageRoot -Force
Copy-Item -LiteralPath (Join-Path $LauncherDistRoot "CamLabel3D-Postprocess.exe") -Destination $StageRoot -Force

$releaseReadme = Join-Path $StageRoot "RELEASE_README.txt"
@"
CamLabel3D Windows Release
Build date: $ReleaseDate

Files:
- CamLabel3D.exe: desktop UI
- CamLabel3D-Postprocess.exe: postprocessing CLI

Notes:
- Extract the whole folder before launching either executable.
- The first launch may take a little longer because the bundled runtime finalizes itself for the extracted path.
- The packaged checkpoints are limited to the files required by the CamLabel3D desktop workflow:
  - ckpts\wilddet3d_alldata_all_prompt_v1.0.pt
  - ckpts\lingbot_depth_model.pt
"@ | Set-Content -LiteralPath $releaseReadme -Encoding ASCII

Write-Host "Smoke testing staged executables ..."
Invoke-Checked -FilePath (Join-Path $StageRoot "CamLabel3D.exe") -ArgumentList @("--self-test")
Invoke-Checked -FilePath (Join-Path $StageRoot "CamLabel3D-Postprocess.exe") -ArgumentList @("--self-test")

Write-Host "Creating final release zip ..."
if (Test-Path -LiteralPath $ReleaseZip) {
    Remove-Item -LiteralPath $ReleaseZip -Force
}
Invoke-Checked -FilePath $EnvPython -ArgumentList @($createZipScript, $StageRoot, $ReleaseZip)

Write-Host ""
Write-Host "Release folder: $StageRoot"
Write-Host "Release zip:    $ReleaseZip"
