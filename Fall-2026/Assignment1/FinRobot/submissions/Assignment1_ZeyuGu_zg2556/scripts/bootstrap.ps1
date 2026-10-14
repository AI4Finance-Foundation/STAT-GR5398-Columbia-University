param(
    [string]$RuntimeRoot = "$env:LOCALAPPDATA\FinRobot-zg2556",
    [string]$PythonVersion = "3.11"
)
$ErrorActionPreference = 'Stop'
$SubmissionRoot = Split-Path -Parent $PSScriptRoot
$ResolvedRuntime = [System.IO.Path]::GetFullPath($RuntimeRoot)
if ($ResolvedRuntime.StartsWith([System.IO.Path]::GetFullPath($SubmissionRoot), [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'Keep upstream checkout and virtual environment outside the course submission.'
}
$UpstreamPath = Join-Path $ResolvedRuntime 'upstream'
$EnvironmentPath = Join-Path $ResolvedRuntime 'venv'
$PinnedSha = '2717499b8e30f242640af08c4ad9afd1113c2d45'
if (Test-Path -LiteralPath $UpstreamPath) {
    throw 'Destination already exists. Reuse a verified installation or choose a new RuntimeRoot; this script never resets a checkout.'
}
New-Item -ItemType Directory -Force -Path $ResolvedRuntime | Out-Null
git clone --no-checkout https://github.com/AI4Finance-Foundation/FinRobot.git $UpstreamPath
if ($LASTEXITCODE -ne 0) { throw 'Clone failed' }
git -C $UpstreamPath checkout --detach $PinnedSha
if ($LASTEXITCODE -ne 0) { throw 'Pinned checkout failed' }
foreach ($PatchName in @('compatibility.patch', 'option_a_verifier.patch')) {
    $PatchPath = Join-Path $SubmissionRoot "patches\$PatchName"
    git -C $UpstreamPath apply --check $PatchPath
    if ($LASTEXITCODE -ne 0) { throw "Patch check failed: $PatchName" }
    git -C $UpstreamPath apply $PatchPath
    if ($LASTEXITCODE -ne 0) { throw "Patch application failed: $PatchName" }
}
uv venv --python $PythonVersion $EnvironmentPath
if ($LASTEXITCODE -ne 0) { throw 'Environment creation failed' }
$PythonPath = Join-Path $EnvironmentPath 'Scripts\python.exe'
uv pip install --python $PythonPath -r (Join-Path $SubmissionRoot 'requirements-lock.txt')
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
$env:FINROBOT_UPSTREAM_SRC = Join-Path $UpstreamPath 'finrobot_equity\core\src'
$env:FINROBOT_UPSTREAM = $UpstreamPath
& $PythonPath -m pytest -q (Join-Path $SubmissionRoot 'tests')
if ($LASTEXITCODE -ne 0) { throw 'Offline regression verification failed' }
Write-Output "Upstream: $UpstreamPath"
Write-Output "Python: $PythonPath"
Write-Output 'No paid request was made. Configure credentials outside the submission, then run preflight.'
