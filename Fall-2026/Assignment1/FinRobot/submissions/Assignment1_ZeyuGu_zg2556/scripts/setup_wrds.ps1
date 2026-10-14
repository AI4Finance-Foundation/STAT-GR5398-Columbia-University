#Requires -Version 5.1
<#
Build a separate WRDS-only environment using the Python available in the user's
current local terminal. Do not rely on the agent's research-environment launcher.
Credentials are entered only into wrds_access.py's hidden interactive prompts.
#>
param()
$ErrorActionPreference = 'Stop'

$localPython = Get-Command python -CommandType Application -ErrorAction Stop | Select-Object -First 1
& $localPython.Source -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"
if ($LASTEXITCODE -ne 0) {
    throw 'The Python in this terminal must be Python 3.10 or newer. Activate a suitable local Python environment and rerun this script.'
}
if ([string]::IsNullOrWhiteSpace($env:LOCALAPPDATA)) {
    throw 'LOCALAPPDATA is unavailable; no environment was created.'
}
$wrdsRoot = [IO.Path]::GetFullPath((Join-Path $env:LOCALAPPDATA 'FinRobotWRDS'))
$wrdsVenv = Join-Path $wrdsRoot 'venv'
$wrdsPython = Join-Path $wrdsVenv 'Scripts\python.exe'
New-Item -ItemType Directory -Path $wrdsRoot -Force | Out-Null
if (Test-Path -LiteralPath $wrdsPython) {
    & $wrdsPython -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" 2>$null
    if ($LASTEXITCODE -ne 0) {
        # Preserve an incompatible prior environment rather than clearing it.
        $wrdsVenv = Join-Path $wrdsRoot ('venv-' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmss'))
        $wrdsPython = Join-Path $wrdsVenv 'Scripts\python.exe'
    }
}
if (-not (Test-Path -LiteralPath $wrdsPython)) {
    Write-Host 'Creating a separate local WRDS environment with the Python from this terminal...'
    & $localPython.Source -m venv $wrdsVenv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the local WRDS environment.' }
}

# Use the public package index explicitly, without displaying any private pip
# index credentials inherited from the shell. Restore environment preferences.
$savedPipConfig = $env:PIP_CONFIG_FILE
$savedPipExtra = $env:PIP_EXTRA_INDEX_URL
$savedPipIndex = $env:PIP_INDEX_URL
try {
    $env:PIP_CONFIG_FILE = 'NUL'
    $env:PIP_EXTRA_INDEX_URL = ''
    $env:PIP_INDEX_URL = 'https://pypi.org/simple'
    & $wrdsPython -m pip install --disable-pip-version-check --no-input --index-url 'https://pypi.org/simple' 'psycopg2-binary==2.9.11'
    if ($LASTEXITCODE -ne 0) { throw 'Installing the pinned WRDS PostgreSQL driver failed.' }
}
finally {
    $env:PIP_CONFIG_FILE = $savedPipConfig
    $env:PIP_EXTRA_INDEX_URL = $savedPipExtra
    $env:PIP_INDEX_URL = $savedPipIndex
}

Write-Host 'WRDS login will now be requested locally with hidden input.'
& $wrdsPython (Join-Path $PSScriptRoot 'wrds_access.py') setup
if ($LASTEXITCODE -ne 0) {
    throw 'WRDS setup did not complete. Sanitized status is in evidence\wrds\access_latest.json.'
}
