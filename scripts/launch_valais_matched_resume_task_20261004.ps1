param([switch]$TransportSmoke)

$ErrorActionPreference = 'Stop'
$resumeWorkspace = 'E:\new research\detection'
$resumePython = 'E:\Download\anaconda3\envs\geosa-basa\python.exe'
$resumeConfig = Join-Path $resumeWorkspace 'configs\valais_matched_resume_epoch31_20261004.yaml'
$resumeOutput = Join-Path $resumeWorkspace 'runs\valais_matched_resume_epoch31_20261004'
Set-Location -LiteralPath $resumeWorkspace
$env:PYTHONUNBUFFERED = '1'
if ($TransportSmoke) {
    $resumeArguments = '-c "import warnings; warnings.warn(''launcher_stderr_transport_smoke''); print(''launcher_transport_smoke_passed'')"'
    $resumeLogPrefix = 'launcher_smoke'
} else {
    $resumeArguments = '-u -m scripts.resume_valais_matched_study run --config "' + $resumeConfig + '"'
    $resumeLogPrefix = 'queue'
}
$resumeStdout = Join-Path $resumeOutput ($resumeLogPrefix + '.stdout.log')
$resumeStderr = Join-Path $resumeOutput ($resumeLogPrefix + '.stderr.log')
if ((Test-Path -LiteralPath $resumeStdout) -or (Test-Path -LiteralPath $resumeStderr)) {
    throw 'Launcher logs already exist; preserve previous evidence before a reviewed relaunch'
}
# Native stderr is logged as data. Windows PowerShell must use the Python exit
# code, rather than turning an ordinary PyTorch warning into a terminating error.
$resumeProcess = Start-Process -FilePath $resumePython -ArgumentList $resumeArguments `
    -WorkingDirectory $resumeWorkspace -WindowStyle Hidden -PassThru -Wait `
    -RedirectStandardOutput $resumeStdout -RedirectStandardError $resumeStderr
if ($null -eq $resumeProcess.ExitCode) { throw 'Missing Python process exit code' }
exit $resumeProcess.ExitCode
