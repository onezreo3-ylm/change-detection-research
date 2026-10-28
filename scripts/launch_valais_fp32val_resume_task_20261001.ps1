$ErrorActionPreference = 'Stop'
$workspace = 'E:\new research\detection'
$python = 'E:\Download\anaconda3\envs\geosa-basa\python.exe'
$config = Join-Path $workspace 'configs\valais_clean_fp32val_seed42_relaunch_20261001.yaml'
$authorization = Join-Path $workspace 'runs\valais_fp32val_relaunch_interrupt_20261001\recovery_authorization.json'
$log = Join-Path $workspace 'runs\valais_fp32val_relaunch_interrupt_20261001\scheduler_launcher.log'
$resultPath = Join-Path $workspace 'runs\valais_fp32val_relaunch_interrupt_20261001\scheduler_result.json'
if (-not (Test-Path -LiteralPath $authorization)) { throw 'Recovery authorization missing' }
if ((Test-Path -LiteralPath $log) -or (Test-Path -LiteralPath $resultPath)) {
    throw 'Scheduler resume artifacts already exist'
}
$auth = Get-Content -LiteralPath $authorization -Raw | ConvertFrom-Json
if ($auth.status -ne 'manual_orphan_recovery_authorized' -or $auth.test_accessed -ne $false) {
    throw 'Recovery authorization is not valid'
}
$run = Join-Path $workspace 'runs\valais_clean_fp32val_seed42_relaunch_20261001'
$statusPath = Join-Path $run 'status.json'
$checkpointPath = Join-Path $run 'last.pt'
$currentStatus = Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
if ($currentStatus.status -ne 'interrupted' -or $currentStatus.epoch -ne 27 -or
    $currentStatus.test_evaluated -ne $false -or
    $currentStatus.recovery_audit_sha256 -ne $auth.orphan_audit_sha256) {
    throw 'Recovery status is not the unique authorized epoch27 transition'
}
if ((Get-FileHash -LiteralPath $statusPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $auth.new_status_sha256 -or
    (Get-FileHash -LiteralPath $checkpointPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $auth.epoch27_checkpoint_sha256) {
    throw 'Recovery status/checkpoint bytes changed'
}
$self = Get-CimInstance Win32_Process -Filter "ProcessId=$PID"
$parent = Get-CimInstance Win32_Process -Filter "ProcessId=$($self.ParentProcessId)"
if ($null -eq $parent -or $parent.Name -notmatch '^(svchost|taskhostw)\.exe$') {
    throw 'Launcher is not hosted by Windows Task Scheduler'
}
Set-Location -LiteralPath $workspace
& $python -m scripts.launch_valais_fp32val --config $config --resume 2>&1 |
    Out-File -LiteralPath $log -Encoding utf8
$exitCode = $LASTEXITCODE
$record = [ordered]@{
    status = if ($exitCode -eq 0) { 'launcher_completed' } else { 'launcher_failed' }
    task_process_id = $PID
    parent_process_id = $self.ParentProcessId
    parent_name = $parent.Name
    exit_code = $exitCode
    finished_utc = [DateTime]::UtcNow.ToString('o')
    authorization_sha256 = (Get-FileHash -LiteralPath $authorization -Algorithm SHA256).Hash.ToLowerInvariant()
    launcher_log_sha256 = (Get-FileHash -LiteralPath $log -Algorithm SHA256).Hash.ToLowerInvariant()
}
$record | ConvertTo-Json -Compress | Set-Content -LiteralPath $resultPath -Encoding utf8
exit $exitCode
