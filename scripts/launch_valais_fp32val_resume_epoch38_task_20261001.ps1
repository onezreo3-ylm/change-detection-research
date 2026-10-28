$ErrorActionPreference = 'Stop'
$workspace = 'E:\new research\detection'
$python = 'E:\Download\anaconda3\envs\geosa-basa\python.exe'
$config = Join-Path $workspace 'configs\valais_clean_fp32val_seed42_relaunch_20261001.yaml'
$recoveryConfig = Join-Path $workspace 'configs\valais_epoch38_supervisor_resume_20261001.yaml'
$folder = Join-Path $workspace 'runs\valais_fp32val_supervisor_loss_20261001'
$authorization = Join-Path $folder 'epoch38_resume_authorization.json'
$log = Join-Path $folder 'epoch38_scheduler_launcher.log'
$resultPath = Join-Path $folder 'epoch38_scheduler_result.json'
$run = Join-Path $workspace 'runs\valais_clean_fp32val_seed42_relaunch_20261001'
$taskName = 'Codex-Valais-FP32Val-Resume-Epoch38-20261001'
if (-not (Test-Path -LiteralPath $authorization)) { throw 'Epoch38 authorization missing' }
if ((Test-Path -LiteralPath $log) -or (Test-Path -LiteralPath $resultPath)) {
    throw 'Epoch38 task action already executed'
}
$auth = Get-Content -LiteralPath $authorization -Raw | ConvertFrom-Json
if ($auth.status -ne 'manual_epoch38_resume_authorized' -or
    $auth.task_name -ne $taskName -or $auth.test_accessed -ne $false -or
    $auth.last_complete_epoch -ne 38 -or $auth.automatic_retry -ne $false) {
    throw 'Epoch38 authorization identity differs'
}
$task = Get-ScheduledTask -TaskName $taskName -ErrorAction Stop
if ($task.Triggers.Count -ne 0 -or $task.Settings.RestartCount -ne 0 -or
    $task.Settings.StopIfGoingOnBatteries -ne $false) {
    throw 'Scheduled task has an unsafe trigger or retry/power setting'
}
$self = Get-CimInstance Win32_Process -Filter "ProcessId=$PID"
$parent = Get-CimInstance Win32_Process -Filter "ProcessId=$($self.ParentProcessId)"
if ($null -eq $parent -or $parent.Name -notmatch '^(svchost|taskhostw)\.exe$') {
    throw 'Resume process is not hosted by Task Scheduler'
}
$statusPath = Join-Path $run 'status.json'
$checkpointPath = Join-Path $run 'last.pt'
$status = Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
if ($status.status -ne 'interrupted' -or $status.epoch -ne 38 -or
    $status.manual_resume_required -ne $true -or $status.test_evaluated -eq $true) {
    throw 'Run is not the authorized interrupted epoch38 state'
}
$checks = @(
    @($statusPath, $auth.interrupted_status_sha256),
    @($checkpointPath, $auth.checkpoint_sha256),
    @($config, $auth.study_config_sha256),
    @($recoveryConfig, $auth.recovery_config_sha256),
    @($PSCommandPath, $auth.scheduler_wrapper_sha256)
)
foreach ($check in $checks) {
    if ((Get-FileHash -LiteralPath $check[0] -Algorithm SHA256).Hash.ToLowerInvariant() -ne $check[1]) {
        throw "Bound resume file changed: $($check[0])"
    }
}
Set-Location -LiteralPath $workspace
& $python -m scripts.launch_valais_fp32val --config $config --resume 2>&1 |
    Out-File -LiteralPath $log -Encoding utf8
$exitCode = $LASTEXITCODE
$record = [ordered]@{
    status = if ($exitCode -eq 0) { 'launcher_completed' } else { 'launcher_failed' }
    task_name = $taskName
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
