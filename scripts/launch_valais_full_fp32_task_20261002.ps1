$ErrorActionPreference = 'Stop'
$workspace = 'E:\new research\detection'
$python = 'E:\Download\anaconda3\envs\geosa-basa\python.exe'
$config = Join-Path $workspace 'configs\valais_clean_full_fp32_seed42_20261002.yaml'
$run = Join-Path $workspace 'runs\valais_clean_full_fp32_seed42_20261002'
$preflight = Join-Path $workspace 'runs\valais_full_fp32_preflight_20261002'
$authorization = Join-Path $preflight 'authorization.json'
$taskName = 'Codex-Valais-FullFP32-Seed42-20261002'
$stdout = Join-Path $preflight 'scheduler_launcher_stdout.log'
$stderr = Join-Path $preflight 'scheduler_launcher_stderr.log'
$resultPath = Join-Path $preflight 'scheduler_result.json'

if ((Test-Path -LiteralPath $run) -or (Test-Path -LiteralPath $stdout) -or
    (Test-Path -LiteralPath $stderr) -or (Test-Path -LiteralPath $resultPath)) {
    throw 'Full-FP32 one-shot task already executed'
}
$auth = Get-Content -LiteralPath $authorization -Raw | ConvertFrom-Json
if ($auth.status -ne 'full_fp32_training_authorized' -or $auth.task_name -ne $taskName -or
    $auth.test_accessed -ne $false -or $auth.training_amp -ne $false -or
    $auth.validation_amp -ne $false) {
    throw 'Full-FP32 authorization differs'
}
$task = Get-ScheduledTask -TaskName $taskName -ErrorAction Stop
if ($task.Triggers.Count -ne 0 -or $task.Settings.RestartCount -ne 0 -or
    $task.Settings.StopIfGoingOnBatteries -ne $false) {
    throw 'Scheduled task trigger/retry/power policy differs'
}
$self = Get-CimInstance Win32_Process -Filter "ProcessId=$PID"
$parent = Get-CimInstance Win32_Process -Filter "ProcessId=$($self.ParentProcessId)"
if ($null -eq $parent -or $parent.Name -notmatch '^(svchost|taskhostw)\.exe$') {
    throw 'Full-FP32 launcher is not hosted by Task Scheduler'
}
foreach ($binding in $auth.bound_sha256.PSObject.Properties) {
    $actual = (Get-FileHash -LiteralPath $binding.Name -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $binding.Value) { throw "Bound training source changed: $($binding.Name)" }
}
$gpu = & 'C:\Windows\System32\nvidia-smi.exe' -i 0 --query-gpu=uuid,temperature.gpu --format=csv,noheader,nounits
$parts = $gpu.Trim().Split(',')
if ($parts[0].Trim() -ne $auth.gpu_uuid -or [int]$parts[1].Trim() -ge 85) {
    throw 'GPU UUID or temperature differs from the frozen training policy'
}
Set-Location -LiteralPath $workspace
$exitCode = 1
$errorText = $null
try {
    $arguments = @('-m', 'scripts.run_valais_clean_guarded', '--config', ('"' + $config + '"'))
    $process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $workspace `
        -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr `
        -Wait -PassThru -ErrorAction Stop
    $exitCode = $process.ExitCode
    if ($exitCode -ne 0) { throw "Guarded training launcher exited with $exitCode" }
    $status = Get-Content -LiteralPath (Join-Path $run 'status.json') -Raw | ConvertFrom-Json
    $guard = Get-Content -LiteralPath (Join-Path $run 'temperature_guard_state.json') -Raw | ConvertFrom-Json
    $history = @(Get-Content -LiteralPath (Join-Path $run 'history.json') -Raw | ConvertFrom-Json)
    if ($status.status -ne 'completed' -or $status.epochs -ne 100 -or
        $status.test_evaluated -ne $false -or $guard.status -ne 'training_process_exited' -or
        $history.Count -ne 100 -or $history[-1].epoch -ne 100 -or
        (Test-Path -LiteralPath (Join-Path $run 'temperature_guard.lock'))) {
        throw 'Training/guard/history did not complete under the frozen protocol'
    }
    foreach ($binding in $auth.bound_sha256.PSObject.Properties) {
        $actual = (Get-FileHash -LiteralPath $binding.Name -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -ne $binding.Value) { throw "Bound training source changed after run: $($binding.Name)" }
    }
    $completion = [ordered]@{
        status = 'guarded_full_fp32_clean_completed'
        test_accessed = $false
        training_amp = $false
        validation_amp = $false
        epochs = 100
        best_epoch = $status.best_epoch
        best_val_f1 = $status.best_val_f1
        authorization_sha256 = (Get-FileHash -LiteralPath $authorization -Algorithm SHA256).Hash.ToLowerInvariant()
        config_sha256 = (Get-FileHash -LiteralPath $config -Algorithm SHA256).Hash.ToLowerInvariant()
        status_sha256 = (Get-FileHash -LiteralPath (Join-Path $run 'status.json') -Algorithm SHA256).Hash.ToLowerInvariant()
        guard_state_sha256 = (Get-FileHash -LiteralPath (Join-Path $run 'temperature_guard_state.json') -Algorithm SHA256).Hash.ToLowerInvariant()
        history_sha256 = (Get-FileHash -LiteralPath (Join-Path $run 'history.json') -Algorithm SHA256).Hash.ToLowerInvariant()
        best_checkpoint_sha256 = (Get-FileHash -LiteralPath (Join-Path $run 'best.pt') -Algorithm SHA256).Hash.ToLowerInvariant()
        last_checkpoint_sha256 = (Get-FileHash -LiteralPath (Join-Path $run 'last.pt') -Algorithm SHA256).Hash.ToLowerInvariant()
        source_snapshot_sha256 = (Get-FileHash -LiteralPath (Join-Path $run 'source_sha256.json') -Algorithm SHA256).Hash.ToLowerInvariant()
        data_audit_sha256 = (Get-FileHash -LiteralPath (Join-Path $run 'data_audit.json') -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    $completion | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $run 'guard_completion.json') -Encoding utf8
} catch {
    $errorText = $_.Exception.Message
} finally {
    $record = [ordered]@{
        task_name = $taskName
        task_process_id = $PID
        parent_process_id = $self.ParentProcessId
        parent_name = $parent.Name
        launcher_exit_code = $exitCode
        status = if ($null -eq $errorText) { 'completed' } else { 'failed' }
        error = $errorText
        finished_utc = [DateTime]::UtcNow.ToString('o')
    }
    $record | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $resultPath -Encoding utf8
}
if ($null -ne $errorText) { throw $errorText }
