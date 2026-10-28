$ErrorActionPreference = 'Stop'
$workspace = 'E:\new research\detection'
$folder = Join-Path $workspace 'runs\valais_scheduler_liveness_probe_v2_20261001'
if (Test-Path -LiteralPath $folder) { throw 'Liveness probe v2 directory already exists' }
New-Item -ItemType Directory -Path $folder | Out-Null
$self = Get-CimInstance Win32_Process -Filter "ProcessId=$PID"
$parent = Get-CimInstance Win32_Process -Filter "ProcessId=$($self.ParentProcessId)"
if ($null -eq $parent -or $parent.Name -notmatch '^(svchost|taskhostw)\.exe$') {
    throw 'Probe is not hosted by Task Scheduler'
}
$start = [DateTime]::UtcNow
for ($i = 0; $i -le 130; $i++) {
    $current = [DateTime]::UtcNow
    $record = [ordered]@{
        status = 'running'
        kind = 'scheduler_liveness_no_gpu_no_data'
        process_id = $PID
        parent_process_id = $self.ParentProcessId
        parent_name = $parent.Name
        iteration = $i
        elapsed_seconds = [Math]::Round(($current - $start).TotalSeconds, 3)
        sampled_utc = $current.ToString('o')
        data_accessed = $false
        gpu_accessed = $false
    }
    $temp = Join-Path $folder 'heartbeat.tmp'
    $state = Join-Path $folder 'heartbeat.json'
    $backup = Join-Path $folder 'heartbeat.backup'
    $record | ConvertTo-Json -Compress | Set-Content -LiteralPath $temp -Encoding utf8
    if (Test-Path -LiteralPath $state) {
        [IO.File]::Replace($temp, $state, $backup)
        if (Test-Path -LiteralPath $backup) { Remove-Item -LiteralPath $backup -Force }
    } else {
        [IO.File]::Move($temp, $state)
    }
    if ($i -lt 130) { Start-Sleep -Seconds 30 }
}
$complete = [ordered]@{
    status = 'completed'
    kind = 'scheduler_liveness_no_gpu_no_data'
    process_id = $PID
    parent_process_id = $self.ParentProcessId
    parent_name = $parent.Name
    duration_seconds = [Math]::Round(([DateTime]::UtcNow - $start).TotalSeconds, 3)
    data_accessed = $false
    gpu_accessed = $false
}
$complete | ConvertTo-Json -Compress |
    Set-Content -LiteralPath (Join-Path $folder 'completion.json') -Encoding utf8
