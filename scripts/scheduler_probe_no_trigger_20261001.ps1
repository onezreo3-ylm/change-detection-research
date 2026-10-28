$ErrorActionPreference = 'Stop'
$workspace = 'E:\new research\detection'
$output = Join-Path $workspace 'runs\valais_scheduler_no_trigger_probe_20261001.json'
if (Test-Path -LiteralPath $output) { throw 'Probe output already exists' }
$self = Get-CimInstance Win32_Process -Filter "ProcessId=$PID"
$parent = Get-CimInstance Win32_Process -Filter "ProcessId=$($self.ParentProcessId)"
$record = [ordered]@{
    status = 'completed'
    kind = 'scheduler_no_trigger_probe_only'
    created_utc = [DateTime]::UtcNow.ToString('o')
    process_id = $PID
    parent_process_id = $self.ParentProcessId
    parent_name = if ($null -eq $parent) { $null } else { $parent.Name }
    data_accessed = $false
    gpu_accessed = $false
}
$record | ConvertTo-Json -Compress | Set-Content -LiteralPath $output -Encoding utf8
