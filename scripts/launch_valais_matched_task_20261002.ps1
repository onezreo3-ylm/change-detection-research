$ErrorActionPreference = 'Stop'
$taskWorkspace = 'E:\new research\detection'
$taskPython = 'E:\Download\anaconda3\envs\geosa-basa\python.exe'
$taskConfig = Join-Path $taskWorkspace 'configs\valais_matched_study_v1_20261002.yaml'
$taskOutput = Join-Path $taskWorkspace 'runs\valais_matched_study_v1_20261002'
Set-Location -LiteralPath $taskWorkspace
$env:PYTHONUNBUFFERED = '1'
& $taskPython -u -m scripts.valais_matched_study run --config $taskConfig `
    1> (Join-Path $taskOutput 'queue.stdout.log') 2> (Join-Path $taskOutput 'queue.stderr.log')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
