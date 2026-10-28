$ErrorActionPreference = 'Stop'
$studyWorkspace = 'E:\new research\detection'
$studyPython = 'E:\Download\anaconda3\envs\geosa-basa\python.exe'
$studyConfig = Join-Path $studyWorkspace 'configs\valais_c5r5_40ep_runtime_v1_20261020.yaml'
$studyOutput = Join-Path $studyWorkspace 'runs\valais_c5r5_40ep_runtime_v1_20261020'
Set-Location -LiteralPath $studyWorkspace
$env:PYTHONUNBUFFERED = '1'
& $studyPython -u -m scripts.valais_c5r5_runtime run --config $studyConfig `
    1> (Join-Path $studyOutput 'queue.stdout.log') 2> (Join-Path $studyOutput 'queue.stderr.log')
exit $LASTEXITCODE
