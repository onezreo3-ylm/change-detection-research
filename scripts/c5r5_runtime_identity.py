"""Identity and exact accounting after the user-requested runtime pilot."""
from pathlib import Path
import statistics
import yaml
from pcdc.config import UniqueKeyLoader
from scripts import five_view_baselines as f
from scripts import c5r5_40ep_identity as parent

FLAVOR='matched_five_view_baseline_40ep_runtime_v1'
EPOCHS,UPDATES,PAIR_VIEWS=40,110280,2205000


def load_study(path):
    source=Path(path).resolve();settings=yaml.load(source.read_bytes(),Loader=UniqueKeyLoader)
    expected={'schema_version':1,'parent_config':'valais_c5r5_40ep_seed42_v1_20261020.yaml',
        'output_dir':'../runs/valais_c5r5_40ep_runtime_v1_20261020',
        'protocol_document':'../docs/C5R5_RUNTIME_PROTOCOL_2026-10-20.md',
        'speed_report':'../runs/valais_c5r5_speed_pilot_v1_20261020/report.json',
        'bf16_report':'../runs/valais_c5r5_bf16_pilot_v1_20261020/report.json',
        'optimization_mode':settings.get('optimization_mode'),'minimum_unittests':407}
    if settings!=expected or settings['optimization_mode'] not in ('reference','pinned_layout_device_metrics','bf16_training_fp32_validation'):
        raise ValueError('Unreviewed runtime protocol or mode')
    _,v,paths,base=parent.load_study(source.parent/settings['parent_config'])
    paths=dict(paths,parent_40_output_dir=paths['output_dir'],parent_40_config=source.parent/settings['parent_config'])
    for key in ('output_dir','protocol_document','speed_report','bf16_report'):
        paths[key]=(source.parent/settings[key]).resolve()
    v=dict(v,name='valais_uniform_c5r5_40ep_runtime_v1_20261020',training_flavor=FLAVOR,
        optimization_mode=settings['optimization_mode'],minimum_unittests=407)
    return source,v,paths,base


def implementation_files():
    files=parent.implementation_files()
    for name in ('scripts/c5r5_runtime_identity.py','scripts/five_view_40ep_runtime.py',
        'scripts/valais_c5r5_runtime.py','scripts/five_view_speed.py','scripts/profile_five_view_speed.py','scripts/five_view_bf16.py','scripts/profile_five_view_bf16.py'):
        files[name]=f.ROOT/name
    return files


def role_debit(paths,role):
    return f.read(paths['output_dir']/'authorization.json')['engineering_seconds_debited'][role]


def project(profile,role,mode):
    row=profile['roles'][role]
    if mode=='bf16_training_fp32_validation':
        mean=statistics.mean(row['timings_seconds'][8:])
        debit=row['budget_projection']['engineering_seconds_debited']
    else:
        name='candidate' if mode=='pinned_layout_device_metrics' else 'reference'
        mean=statistics.mean(row['timings'][name]['timings_seconds'][4:])
        debit=row['total_engineering_debit_seconds']
    if not 0<mean<1800 or not 0<debit<2400:raise ValueError('Invalid runtime timings/debit')
    worker=mean*110280*1.25+40*60
    return {'steady_mean_seconds_per_step':mean,'engineering_seconds_debited':debit,
        'projected_worker_seconds_including_margin':worker,'projected_total_role_seconds':worker+debit,
        'within_frozen_budget':worker+debit<=86400,'maximum_role_seconds':86400}
