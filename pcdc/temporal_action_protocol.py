"""Exact count bookkeeping for mutually exclusive fixed temporal interventions."""
from itertools import product

from .local_composition_protocol import f1_fraction

ASSIGNMENTS=tuple(product(range(4),repeat=4))

def assignment_counts(base,regional_deltas,assignment):
    if (len(base)!=4 or len(regional_deltas)!=4 or len(assignment)!=4
            or any(type(x) is not int or x<0 for x in base)
            or any(type(i) is not int or i not in range(4) for i in assignment)
            or any(len(region)!=3 or any(len(delta)!=4 or any(type(x) is not int for x in delta)
                   for delta in region) for region in regional_deltas)):
        raise ValueError('Four ordered mutually exclusive regions and integer TN/FP/FN/TP required')
    result=[base[k]+sum(regional_deltas[r][a-1][k] for r,a in enumerate(assignment) if a) for k in range(4)]
    if (min(result)<0 or sum(result)!=sum(base) or
            result[0]+result[1]!=base[0]+base[1] or result[2]+result[3]!=base[2]+base[3]):
        raise ArithmeticError('Assignment changed label support or made impossible confusion counts')
    return result

def expanded_menu(row):
    base,deltas=row['a0_counts'],row['regional_deltas']
    assignment_counts(base,deltas,(0,0,0,0))
    if any(sum(d)!=0 or d[0]+d[1]!=0 or d[2]+d[3]!=0 for region in deltas for d in region):
        raise ArithmeticError('A local alternative changed label/pixel support')
    menu=[list(base)]
    for region in deltas:
        menu=[[p[k]+d[k] for k in range(4)] for p in menu for d in [(0,0,0,0),*region]]
    if any(min(p)<0 for p in menu):raise ArithmeticError('Impossible composed confusion counts')
    return menu

def old_two_action_menu(row):
    return [assignment_counts(row['a0_counts'],row['regional_deltas'],
        tuple(3 if subset&(1<<r) else 0 for r in range(4))) for subset in range(16)]

def require_old_menu_replay(row,old):
    if (row['a0_counts']!=old['a0_counts'] or row['full_both_counts']!=old['full_adapter_counts']
            or old_two_action_menu(row)!=old['menus']['reference_replay_joint']):
        raise ArithmeticError('Original a0/full-both or old16 local action counts changed')

def joint_resource_gates(new_mf1,old_mf1,constant_means,sign_fractions,families,v):
    if len(constant_means)!=7:raise ValueError('All seven frozen constant controls are required')
    return dict(oracle_margin=new_mf1-max(constant_means.values())>=v['minimum_oracle_mf1_gain_over_best_seven_constants'],
        temporal_increment=new_mf1-old_mf1>=v['minimum_oracle_mf1_gain_over_old_two_action_oracle'],
        positive_fraction=sign_fractions['positive']>=v['minimum_positive_action_fraction'],
        negative_fraction=sign_fractions['negative']>=v['minimum_negative_action_fraction'],
        positive_families=sum('positive' in s for s in families.values())>=v['minimum_families_each_sign'],
        negative_families=sum('negative' in s for s in families.values())>=v['minimum_families_each_sign'])
