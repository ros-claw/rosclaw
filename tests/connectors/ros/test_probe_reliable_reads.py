"""Operator transport contracts use offered QoS, with no topic-name exemption."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


def discover(offers, *, selected=True, topic='/renamed/depth'):
    path=Path(os.environ.get('ROSCLAW_PROBE_UNDER_TEST',
        str(Path(__file__).resolve().parents[3]/'integrations/ros_probe/ros2/probe.py')))
    tree=ast.parse(path.read_text())
    latched=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='latched_observation')
    method=next(n for c in tree.body if isinstance(c,ast.ClassDef) and c.name=='ReadOnlyProbe'
                for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='discover_reads')
    namespace=dict(QoSProfile=lambda **kw:NS(**kw),
        ReliabilityPolicy=NS(RELIABLE=1,BEST_EFFORT=2),
        DurabilityPolicy=NS(TRANSIENT_LOCAL=1),READ_TYPES={'sensor_msgs/msg/PointCloud2'},
        TF_ATTRIBUTION_TOPIC='/attribution',get_message=lambda t:t)
    exec(compile(ast.Module(body=[latched,method],type_ignores=[]),str(path),'exec'),namespace)
    calls=[]
    probe=NS(subscriptions_by_topic={},errors=[],observe=lambda *_:None,
        reliable_read_types={'sensor_msgs/msg/PointCloud2'} if selected else set(),
        get_topic_names_and_types=lambda:[(topic,['sensor_msgs/msg/PointCloud2'])],
        get_publishers_info_by_topic=lambda _:[NS(qos_profile=NS(reliability=r,durability=2)) for r in offers],
        get_service_names_and_types=lambda:[],
        create_subscription=lambda *a:calls.append(a) or object())
    namespace['discover_reads'](probe)
    namespace['discover_reads'](probe)
    return probe,calls


@pytest.mark.parametrize('topic',['/depth/points','/another_namespace/cloud'])
def test_explicit_reliable_contract_requests_latest_complete_sample(topic):
    probe,calls=discover([1],topic=topic)
    assert len(calls)==1 and not probe.errors
    assert calls[0][-1].reliability==1 and calls[0][-1].depth==1


@pytest.mark.parametrize('offered',[1,2])
def test_default_sensor_profile_stays_compatible_without_an_operator_contract(offered):
    probe,calls=discover([offered],selected=False)
    assert len(calls)==1 and calls[0][-1].reliability==2 and not probe.errors


@pytest.mark.parametrize('offers',[[2],[1,2],[0]])
def test_incompatible_or_unknown_offers_never_silently_downgrade_contract(offers):
    probe,calls=discover(offers)
    assert not calls and len(probe.errors)==1
    assert 'publisher contract mismatch' in probe.errors[0]


def test_discovery_without_publishers_waits_instead_of_guessing_reliable_qos():
    probe,calls=discover([])
    assert not calls and not probe.errors
