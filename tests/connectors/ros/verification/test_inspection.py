"""Passage classification must retain unknown cells and separate obstacles from blockage."""
import pytest
from rosclaw.connectors.ros.verification.inspection import classify_passage


def classify(free=(), occupied=(), **kw):
    return classify_passage(width=10,height=10,observed_free=free,occupied=occupied,
                            clearance_cells=1,minimum_coverage=.7,**kw)


def test_no_sensor_evidence_is_unknown():
    assert classify()['result']=='UNKNOWN'


def test_seen_extra_obstacle_does_not_mean_blocked():
    occ={(5,5)}
    free={(x,y) for x in range(10) for y in range(10)}-occ
    out=classify(free,occ)
    assert out['result']=='CLEAR'
    assert out['obstacle_cells']==1


def test_actual_cross_passage_barrier_is_obstructed():
    out=classify(occupied={(x,5) for x in range(10)})
    assert out['result']=='OBSTRUCTED'


def test_missing_cells_are_not_free_space():
    out=classify(free={(x,y) for x in range(10) for y in range(4)})
    assert out['result']=='UNKNOWN'


def test_stale_evidence_cannot_classify_barrier():
    assert classify(occupied={(x,5) for x in range(10)},evidence_valid=False)['result']=='UNKNOWN'


def test_inflated_obstacle_can_close_narrow_gap():
    assert classify(occupied={(x,5) for x in range(10) if x!=5})['result']=='OBSTRUCTED'


@pytest.mark.parametrize('value',[float('nan'),-1,1.1])
def test_invalid_coverage_rejected(value):
    with pytest.raises(ValueError):
        classify_passage(width=10,height=10,observed_free=[],occupied=[],clearance_cells=1,minimum_coverage=value)


def test_out_of_region_evidence_rejected():
    with pytest.raises(ValueError): classify(occupied={(10,0)})


def test_occupied_wins_over_ground_return():
    full={(x,y) for x in range(10) for y in range(10)}
    assert classify(free=full,occupied={(x,5) for x in range(10)})['result']=='OBSTRUCTED'


def test_report_retains_evidence_and_scope():
    out=classify()
    assert out['result']!='PASS'
    assert out['unknown_cells']==100
    assert 'sensor' in out['scope']
