"""Admission rejects candidates without deleting work or manufacturing credit."""
import numpy as np
import pytest
from rosclaw.connectors.ros.public_coverage.coverage_ledger import CoverageLedger
from rosclaw.connectors.ros.public_coverage.coverage_work import select
from rosclaw.connectors.ros.public_coverage.coverage_variants import prefix


class Grid:
    data=np.zeros((100,100),np.int16)
    reachable=np.ones((100,100),bool)
    origin=np.array([0.,0.])
    resolution=.05
    def safe_path(self,path): return bool(np.all(np.asarray(path)[:,1]>=0))


def test_far_work_can_be_rejected_while_near_work_is_selected_and_debt_is_intact():
    ledger=CoverageLedger()
    near=np.array([[.4,.4],[.9,.4]])
    far=np.array([[3.,3.],[4.,3.]])
    ledger.opportunities=ledger._sweep(*near)|ledger._sweep(*far)
    original=set(ledger.opportunities); rejected=[]
    def prepare(path,transit):
        if transit>1: rejected.append(path); return None
        return path
    chosen=select([far,near],Grid(),ledger,near[0],{},continuous_only=True,prepare_candidate=prepare)
    np.testing.assert_array_equal(chosen[0],near)
    assert len(rejected)==1 and ledger.opportunities==original and not ledger.cleaned


def test_shorter_unissued_prefix_is_rescored_and_tail_remains_unswept():
    ledger=CoverageLedger();path=np.array([[.4,.4],[1.4,.4]])
    ledger.opportunities=ledger._sweep(*path); original=set(ledger.opportunities)
    chosen=select([path],Grid(),ledger,path[0],{},continuous_only=True,
        prepare_candidate=lambda p,t:prefix(p,.2))
    assert np.linalg.norm(np.diff(chosen[0],axis=0),axis=1).sum()==pytest.approx(.2)
    assert chosen[-1]<len(original)
    assert ledger.opportunities==original and not ledger.cleaned


@pytest.mark.parametrize('adapt',[
    lambda p:p+[.01,0], lambda p:np.array([p[0],p[-1]+[1.,0]]),
    lambda p:np.array([p[0],p[0]+[0,.1]]),
])
def test_candidate_hook_cannot_shift_entry_extend_or_replace_a_stroke(adapt):
    ledger=CoverageLedger();path=np.array([[.4,.4],[1.4,.4]])
    ledger.opportunities=ledger._sweep(*path)
    with pytest.raises(ValueError,match='original unissued prefix'):
        select([path],Grid(),ledger,path[0],{},prepare_candidate=lambda p,t:adapt(p))


def test_refusal_of_every_candidate_has_no_completion_credit():
    ledger=CoverageLedger();path=np.array([[.4,.4],[1.4,.4]])
    ledger.opportunities=ledger._sweep(*path)
    assert select([path],Grid(),ledger,path[0],{},prepare_candidate=lambda p,t:None) is None
    assert not ledger.cleaned and ledger.opportunities
