from types import SimpleNamespace
def message(data, resolution=.05):
    height,width=data.shape
    return SimpleNamespace(data=data.ravel().tolist(),info=SimpleNamespace(
        width=width,height=height,resolution=resolution,
        origin=SimpleNamespace(position=SimpleNamespace(x=0.,y=0.),
                               orientation=SimpleNamespace(x=0.,y=0.,z=0.,w=1.))))

"""Validate executable sweep geometry and physical boundary opportunities."""
import unittest
import numpy as np
from rosclaw.connectors.ros.public_coverage.coverage_paths import densify,remaining,boundary_strokes,_simplify_and_split
from rosclaw.connectors.ros.public_coverage.grid import Grid



class CoveragePolylines(unittest.TestCase):
    def test_direct_sweep_preserves_line_and_both_endpoints(self):
        path=densify([[1.,1.],[2.,1.]],step=.025)
        np.testing.assert_array_equal(path[[0,-1]],[[1.,1.],[2.,1.]])
        self.assertTrue((path[:,1]==1.).all())
        self.assertLessEqual(np.linalg.norm(np.diff(path,axis=0),axis=1).max(),.025+1e-12)

    def test_resume_does_not_retrace_completed_sweep_or_lose_last_endpoint(self):
        path=densify([[0.,0.],[1.,0.]])
        suffix=remaining(path,[.6,.005])
        self.assertGreaterEqual(suffix[1:,0].min(),.6)
        np.testing.assert_array_equal(suffix[-1],[1.,0.])
        np.testing.assert_array_equal(remaining(path,[1.,.02])[-1],[1.,0.])

    def test_boundary_routes_are_swept_safe_and_extend_beyond_center_only_edges(self):
        data=np.full((60,60),100,dtype=int);data[5:55,5:55]=0
        grid=Grid(message(data),(1.,1.));strokes=boundary_strokes(grid)
        self.assertTrue(strokes)
        points=np.vstack(strokes)
        self.assertLess(points[:,0].min(),min(grid.xy(cell)[0] for cell in grid.centers)-.005)
        for stroke in strokes:
            self.assertTrue(grid.safe_path(stroke))
            self.assertLessEqual(np.linalg.norm(np.diff(stroke,axis=0),axis=1).max(),.025+1e-12)

    def test_boundary_paths_do_not_cut_an_interior_furniture_corner(self):
        data=np.full((70,80),100,dtype=int);data[5:65,5:75]=0;data[25:45,30:50]=100
        grid=Grid(message(data),(1.,1.));strokes=boundary_strokes(grid)
        self.assertGreater(len(strokes),4)
        for stroke in strokes:self.assertTrue(grid.safe_path(stroke))

    def test_closed_smooth_route_has_distinct_native_action_endpoints(self):
        grid=Grid(message(np.zeros((80,80),dtype=int)),(1.,1.))
        angles=np.linspace(0.,2*np.pi,65)
        loop=np.column_stack([2.+.5*np.cos(angles),2.+.5*np.sin(angles)])
        strokes=_simplify_and_split(loop,grid)
        self.assertEqual(len(strokes),2)
        for stroke in strokes:
            self.assertGreater(np.linalg.norm(stroke[-1]-stroke[0]),.005)
            self.assertTrue(grid.safe_path(stroke))


if __name__=='__main__':unittest.main()
