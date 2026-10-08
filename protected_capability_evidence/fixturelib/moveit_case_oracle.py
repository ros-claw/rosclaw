"""Independent finite generated-model mathematics and protected actual API traces.

No candidate implementation or robot control is provided by this oracle.
Only generated public numeric test fixtures are accepted.
"""
import math

GROUP = ['slide_x', 'slide_y', 'slide_z', 'roll', 'pitch', 'yaw']


def near(a, b, tol=1e-5):
    assert len(a) == len(b)
    assert all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in a)
    assert all(abs(x-y) <= tol for x, y in zip(a, b)), 'GENERATED_NUMERIC_ORACLE'


def state(q):
    assert len(q) == 6
    near(q, q)
    assert -1-1e-6 <= q[0] <= 1+1e-6 and -1-1e-6 <= q[1] <= 1+1e-6
    assert .1-1e-6 <= q[2] <= 1+1e-6
    assert abs(q[3]) <= 3.140001 and abs(q[4]) <= 1.400001 and abs(q[5]) <= 3.140001


def path(p, start, goal, obstacle=False):
    assert p['solved'] is True and 2 <= len(p['points']) <= 1000
    near(p['points'][0]['q'], start, .002)
    near(p['points'][-1]['q'], goal, .002)
    for row in p['points']:
        state(row['q'])
        near(row['tool_position_m'], row['q'][:3])
        assert row['bounds'] is True
    if obstacle:
        # Exact declared generated geometry: only tool sphere (.04 m) and world
        # sphere (.12 m); verify every linear group segment's closest position.
        for a, b in zip(p['points'], p['points'][1:]):
            a, b = a['q'][:3], b['q'][:3]
            c = [0., 0., .5]
            v = [y-x for x, y in zip(a, b)]
            vv = sum(x*x for x in v)
            u = max(0., min(1., sum((z-x)*d for z, x, d in zip(c, a, v))/vv)) if vv else 0.
            closest = [x+u*d for x, d in zip(a, v)]
            assert sum((x-z)**2 for x, z in zip(closest, c)) >= .16**2-1e-7, 'REPLAN_SEGMENT_GENERATED_COLLISION'


def validate(requests, raw):
    assert raw['status'] == 'ACTUAL_GENERATED_MOVEIT_CASES_COMPLETED_NOT_ORACLE'
    rows = raw['cases']
    assert len(rows) == 10 and [r['case_id'] for r in rows] == [r['id'] for r in requests]
    verdicts = []
    for case, row in zip(requests, rows):
        cid = case['id']
        assert row['input'] == case['request'], 'EXACT_CASE_INPUT_REQUIRED'
        answer = row['Native_answer']
        assert answer.get('passed') is True and isinstance(answer.get('criteria'), list) and answer['criteria'], 'NATIVE_MEANINGFUL_ASSERTIONS_REQUIRED'
        trace = row['observed_actual_interfaces']
        assert 1 <= len(trace) <= 2000
        def records(api):
            return [r for r in trace if r['api'] == api]
        def only(api):
            x = records(api)
            assert len(x) == 1, 'EXACT_ACTUAL_INTERFACE_BINDING'
            return x[0]
        if cid == 'F38_float_arm_variable_map':
            m = only('RobotModel.metadata')['output']
            assert len(m['variables']) == 13 and len(set(m['variables'])) == 13
            assert m['group_variables'] == GROUP and m['KDL_joint_names'] == GROUP
            joints = {j['name']: j for j in m['joints']}
            assert len(joints['floating_base']['variables']) == 7
            assert all(len(joints[n]['variables']) == 1 for n in GROUP)
            indexes = [joints[n]['first_variable_index'] for n in GROUP]
            assert len(set(indexes)) == 6 and all(m['variables'][i] == n for n, i in zip(GROUP, indexes))
            t = only('RobotState.getGlobalLinkTransform')
            near(t['input']['q'], [.2, -.1, .5, 0, 0, 0])
            near(t['output']['position_m'], [.2, -.1, .5])
        elif cid == 'F38_unknown_variable_reject':
            m = only('RobotModel.metadata')['output']
            assert 'unknown_generated_joint' not in m['variables']
            assert answer.get('error_code') == 'UNKNOWN_VARIABLE'
            assert not records('RobotState.getGlobalLinkTransform'), 'NO_UNKNOWN_STATE_ASSIGNMENT'
        elif cid.startswith('F39_'):
            sphere = only('PlanningScene.processCollisionObjectMsg')
            near(sphere['input']['center_m'], case['request']['sphere_center'])
            assert sphere['input']['id'] == sphere['output']['id'] == 'generated_obstacle' and sphere['output']['world_contains'] is True
            near(sphere['output']['center_m'], case['request']['sphere_center'])
            assert abs(sphere['output']['radius_m']-.12)<1e-9
            collision = only('PlanningScene.checkCollision')
            near(collision['input']['q'], case['request']['state'])
            assert collision['output']['collision'] is (cid == 'F39_changed_pose_collision')
        elif cid == 'F40_obstacle_replan':
            plans = records('OMPLPlanner.getPlanningContext.solve')
            assert len(plans) == 2 and records('World.clearObjects')
            sphere = only('PlanningScene.processCollisionObjectMsg')
            near(sphere['input']['center_m'], [0, 0, .5])
            for p in plans:
                near(p['input']['start'], case['request']['start'])
                near(p['input']['goal'], case['request']['goal'])
                assert p['input']['planning_time_s'] <= .5
            clear_index = next(i for i, r in enumerate(trace) if r['api'] == 'World.clearObjects')
            plan_indexes = [i for i, r in enumerate(trace) if r['api'] == 'OMPLPlanner.getPlanningContext.solve']
            sphere_index = trace.index(sphere)
            assert clear_index < plan_indexes[0] < sphere_index < plan_indexes[1]
            path(plans[0]['output'], case['request']['start'], case['request']['goal'])
            path(plans[1]['output'], case['request']['start'], case['request']['goal'], True)
        elif cid == 'F40_colliding_goal_no_plan':
            only('PlanningScene.processCollisionObjectMsg')
            p = only('OMPLPlanner.getPlanningContext.solve')
            near(p['input']['goal'], [0, 0, .5, 0, 0, 0])
            assert p['output']['solved'] is False and p['output']['points'] == []
        elif cid == 'F41_link_contact_frame':
            t = only('RobotState.getGlobalLinkTransform')
            assert t['input']['link'] == 'tool'
            near(t['input']['q'], case['request']['state'])
            near(t['output']['position_m'], [.2, -.1, .5])
            for got, expected in zip(t['output']['rotation'], [[0, -1, 0], [1, 0, 0], [0, 0, 1]]):
                near(got, expected)
            near(answer['contact_world_m'], [.2, -.06, .5])
            near(answer['normal_world'], [0, 1, 0])
            near(answer['lever_arm_world_m'], [0, .04, 0])
        elif cid == 'F41_unknown_contact_frame':
            r = only('RobotModel.hasLinkModel')
            assert r['input']['link'] == 'missing_generated_frame' and r['output']['error'] == 'UNKNOWN_FRAME'
            assert answer.get('error_code') == 'UNKNOWN_FRAME'
        elif cid.startswith('F42_'):
            r = only('CartesianInterpolator.computeCartesianPath.KDL')
            near(r['input']['start'], case['request']['start'])
            near(r['input']['target_m'], case['request']['target_m'])
            p = r['output']
            assert math.isfinite(p['fraction']) and 0 <= p['fraction'] <= 1 and len(p['points']) <= 1000
            for rowp in p['points']:
                state(rowp['q'])
                near(rowp['tool_position_m'], rowp['q'][:3])
                assert rowp['bounds'] is True
            if cid == 'F42_cartesian_path_finite':
                assert p['fraction'] >= 1-1e-8 and len(p['points']) > 1
                near(p['points'][-1]['tool_position_m'], [.2, 0, .5], .002)
                for a, b in zip(p['points'], p['points'][1:]):
                    assert max(abs(x-y) for x, y in zip(a['q'], b['q'])) <= .05
            else:
                assert p['fraction'] < 1-1e-6, 'OUT_OF_BOUNDS_PATH_NOT_FULL_SUCCESS'
        else:
            raise AssertionError('UNKNOWN_CASE')
        verdicts.append({'case': cid, 'direction': case['direction'], 'status': 'PASS_ACTUAL_GENERATED_CPP_LIBRARY_AND_INDEPENDENT_ORACLE'})
    return verdicts
