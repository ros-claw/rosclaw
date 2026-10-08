"""Pinned Fields2Cover predictions only; no ROS node, action or credit."""
import json
import math
import time
from pathlib import Path
import fields2cover as f

rows=[]
for name,radius,robot_width,operation_width in [('waffle',.25,.5,.45),('burger',.15,.3,.3)]:
 for label,hl,angle,mode in [('baseline',.5,0.,'BOUSTROPHEDON'),('diagonal',.5,math.pi/4,'BOUSTROPHEDON'),('diagonal_snake',.5,math.pi/4,'SNAKE'),('diagonal_spiral',.5,math.pi/4,'SPIRAL'),('headland45',.45,0.,'BOUSTROPHEDON'),('headland45_snake',.45,0.,'SNAKE')]:
  hl = hl if name == 'waffle' else (.3 if hl == .5 else .35)
  if name == 'burger':label=label.replace('headland45','conservative35')
  start=time.perf_counter();ring=f.LinearRing()
  for x,y in [(-1.5,-1.5),(1.5,-1.5),(1.5,1.5),(-1.5,1.5),(-1.5,-1.5)]:ring.addPoint(f.Point(x,y))
  cell=f.Cell();cell.addRing(ring);cells=f.Cells();cells.addGeometry(cell)
  robot=f.Robot(robot_width,operation_width);robot.setMinTurningRadius(.1);robot.setMaxDiffCurv(200.)
  field=f.HG_Const_gen().generateHeadlands(cells,hl).getGeometry(0)
  sg=f.SG_BruteForce();sg.setAllowOverlap(False);swaths=sg.generateSwaths(angle,operation_width,field)
  gen={'BOUSTROPHEDON':f.RP_Boustrophedon,'SNAKE':f.RP_Snake,'SPIRAL':f.RP_Spiral}[mode]()
  if mode=='SPIRAL':gen.setSpiralSize(4)
  route=gen.genSortedSwaths(swaths)
  curve=f.PP_DubinsCurvesCC();curve.setDiscretization(.1)
  path=f.PP_PathPlanning().planPath(robot,route,curve).discretizeSwath(.1);states=list(path.getStates())
  poses=[{'x':s.point.getX(),'y':s.point.getY(),'yaw':s.angle} for s in states]
  steps=[math.hypot(b['x']-a['x'],b['y']-a['y']) for a,b in zip(poses,poses[1:])]
  clr=min(1.5-max(abs(p['x']),abs(p['y']))-radius for p in poses)
  rows.append({'candidate_id':name+'-'+label,'profile':name,'headland_width':hl,'swath_angle':angle,'route_mode':mode,'frame_id':'map','poses':poses,'physical_radius_m':radius,'robot_width':robot_width,'operation_width':operation_width,'min_turning_radius':.1,'swath_count':swaths.size(),'turn_connector_count':swaths.size()-1,'swath_length_m':sum(swaths.at(i).length() for i in range(swaths.size())),'connector_length_m':sum(s.len for s in states if s.type==f.PathSectionType_TURN),'planned_length_m':sum(steps),'fields2cover_length_m':path.length(),'sampled_clearance_min_m':clr,'max_sample_chord_m':max(steps),'polyline_room_circle_clearance_min_m':clr,'tracking_allowance_screen_m':.05,'offline_tracking_screen_pass':clr>=.05,'plan_generation_time_ms':(time.perf_counter()-start)*1000,'estimated_tracking_risk':'SCREEN_ONLY_NO_PHYSICAL_TRACKING_MODEL','evidence_role':'offline_pinned_F2C_prediction_only','limitations':['For this axis-aligned convex room, minimum circle-to-wall clearance of piecewise-linear center path occurs at a segment endpoint; this does not bound physical tracking error','No Nav2 controller, dynamic occupancy or physical execution is evaluated','All footprint predictions remain separate from measured verifier credit']})
Path('/evidence/candidate-paths.json').write_text(json.dumps(rows,indent=2)+'\n')
print([(p['candidate_id'],round(p['sampled_clearance_min_m'],4),p['offline_tracking_screen_pass']) for p in rows])
