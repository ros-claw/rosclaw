"""Independent finite software/model interface oracle, never robot success."""
import math
IDS=["F43_normalized_model_space","F43_explicit_saved_stats_profile","F44_actual_policy_chunk_consumption","F44_generated_feedback_second_observation","F45_label_frame_freshness_match","F45_no_grounding_blank_or_stale","F46_reverse_edge_conflict","F46_persistent_endpoint_and_disjoint_goal"]
def finite(x):return type(x)in(int,float)and math.isfinite(x)
def vector(v,n):assert type(v)is list and len(v)==n and all(finite(x)for x in v)
def actions(v):
 assert type(v)is list and len(v)==1 and type(v[0])is list and len(v[0])==50
 for row in v[0]:vector(row,6)
 return [x for row in v[0]for x in row]
def validate_pure(cases,answers,calls):
 assert {r["id"]for r in answers}=={c["id"]for c in cases}and len(answers)==4
 expected=[]
 for c in cases:
  for sub in c.get("subcases",[c]):expected.append({"module":c["direction"],"input":sub["input"],"result":sub["expected"]})
 assert calls==expected,"ACTUAL_EXISTING_NATIVE_SOLVE_CALL_AND_INPUT_REQUIRED"
 for c in cases:
  row=next(r for r in answers if r["id"]==c["id"]);assert row["passed"]is True and type(row["criteria"])is list and row["criteria"]
  assert row["outputs"]==[v["expected"]for v in c.get("subcases",[c])]
 return [{"id":c["id"],"status":"PASS_FINITE_EXISTING_NATIVE_STRUCTURED_SOFTWARE"}for c in cases]
def validate_models(outputs,observed,requests,stats):
 assert len(outputs)==len(observed["predict"])==len(observed["preprocessor"])==len(observed["postprocessor"])==2
 assert 1<=len(observed["loaded"])<=2 and observed["module_calls"]
 for load in observed["loaded"]:
  assert load["actual_parameter_count"]>=400000000 and load["flow_matching_num_steps"]==10
  assert len(load["sampled_checkpoint_comparisons"])==16 and all(x["equal"]is True for x in load["sampled_checkpoint_comparisons"])
 for i,out in enumerate(outputs):
  assert out["dependencies_provenance"]=="OFFICIAL_LEROBOT_DEFAULT_BACKEND"and out["mock_inference_not_actual_nn"]is False
  assert out["physical_action_executed"]is False and out["device"]=="cpu"and out["action_shape"]==[1,50,6]
  assert out["action_space"]["robot_units_verified"]is False and out["action_space"]["physical_calibration_established"]is False
  assert out["action_space"]["g1_rh56_mapping"]=="NOT_ESTABLISHED_NO_CONVERSION"
  a=actions(out["actions"]);pred=observed["predict"][i]
  assert pred["device"]=="cpu"and pred["raw_actions"]["shape"]==[1,50,6]and pred["raw_actions"]["finite"]is True
  raw=actions(pred["raw_actions"]["values"]);post=observed["postprocessor"][i]
  assert actions(post["raw_before_post"]["values"])==raw and actions(post["postprocessed"]["values"])==a
  got=observed["preprocessor"][i]["observation.state"];assert got["finite"]is True and got["shape"]==[1,6]
  expected=requests[0]["state"]if i==0 else outputs[0]["actions"][0][0]
  vector(expected,6);vector(got["values"][0],6)
  assert all(abs(x-y)<=1e-6*max(1,abs(y))for x,y in zip(got["values"][0][:6],expected)),"ACTUAL_FIRST_ACTION_TO_SECOND_INPUT_REQUIRED"
  for camera in ["camera1","camera2","camera3"]:
   t=observed["preprocessor"][i]["observation.images."+camera];assert t["finite"]is True and t["device"]=="cpu"and len(t["sha256"])==64
  if i==0:
   assert out["action_space"]["kind"]=="NORMALIZED_MODEL_ACTION"and out["action_space"]["profile"]is None
   assert raw==a
  else:
   assert out["action_space"]["kind"]=="ROBOT_PROFILE_ACTION"and out["action_space"]["profile"]=="so100.buffer.action"
   assert out["action_space"]["stats_sha256"]==stats["stats_sha256"]and out["action_space"]["action_mean"]==stats["mean"]and out["action_space"]["action_std"]==stats["std"]
   expected_post=[x*stats["std"][j%6]+stats["mean"][j%6]for j,x in enumerate(raw)]
   assert all(abs(x-y)<=1e-4*max(1,abs(y))for x,y in zip(a,expected_post)),"OFFICIAL_PINNED_PROFILE_POSTPROCESS_VALUES_REQUIRED"
 return [{"id":c,"status":"PASS_ACTUAL_PRETRAINED_MODEL_COORDINATE_INTERFACE_NOT_PHYSICAL"}for c in IDS[:4]]
