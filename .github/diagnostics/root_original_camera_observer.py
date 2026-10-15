"""External readonly return observer; no added oracle/model/GL invocation."""
from pathlib import Path
import hashlib,json,os,sys,types
import pytest
TARGET='tests/sim/test_public_snapshot_camera_fidelity.py::test_actual_cli_default_snapshot_observe_passes_strong_camera_binding'
PINS={'tests/sim/test_public_snapshot_camera_fidelity.py':'c48afc8cf90e0a0f4cd319c4ba8422449f5cbe24efd6b7617eae9c8c1fde992e','benchmarks/harnessbench/vision_oracle.py':'505477a730a4c0ea4d400aaa3be15bae1e972bd5aeac4a8e7c7dcdd755bb8b50'}
OBS={'target_calls':0,'verify_calls':0,'returns':[],'phases':[],'errors':[],'original_profile_restored':False}
H=lambda b:hashlib.sha256(b).hexdigest()
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
 if item.nodeid!=TARGET:
  yield;return
 OBS['target_calls']+=1;previous=sys.getprofile();installed=False
 try:
  root=Path(str(item.config.rootpath)).resolve()
  for rel,sha in PINS.items():
   p=root/rel;assert p.is_file()and not p.is_symlink()and p.stat().st_size<=131072 and H(p.read_bytes())==sha,'SOURCE_PIN'
  fn=item.obj.__globals__['verify_camera'];assert type(fn)is types.FunctionType and fn.__module__=='benchmarks.harnessbench.vision_oracle'and Path(fn.__code__.co_filename).resolve()==root/'benchmarks/harnessbench/vision_oracle.py','ORIGINAL_CODE_IDENTITY'
  assert previous is None,'EXISTING_PROFILE_LEFT_UNCHANGED'
  code=fn.__code__
  def profile(frame,event,arg):
   if frame.f_code is not code:return
   if event=='call':OBS['verify_calls']+=1
   if event=='return':
    if len(OBS['returns'])>=2:OBS['errors'].append('RETURN_OVERFLOW');return
    if type(arg)is not dict:OBS['errors'].append('NO_TYPED_RETURN');return
    pixel=arg.get('pixel_count');steps=arg.get('physics_steps_by_oracle')
    if type(pixel)is not int or type(steps)is not int or pixel<0 or steps<0:OBS['errors'].append('RETURN_TYPES');return
    OBS['returns'].append({'pixel_count':pixel,'physics_steps_by_oracle':steps})
  sys.setprofile(profile);installed=True
 except (AssertionError,KeyError,AttributeError)as error:
  OBS['errors'].append(str(error)if type(error)is AssertionError else'OBSERVER_BINDING_MISSING')
 try:yield
 finally:
  if installed:sys.setprofile(previous)
  OBS['original_profile_restored']=sys.getprofile()is previous
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item,call):
 outcome=yield
 if item.nodeid==TARGET:
  report=outcome.get_result();OBS['phases'].append({'when':report.when,'outcome':report.outcome,'wasxfail':bool(getattr(report,'wasxfail',False))})
def pytest_sessionfinish(session,exitstatus):
 if os.environ.get('ROOT_CAMERA_OBSERVER_RESULT')is None:return
 output=Path(os.environ['ROOT_CAMERA_OBSERVER_RESULT']);assert output.is_absolute()and output.parent.is_dir()and not output.parent.is_symlink()and not output.exists(),'EXCLUSIVE_ROOT_OBSERVER_OUTPUT'
 passed=OBS['target_calls']==OBS['verify_calls']==len(OBS['returns'])==1 and not OBS['errors']and OBS['original_profile_restored']and [r['when']for r in OBS['phases']]==['setup','call','teardown']and all(r['outcome']=='passed'and not r['wasxfail']for r in OBS['phases'])and OBS['returns'][0]['pixel_count']>4 and OBS['returns'][0]['physics_steps_by_oracle']==0
 result={'status':'ROOT_OBSERVED_ORIGINAL_CAMERA_RETURN_AND_TEST_PASS'if passed else'ROOT_ORIGINAL_CAMERA_OBSERVATION_INCOMPLETE_OR_TEST_FAILED','target':TARGET,'observation':OBS,'pytest_session_exitstatus':int(exitstatus),'pins':PINS,'scope':'Only original verify_camera return in original test; no extra invocation, arrays copied, model/GL/SDK/step or product mutation. Target result is not whole-suite PASS.'}
 b=(json.dumps(result,indent=2)+'\n').encode();assert len(b)<=8192;fd=os.open(output,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400)
 try:os.write(fd,b);os.fsync(fd)
 finally:os.close(fd)
 print('ROOT_CAMERA_RETURN_OBSERVER '+json.dumps({'status':result['status'],'result_sha256':H(b),'observation':OBS,'pytest_session_exitstatus':int(exitstatus)},separators=(',',':')))
