"""External one-process observer for the already authored Native diagnostic."""
import hashlib,json,os,pathlib,signal,subprocess,sys,time
p=pathlib.Path('tools/diagnostics/macos_cgl_probe_native.py');assert hashlib.sha256(p.read_bytes()).hexdigest()=='09644dedd733e183fede7cb7adb94172ea2f3a46124f4fe74125b55ddd9ae9ed'
start=time.monotonic();child=subprocess.Popen([sys.executable,'-B',str(p)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYOPENGL_PLATFORM':'darwin','MUJOCO_GL':'cgl'});timed=False;signals=[]
try:out,err=child.communicate(timeout=90)
except subprocess.TimeoutExpired:
 timed=True
 if child.poll()is None:os.killpg(child.pid,signal.SIGTERM);signals.append('SIGTERM_OWN_CREATED_GROUP')
 try:out,err=child.communicate(timeout=3)
 except subprocess.TimeoutExpired:
  if child.poll()is None:os.killpg(child.pid,signal.SIGKILL);signals.append('SIGKILL_OWN_CREATED_GROUP')
  out,err=child.communicate(timeout=3)
assert len(out.encode())<=1048576 and len(err.encode())<=131072
try:native=json.loads(out)
except (ValueError,TypeError):native=None
record={'status':'ROOT_OBSERVED_ONE_NATIVE_CGL_DIAGNOSTIC_PROCESS_CLOSED'if child.returncode is not None and not timed else'ROOT_CGL_DIAGNOSTIC_TIMEOUT','elapsed_s':time.monotonic()-start,'timed_out':timed,'exit_code':child.returncode,'signals':signals,'stdout_sha256':hashlib.sha256(out.encode()).hexdigest(),'stderr':err,'Native_report':native,'scope':'One actual hosted macOS CGL negotiation; no model/STEP/NN/render or original camera validation; process group ownership only, not whole OS namespace'}
print('ROOT_CGL_DIAGNOSTIC_RESULT '+json.dumps(record,separators=(',',':')),flush=True)
raise SystemExit(child.returncode if not timed and child.returncode is not None else 124)
