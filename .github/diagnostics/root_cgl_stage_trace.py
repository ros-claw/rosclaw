"""External crash-stage tracing of unchanged pinned Native source; no CGL calls."""
import faulthandler,hashlib,json,pathlib,runpy,sys,threading
p=pathlib.Path('tools/diagnostics/macos_cgl_probe_native.py');assert hashlib.sha256(p.read_bytes()).hexdigest()=='6fd4319ff26e7c51cc289661595df877aa2ba47f34f79c5db8124b4b4010caf5';faulthandler.enable(all_threads=True)
count=0
allowed={'__init__','choose','create','set_current','query','release_context','release_pixel_format'}
def trace(frame,event,arg):
 global count
 if pathlib.Path(frame.f_code.co_filename).name==p.name and frame.f_code.co_name in allowed and event in ['call','return']:
  assert count<128,'TRACE_EVENT_BOUND';count+=1;record={'event':event,'function':frame.f_code.co_qualname,'line':frame.f_lineno}
  if frame.f_code.co_name=='choose'and'attrs'in frame.f_locals:record['attrs']=frame.f_locals['attrs']
  if frame.f_code.co_name=='query':record['GL_field']=frame.f_locals.get('name')
  h=frame.f_locals.get('handles')
  if type(h)is dict:record['handles']={'pix_nonnull':bool(h.get('pix')),'ctx_nonnull':bool(h.get('ctx')),'npix':h.get('npix')}
  if event=='return':record['return']=arg if type(arg)in(int,bool,str,type(None))else type(arg).__name__
  print('ROOT_NATIVE_CGL_STAGE '+json.dumps(record,separators=(',',':')),file=sys.stderr,flush=True)
 return trace
threading.settrace(trace);sys.settrace(trace);runpy.run_path(str(p),run_name='__main__')
