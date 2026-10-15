"""External observer only: no extra ROS queries, retries, fixture or assertion edits."""
import dataclasses,json,sys,time
import pytest
_starts={};_rows=[];_old=None
def small(v,depth=0):
 if depth>3:return '<depth_bound>'
 if v is None or type(v)in (bool,int,float):return v
 if isinstance(v,str):return v[:256]
 if isinstance(v,(list,tuple)):return {'count':len(v),'prefix':[small(x,depth+1)for x in v[:32]]}
 if isinstance(v,dict):return {str(k)[:128]:small(x,depth+1)for k,x in list(v.items())[:24]}
 if hasattr(v,'value'):return small(v.value,depth+1)
 return type(v).__name__
def profile(v):return {n:small(getattr(v,n,None))for n in ['version','distro','service_prefix','type_prefix']}
def trace(frame,event,arg):
 if frame.f_code.co_name!='_call_rosapi'or not frame.f_code.co_filename.endswith('/discovery/graph.py'):return
 if event=='call':_starts[id(frame)]=time.monotonic()
 if event!='return'or len(_rows)>=400:return
 name=frame.f_locals.get('name',frame.f_locals.get('service',frame.f_locals.get('service_name')))
 data=getattr(arg,'data',None);ok=getattr(arg,'ok',None);error=getattr(arg,'error',None)
 row={'kind':'ROSAPI_ORIGINAL_RETURN','method':small(name),'elapsed_s':time.monotonic()-_starts.pop(id(frame),time.monotonic()),'ok':small(ok),'error':small(error),'profile':profile(getattr(frame.f_locals.get('self'),'_profile',None))}
 if name in ['topics','topic_type','services','service_type','action_servers','action_type']:row['data']=small(data)
 _rows.append(row)
def pytest_sessionstart(session):
 global _old
 _old=sys.getprofile();sys.setprofile(trace)
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item,call):
 result=yield
 report=result.get_result()
 if report.when!='call':return
 record={'kind':'ROOT_ROS_ORIGINAL_CASE_OBSERVATION','nodeid':item.nodeid,'outcome':report.outcome,'original_rosapi_returns':_rows.copy()};_rows.clear()
 if call.excinfo:
  for entry in call.excinfo.traceback:
   if entry.frame.code.path.basename!='test_ros2_container_smoke.py':continue
   loc=entry.frame.f_locals;snapshot=loc.get('snapshot');manifest=loc.get('manifest');disc=loc.get('discovery')
   if snapshot is not None:
    record['snapshot']={n:{'count':len(getattr(snapshot,n,[])),'prefix':[small(dataclasses.asdict(v)if dataclasses.is_dataclass(v)else v)for v in getattr(snapshot,n,[])[:64]]}for n in ['topics','services','actions','nodes','params']}
    record['snapshot_ros_version']=small(getattr(snapshot,'ros_version',None))
   if disc is not None:record['discovery_profile']=profile(getattr(disc,'_profile',None))
   if manifest is not None:record['manifest_capabilities_count']=len(manifest.capabilities)
   if 'cap_ids'in loc:record['cap_ids']=sorted(loc['cap_ids'])
 print('ROOT_ROS_DIAG '+json.dumps(record,ensure_ascii=True,separators=(',',':')),flush=True)
def pytest_sessionfinish(session,exitstatus):sys.setprofile(_old)
