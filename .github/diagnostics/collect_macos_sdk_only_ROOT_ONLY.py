"""Root remote readonly source/SDK observer. Never imports MuJoCo/OpenGL or invokes GL."""
from pathlib import Path
import argparse,hashlib,importlib.metadata,json,os,platform,re,stat,subprocess,sys
NAMES=('CGLChoosePixelFormat','CGLCreateContext','CGLGetCurrentContext','CGLSetCurrentContext','CGLLockContext','CGLUnlockContext','CGLReleaseContext','CGLReleasePixelFormat')
PINS={'mujoco/rendering/classic/renderer.py':'c193df6a8b8cc1659819abd0ded5c17dab0e3e64edb7af6a1e8d249bb4a4a548','mujoco/cgl/__init__.py':'075193793fb0743385eb37d0ca46ac3949a6eb1db6ad5bf41091bc588d96d41b','mujoco/cgl/cgl.py':'b836c7148846cd0d6ea1f461c3ad60eade74e2687b10d9c68a7de06db50467c0'}
H=lambda b:hashlib.sha256(b).hexdigest()
def need(v,c):
 if not v:raise AssertionError(c)
def read_regular(p,cap):
 p=Path(p);need(p.is_absolute()and not p.is_symlink(),'PRIMARY_LINK');fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
 try:
  z=os.fstat(fd);need(stat.S_ISREG(z.st_mode)and 0<z.st_size<=cap,'PRIMARY_REGULAR_SIZE');b=os.read(fd,cap+1);q=os.fstat(fd);need(len(b)==z.st_size and (z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns,z.st_ctime_ns)==(q.st_dev,q.st_ino,q.st_size,q.st_mtime_ns,q.st_ctime_ns),'PRIMARY_CHANGED');return b
 finally:os.close(fd)
def write_exclusive(p,b):
 need(len(b)<=4194304,'OUTPUT_BOUND');fd=os.open(p,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o400)
 try:
  with os.fdopen(os.dup(fd),'wb')as f:f.write(b);f.flush();os.fsync(f.fileno())
 finally:os.close(fd)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();need(sys.platform=='darwin','DARWIN_ONLY');need(a.output.is_absolute()and not a.output.exists(),'EXCLUSIVE_OUTPUT');a.output.mkdir(mode=0o700)
 sdk=Path(subprocess.check_output(['xcrun','--sdk','macosx','--show-sdk-path'],text=True,timeout=10).strip());need(sdk.is_absolute()and sdk.is_dir(),'SDK_ROOT')
 sdk_version=subprocess.check_output(['xcrun','--sdk','macosx','--show-sdk-version'],text=True,timeout=10).strip();root=(sdk/'System/Library/Frameworks/OpenGL.framework/Headers').resolve();need(root.is_dir(),'OPENGL_HEADERS_PRESENT')
 # SDK Versions/Current symlinks are metadata resolver paths. Only resolved regular
 # files under the exact selected SDK are read; original and resolved path recorded.
 need(root.is_relative_to(sdk.resolve()),'SDK_RESOLVED_ROOT');todo=[root/'OpenGL.h'];seen={};edges=[];missing=[];total=0
 while todo:
  p=todo.pop(0).resolve();need(p.is_relative_to(sdk.resolve()),'HEADER_ESCAPES_SDK');name=p.relative_to(sdk.resolve()).as_posix()
  if name in seen:continue
  need(len(seen)<128,'HEADER_CLOSURE_COUNT');b=read_regular(p,1048576);total+=len(b);need(total<=16777216,'HEADER_CLOSURE_BYTES');text=b.decode('utf-8');seen[name]={'bytes':len(b),'sha256':H(b),'source':str(p),'text':text}
  dest=a.output/'headers'/name;dest.parent.mkdir(parents=True,exist_ok=True,mode=0o700);write_exclusive(dest,b)
  for match in re.finditer(r'^\s*#\s*(?:include|import)\s*([<"])([^>"\n]+)[>"]',text,re.M):
   kind,child=match.groups();need(not child.startswith('/')and '..'not in Path(child).parts,'RELATIVE_HEADER_INCLUDE')
   candidates=([p.parent/child]if kind=='"'else[])+[sdk/'usr/include'/child]
   if '/'in child:
    framework,rest=child.split('/',1);candidates.append(sdk/'System/Library/Frameworks'/(framework+'.framework')/'Headers'/rest)
   else:candidates.append(root/child)
   found=next((q.resolve()for q in candidates if q.is_file()),None)
   edge={'from':name,'include':child,'line':text.count('\n',0,match.start())+1}
   if found is None:missing.append(edge);edges.append(edge|{'resolved':'MISSING_IN_SELECTED_SDK'});continue
   need(found.is_relative_to(sdk.resolve()),'INCLUDE_RESOLVES_OUTSIDE_SDK');edges.append(edge|{'resolved':found.relative_to(sdk.resolve()).as_posix()});todo.append(found)
 # Capture full original C declarations, not a guessed ctypes return/argument ABI.
 prototypes={};prototype_errors=[]
 for name in NAMES:
  matches=[]
  for filename,row in seen.items():
   text=re.sub(r'/\*.*?\*/',lambda m:'\n'*m.group(0).count('\n'),row['text'],flags=re.S)
   for m in re.finditer(r'(?m)^[^#;{}\n]*\b'+re.escape(name)+r'\s*\([^;{}]*\)[^;{}]*;',text):
    declaration=m.group(0).strip();matches.append({'header':filename,'line':text.count('\n',0,m.start())+1,'declaration':declaration,'normalized':' '.join(declaration.split())})
  prototype_errors.extend([name]if len(matches)!=1 else[]);prototypes[name]=matches[0]if len(matches)==1 else{'status':'MISSING_OR_AMBIGUOUS','matches':matches}
 headers=[{k:v for k,v in row.items()if k!='text'}|{'name':name}for name,row in sorted(seen.items())];manifest={'headers':headers,'prototypes':prototypes,'include_edges':edges,'missing_includes':missing,'closure_kind':'all literal lexical #include/#import under selected SDK; inactive conditional branches included, no preprocessor/CGL execution','header_limits':{'files':128,'bytes':16777216,'each_bytes':1048576}};mb=(json.dumps(manifest,sort_keys=True,separators=(',',':'))+'\n').encode();write_exclusive(a.output/'SDK_CGL_MANIFEST.json',mb)
 dist=importlib.metadata.distribution('mujoco');version_matches=dist.version=='3.13.0';installed=[];primary_mismatches=[]
 for rel,expected in PINS.items():
  p=Path(dist.locate_file(rel)).resolve();b=read_regular(p,1048576);primary_mismatches.extend([rel]if H(b)!=expected else[]);installed.append({'relative_path':rel,'path':str(p),'bytes':len(b),'sha256':H(b),'expected_linux_sha256':expected,'matches_expected':H(b)==expected});write_exclusive(a.output/('installed_'+rel.replace('/','__')),b)
 result={'status':'MAC_SDK_ONLY_PRIMARY_COLLECTED_NO_CAMERA'if not missing and version_matches and not primary_mismatches and not prototype_errors else'MAC_SDK_ONLY_PRIMARY_INCOMPLETE_OR_MISMATCH_NO_CAMERA','system':platform.platform(),'machine':platform.machine(),'sdk_original':str(sdk),'sdk_resolved':str(sdk.resolve()),'sdk_version':sdk_version,'OpenGL_headers_resolved':str(root),'sdk_manifest_sha256':H(mb),'prototypes':prototypes,'prototype_errors':prototype_errors,'installed_primary':installed,'mujoco_version':dist.version,'version_matches_3_13_0':version_matches,'installed_primary_mismatches':primary_mismatches,'CGL_GL_model_STEP_NN_render_queries':0,'semantic_ABI_comparison':'ROOT_REVIEW_REQUIRED_NOT_INFERRED_FROM_CAPTURE','include_closure_complete':not missing,'missing_includes':missing,'original_PR693_probe':'retained; not invoked by collector'}
 write_exclusive(a.output/'receipt.json',(json.dumps(result,indent=2)+'\n').encode());print(json.dumps({'status':result['status'],'sdk_manifest_sha256':H(mb),'prototype_count':len(prototypes),'actual_renderer_sha256':next(x['sha256']for x in installed if x['relative_path']=='mujoco/rendering/classic/renderer.py'),'missing_include_count':len(missing),'primary_mismatches':primary_mismatches}))
 need(not prototype_errors,'EIGHT_PROTOTYPES_MISSING_OR_AMBIGUOUS_RECORDED')
 need(not missing,'SDK_LITERAL_INCLUDE_CLOSURE_MISSING')
 need(version_matches and not primary_mismatches,'INSTALLED_VERSION_OR_SOURCE_MISMATCH_RECORDED')
if __name__=='__main__':
 try:main()
 except AssertionError as e:print(json.dumps({'status':'SDK_ONLY_COLLECTION_FAILED_NO_CAMERA','code':str(e)}),file=sys.stderr);raise SystemExit(2)
