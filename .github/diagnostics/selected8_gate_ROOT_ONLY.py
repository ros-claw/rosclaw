"""Independent selected8 source/typedef gate, not complete SDK closure or compiled ABI."""
from pathlib import Path
import argparse,ast,ctypes,hashlib,importlib.metadata,json,os,platform,stat,subprocess,sys
H=lambda b:hashlib.sha256(b).hexdigest()
def read(p,cap=1048576):
 p=Path(p);assert p.is_absolute()and not p.is_symlink(),'SOURCE_LINK';fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
 try:
  z=os.fstat(fd);assert stat.S_ISREG(z.st_mode)and 0<z.st_size<=cap,'SOURCE_SIZE';b=os.read(fd,cap+1);q=os.fstat(fd);assert len(b)==z.st_size and (z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns,z.st_ctime_ns)==(q.st_dev,q.st_ino,q.st_size,q.st_mtime_ns,q.st_ctime_ns),'SOURCE_CHANGED';return b
 finally:os.close(fd)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--manifest-sha256',required=True);ap.add_argument('--root-authority',type=Path,required=True);ap.add_argument('--root-authority-sha256',required=True);ap.add_argument('--factory',type=Path,required=True);ap.add_argument('--result',type=Path,required=True);a=ap.parse_args()
 mb=read(a.manifest,4194304);assert H(mb)==a.manifest_sha256;expected=json.loads(mb);ab=read(a.root_authority);assert H(ab)==a.root_authority_sha256;authority=json.loads(ab)
 assert authority['status']=='ROOT_SELECTED8_DECLARATION_SCOPE_ACCEPTED_NOT_FULL_CLOSURE'and authority['Root_manual_selected8_scope_approval']is True and authority['original_sdk_only_FAILURE_preserved']is True and authority['compiled_enum_ABI_not_claimed']is True,'ROOT_SCOPE_AUTHORITY_REQUIRED'
 assert authority['selected_manifest_sha256']==H(mb)and authority['factory_path']==str(a.factory)and len(authority['factory_sha256'])==64,'EXACT_SELECTED_SCOPE_FACTORY_BINDING'
 assert sys.platform=='darwin'and platform.machine()==expected['machine']=='arm64','EXACT_TARGET'
 sdk=Path(subprocess.check_output(['xcrun','--sdk','macosx','--show-sdk-path'],text=True,timeout=10).strip()).resolve();version=subprocess.check_output(['xcrun','--sdk','macosx','--show-sdk-version'],text=True,timeout=10).strip();assert version==expected['sdk_version'],'SDK_VERSION'
 observed=[]
 for row in expected['selected_headers']:
  p=(sdk/row['relative_sdk_path']).resolve();assert p.is_relative_to(sdk),'SDK_BOUNDARY';b=read(p);assert len(b)==row['bytes']and H(b)==row['sha256'],'SELECTED_HEADER_EXACT';observed.append({'relative_sdk_path':row['relative_sdk_path'],'bytes':len(b),'sha256':H(b)})
 dist=importlib.metadata.distribution('mujoco');assert dist.version=='3.13.0';primary=[]
 for row in expected['installed_primary']:
  p=Path(dist.locate_file(row['relative_path'])).resolve();b=read(p);assert H(b)==row['sha256']and len(b)==row['bytes'],'INSTALLED_PRIMARY_EXACT';primary.append({'relative_path':row['relative_path'],'sha256':H(b),'bytes':len(b)})
 b=read(a.factory);assert H(b)==authority['factory_sha256'],'CURRENT_ACCEPTED_FACTORY_SHA';t=ast.parse(b);c=next(n for n in t.body if isinstance(n,ast.ClassDef)and n.name=='_CGLAPI');f=next(n for n in c.body if isinstance(n,ast.FunctionDef)and n.name=='__init__');d=next(n.value for n in f.body if isinstance(n,ast.Assign)and any(isinstance(x,ast.Name)and x.id=='signatures'for x in n.targets));assert H(ast.dump(d,include_attributes=False).encode())==expected['binding_source']['CGL_API_signature_dict_ast_sha256'],'SAME_EIGHT_DECLARED_CTYPES_SHAPES'
 # Host ctypes metadata only, not C sizeof(enum), compiler flags, or framework invocation.
 assert ctypes.sizeof(ctypes.c_int)==4 and ctypes.sizeof(ctypes.c_void_p)==8,'HOST_CTYPE_WIDTHS'
 result={'status':'ROOT_SELECTED8_DECLARATIONS_TYPES_PRIMARY_MATCH_NOT_COMPILED_ABI_NOT_CAMERA','selected_manifest_sha256':H(mb),'root_authority_sha256':H(ab),'headers':observed,'installed_primary':primary,'factory_sha256':H(b),'declared_ctypes_ast_sha256':H(ast.dump(d,include_attributes=False).encode()),'host_ctypes_int_bytes':4,'host_ctypes_pointer_bytes':8,'compiled_C_enum_ABI':'UNKNOWN_NOT_MEASURED','complete_SDK_literal_closure':False,'original_SDK_ONLY_FAILURE':'PRESERVED_NOT_RESCORED','actual_CGL_model_STEP_NN_render_calls':0,'next':'Root originalcamera job only if separately authorized; no finiteprobe repeated'}
 rb=(json.dumps(result,indent=2)+'\n').encode();assert len(rb)<=32768 and a.result.is_absolute()and not a.result.exists();fd=os.open(a.result,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400)
 try:os.write(fd,rb);os.fsync(fd)
 finally:os.close(fd)
 print(json.dumps({'status':result['status'],'result_sha256':H(rb),'headers':len(observed),'prototypes':len(expected['prototypes'])}))
if __name__=='__main__':main()
