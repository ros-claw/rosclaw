"""Read-only Docker evidence while the original module fixture is still alive."""
import hashlib
import json
import subprocess
from urllib.parse import urlparse

import pytest

_installed = False
_script = r'''
import hashlib,json,pathlib,subprocess
packages=['ros-humble-rosapi','ros-humble-rosapi-msgs','ros-humble-rosbridge-library','ros-humble-rosbridge-server','ros-humble-rclpy']
v=subprocess.run(['dpkg-query','-W','-f=${Package} ${Version}\n',*packages],capture_output=True,text=True,timeout=10)
print(json.dumps({'kind':'PACKAGES','returncode':v.returncode,'stdout':v.stdout,'stderr':v.stderr}))
q=subprocess.run(['dpkg-query','-L',*packages],capture_output=True,text=True,timeout=10)
selected=[]
for name in sorted(set(q.stdout.splitlines())):
 p=pathlib.Path(name);base=p.name.lower().replace('_','')
 wanted=p.name in ['proxy.py','rosapi_node','rosapi_node.py','call_service.py'] or '/rclpy/action/'in name and p.name=='__init__.py' or p.suffix=='.srv'and any(k in base for k in ['actiontype','rosversion','getactionservers','services','servicetype','topics','topictype','getparam'])
 if wanted and p.is_file()and not p.is_symlink():
  b=p.read_bytes()
  if len(b)<=131072 and len(selected)<24:
   selected.append(name);print(json.dumps({'kind':'INSTALLED_FILE','path':name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'text':b.decode('utf-8')}))
print(json.dumps({'kind':'FILE_SELECTION','paths':selected,'dpkg_list_returncode':q.returncode}))
'''


def emit(kind, nodeid, text):
    encoded = text.encode()
    total = (len(text) + 4095) // 4096
    for index in range(total):
        print('ROOT_ROS_CONTAINER_CHUNK ' + json.dumps({
            'kind': kind, 'nodeid': nodeid, 'index': index, 'chunks': total,
            'bytes': len(encoded), 'sha256': hashlib.sha256(encoded).hexdigest(),
            'text': text[index * 4096:(index + 1) * 4096],
        }, ensure_ascii=True, separators=(',', ':')), flush=True)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    global _installed
    result = yield
    report = result.get_result()
    if report.when != 'call':
        return
    endpoint = item.funcargs.get('deployed_ros_stack')
    if not isinstance(endpoint, str):
        return
    port = urlparse(endpoint).port
    if not isinstance(port, int):
        return
    container = 'rosclaw-ros2-humble-bridge-smoke-' + str(port)
    commands = []
    if not _installed:
        commands.append(('INSTALLED_METADATA', ['docker', 'exec', container, 'python3', '-c', _script]))
        _installed = True
    if item.name in ['test_discovery_finds_turtlesim_capabilities', 'test_compile_manifest_for_turtlesim']:
        commands += [
            ('CONTAINER_LOGS', ['docker', 'logs', '--tail', '200', container]),
            ('CONTAINER_STATE', ['docker', 'inspect', '--format', '{{json .State}}', container]),
        ]
    for kind, argv in commands:
        try:
            completed = subprocess.run(argv, capture_output=True, text=True, timeout=30)
            payload = json.dumps({'returncode': completed.returncode,
                                  'stdout': completed.stdout[:1048576],
                                  'stderr': completed.stderr[:131072]}, ensure_ascii=True)
        except Exception as error:
            payload = json.dumps({'observer_error': type(error).__name__})
        emit(kind, item.nodeid, payload)
