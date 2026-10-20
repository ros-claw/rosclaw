"""Root-owned finite fresh-process acceptance scenarios; never launch locally.

Required external fixture ``root_owned_process`` accepts (stage, source) and
returns the completed process result. Root must review each exact source/argv,
launch once with a timeout (no retries), real installed dependencies and genuine
complete candidate product source, and reject all SDK/device/network/process
side effects independently. This fixture is deliberately NOT supplied here:
source scope does not authorize launching a worker. No fake package fixture.

Each child receives ROSCLAW_CANDIDATE_SRC: the reviewed complete source root
containing rosclaw/, not this sparse author tree. Root must validate fixture
provenance and candidate bytes, disable bytecode writes, and constrain filesystem
writes. Stage B needs separately reviewed full-dependency limits; SciPy is a
legitimate core dependency, not a product defect. Import availability grants no
service execution permission.

Additional mandatory acceptance: run the ORIGINAL tests/perception/handcam/
test_handcam.py unchanged: seven tests, all 21 phase reports passed, no skip,
xfail or denial. Those numerical tests are not replaced by these import tests.
The two original failed purposes remain failures until separately qualified.
"""

COMMON = r"""
import importlib
import os
from pathlib import Path
import sys

assert not any(n == 'rosclaw' or n.startswith('rosclaw.') for n in sys.modules)
source_root = Path(os.environ['ROSCLAW_CANDIDATE_SRC']).resolve(strict=True)

def origin(module):
    name = module.__name__
    path = source_root.joinpath(*name.split('.'))
    expected = path / '__init__.py' if hasattr(module, '__path__') else path.with_suffix('.py')
    expected = expected.resolve(strict=True)
    assert Path(module.__file__).resolve(strict=True) == expected, name
    assert Path(module.__spec__.origin).resolve(strict=True) == expected, name
    if hasattr(module, '__path__') and name != 'rosclaw':
        assert [Path(p).resolve(strict=True) for p in module.__path__] == [expected.parent]

def all_origins():
    for name, module in tuple(sys.modules.items()):
        if name == 'rosclaw' or name.startswith('rosclaw.'):
            origin(module)

names = ['PracticeConfig', 'PracticeCoordinator', 'PracticeRecorder',
         'PracticeSession', 'PracticeSummary', 'UnifiedTimeline',
         'TimelineChannel', 'TimelineEntry', 'EpisodeRecorder']
defining = ['config', 'coordinator', 'recorder', 'config', 'config',
            'timeline', 'timeline', 'timeline', 'episode_recorder']
"""

COLD = r"""
forbidden = ('sklearn', 'scipy', 'rosclaw.practice.coordinator',
             'rosclaw.memory', 'pyrealsense2', 'rclpy', 'rospy',
             'cyclonedds', 'serial', 'can', 'mujoco', 'torch',
             'transformers', 'pyseekdb')
attempts = []
def audit(event, args):
    if event == 'import':
        name = args[0]
        if any(name == p or name.startswith(p + '.') for p in forbidden):
            attempts.append(name)
            raise PermissionError('forbidden import: ' + name)
    if (event.startswith('socket.') or event.startswith('subprocess.')
            or event in ('os.system', 'os.fork', 'os.posix_spawn', 'pty.spawn')):
        attempts.append(event)
        raise PermissionError(event)
    if event == 'open' and isinstance(args[0], (str, bytes)):
        path = os.fsdecode(args[0])
        if path == '/dev' or path.startswith('/dev/') or path.startswith('/sys/'):
            attempts.append(path)
            raise PermissionError(path)
class ImportObserver:
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == p or fullname.startswith(p + '.') for p in forbidden):
            attempts.append(fullname)
            raise PermissionError('forbidden import: ' + fullname)
        return None
sys.meta_path.insert(0, ImportObserver())
sys.addaudithook(audit)
try:
    observation = importlib.import_module('rosclaw.practice.physical_observation')
    handcam = importlib.import_module('rosclaw.perception.handcam')
    import rosclaw.practice as practice
    import numpy
    assert numpy.__file__ and numpy.ndarray.__module__ == 'numpy'
    assert practice.physical_observation is observation
    assert handcam.CameraIntrinsics.__module__ == 'rosclaw.perception.handcam.calibration'
    assert practice.__all__ == names
    assert not any(n in vars(practice) for n in names)
    assert set(names) <= set(dir(practice))
    for unknown in ('NotAPracticeExport', '_missing_export', '__not_an_export__'):
        try:
            getattr(practice, unknown)
        except AttributeError:
            pass
        else:
            raise AssertionError('unknown lookup succeeded: ' + unknown)
    # Genuine canonical_hash API, known digest plus ordering and content controls.
    h = observation.canonical_hash
    assert h({}, prefix='test') == 'test_44136fa355b3678a'
    assert h({'b': 2, 'a': 1}, prefix='x') == h({'a': 1, 'b': 2}, prefix='x')
    assert h({'a': 1}, prefix='x') != h({'a': 2}, prefix='x')
    assert h({}, prefix='other') == 'other_44136fa355b3678a'
    all_origins()
    assert not any(any(n == p or n.startswith(p + '.') for p in forbidden)
                   for n in sys.modules)
finally:
    # Optional dependency handlers must not hide a rejected import attempt.
    assert not attempts, repr(attempts)
"""

PUBLIC = r"""
import rosclaw.practice as practice
assert practice.__all__ == names
from rosclaw.practice import (
    PracticeConfig, PracticeCoordinator, PracticeRecorder, PracticeSession,
    PracticeSummary, UnifiedTimeline, TimelineChannel, TimelineEntry, EpisodeRecorder,
)
imported = [PracticeConfig, PracticeCoordinator, PracticeRecorder, PracticeSession,
            PracticeSummary, UnifiedTimeline, TimelineChannel, TimelineEntry, EpisodeRecorder]
for name, module_name, value in zip(names, defining, imported, strict=True):
    module = importlib.import_module('rosclaw.practice.' + module_name)
    assert value is getattr(module, name), name
    assert value.__module__ == module.__name__, name
    assert getattr(practice, name) is value
    assert getattr(practice, name) is value
    assert vars(practice)[name] is value
    assert getattr(practice, module_name) is module
assert practice.__all__ == names
for unknown in ('NotAPracticeExport', '_missing_export', '__not_an_export__'):
    try:
        getattr(practice, unknown)
    except AttributeError:
        pass
    else:
        raise AssertionError(unknown)
assert not hasattr(practice, 'NotAPracticeExport')
all_origins()
"""

STAR = r"""
import rosclaw.practice as practice
from rosclaw.practice import *
assert practice.__all__ == names
for name, module_name in zip(names, defining, strict=True):
    module = importlib.import_module('rosclaw.practice.' + module_name)
    assert globals()[name] is getattr(module, name)
    assert getattr(practice, name) is globals()[name]
all_origins()
"""

NEGATIVE_EAGER = r"""
# Genuine explicit heavy requests exercise the SAME cold observer, not a mock
# resolver. A caught dependency denial must remain observable. Run separately
# from A; a denial here is an expected negative control, never an A pass.
import rosclaw.practice as practice
attempts = []
class ImportObserver:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'rosclaw.practice.coordinator':
            attempts.append(fullname)
            raise PermissionError(fullname)
        return None
sys.meta_path.insert(0, ImportObserver())
try:
    practice.PracticeCoordinator
except PermissionError:
    pass
else:
    raise AssertionError('heavy request did not expose eager dependency')
assert attempts == ['rosclaw.practice.coordinator']
assert 'PracticeCoordinator' not in vars(practice)
all_origins()
"""

NEGATIVE_API = r"""
import rosclaw.practice as practice
assert practice.__all__ == names
# Wrong expectations against actual objects: no mutation of package or resolver.
for name, module_name in zip(names, defining, strict=True):
    real = getattr(practice, name)
    wrong = getattr(importlib.import_module('rosclaw.practice.config'), 'PracticeSummary')
    if name != 'PracticeSummary':
        assert real is not wrong, name
    assert real.__module__ == 'rosclaw.practice.' + module_name
assert practice.__all__ != names[:-1]
try:
    practice.NotAPracticeExport
except AttributeError:
    pass
else:
    raise AssertionError('unknown lookup accepted')
# A foreign expected source root MUST fail against actual module origins.
all_origins()
source_root = source_root / '__foreign_product_negative_control__'
try:
    origin(practice)
except (AssertionError, FileNotFoundError):
    pass
else:
    raise AssertionError('foreign origin accepted')
"""


def _check(root_owned_process, stage, scenario):
    result = root_owned_process(stage, COMMON + scenario)
    assert result.returncode == 0, (result.stdout, result.stderr)


def test_cold_product_imports(root_owned_process):
    _check(root_owned_process, "A_cold", COLD)


def test_all_public_from_imports(root_owned_process):
    _check(root_owned_process, "B_full_dependencies", PUBLIC)


def test_public_star_import(root_owned_process):
    _check(root_owned_process, "B_star_full_dependencies", STAR)


def test_explicit_heavy_dependency_negative_control(root_owned_process):
    _check(root_owned_process, "C_dependency_denial_control", NEGATIVE_EAGER)


def test_actual_api_and_foreign_origin_negative_controls(root_owned_process):
    _check(root_owned_process, "C_full_dependencies_controls", NEGATIVE_API)


# Repository fixture contract controls. These do not substitute for the five
# genuine product scenarios above; future execution needs its own admission.
# Keep these imports late to preserve the frozen 9573-byte scenario prefix.
import inspect  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402


def _fixture_transport_module(runner):
    module = inspect.getmodule(type(runner))
    assert module is not None
    return module


def test_fixture_rejects_foreign_pairs_before_transport(root_owned_process, monkeypatch):
    module = _fixture_transport_module(root_owned_process)
    calls = []

    def forbidden_transport(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("rejection reached transport or source discovery")

    monkeypatch.setattr(module, "_run_bounded", forbidden_transport)
    monkeypatch.setattr(module, "_source_tree", forbidden_transport)
    with pytest.raises(ValueError, match="unknown stage"):
        root_owned_process("foreign_stage", COMMON + COLD)
    with pytest.raises(ValueError, match="source does not match stage"):
        root_owned_process("A_cold", COMMON + PUBLIC)
    with pytest.raises(ValueError, match="source does not match stage"):
        root_owned_process("A_cold", COMMON + COLD + "\n")
    with pytest.raises(ValueError, match="source does not match stage"):
        root_owned_process("B_full_dependencies", None)
    assert calls == []


def test_fixture_transport_preserves_failure_and_removes_directory(root_owned_process):
    module = _fixture_transport_module(root_owned_process)
    result = module._run_bounded(
        "import sys\nprint('finite stdout', flush=True)\n"
        "print('finite stderr', file=sys.stderr, flush=True)\nsys.exit(7)\n",
        timeout=10.0,
    )
    assert result.returncode == 7
    assert result.stdout == "finite stdout\n"
    assert result.stderr == "finite stderr\n"
    assert not Path(result.args[-1]).parent.exists()


def test_fixture_transport_timeout_is_reaped_and_cleaned(root_owned_process):
    module = _fixture_transport_module(root_owned_process)
    # Finite even without the timeout; no sleeps, services or child descendants.
    with pytest.raises(module.ProcessBoundaryError) as caught:
        module._run_bounded(
            "import time\nend = time.monotonic() + 10\nwhile time.monotonic() < end:\n    pass\n",
            timeout=0.1,
        )
    error = caught.value
    assert error.reason == "timeout"
    assert error.result.returncode < 0
    assert isinstance(error.result.stdout, str)
    assert isinstance(error.result.stderr, str)
    assert not Path(error.directory).exists()
    assert not Path(error.result.args[-1]).exists()


def test_fixture_transport_output_limit_is_failure_not_success(root_owned_process):
    module = _fixture_transport_module(root_owned_process)
    with pytest.raises(module.ProcessBoundaryError) as caught:
        module._run_bounded("print('x' * 65536, flush=True)\n", timeout=10.0, output_limit=1024)
    error = caught.value
    assert error.reason == "output_limit"
    # The child may already have exited normally; the boundary still fails.
    assert isinstance(error.result.returncode, int)
    assert len(error.result.stdout.encode()) + len(error.result.stderr.encode()) <= 1024
    assert not Path(error.directory).exists()
