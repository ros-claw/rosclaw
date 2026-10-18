"""Independent storage import regressions; source only until execution review.

No use or extension of the frozen five-stage fixture. Every case launches its
own ordinary product interpreter, never a replacement namespace. Root must
review argv/environment/dependency origins and observe effects independently
before running these tests. The Python observer below is not an OS sandbox.

BLOCKED for full-dependency stages until a legitimate complete environment
(including declared core SciPy) is prepared/authorized. Only this interpreter's
sysconfig install directories are used; no base-venv path injection or skips.
Explicit fallback/from and storage/star are separate compatibility/effects
stages: genuine urllib3 can attempt its IPv6 socket probe. A rejected attempt
fails the effects stage even when urllib3 catches PermissionError; it is not a
pure import pass. Only completed compatibility followed by the original strict
assertion rejecting exclusively socket.__new__ denials is an accepted negative
control. Missing dependencies or incomplete checks remain failures. Child
returncode and both real streams are retained, never rewritten as success.
No instances, services, HTTP calls or socket allowances here.
Prior failures and original tests remain unchanged; no physical/SDK credit.
"""

from pathlib import Path

import pytest


def _common_source():
    return r"""
import importlib
import os
from pathlib import Path
import sys

assert not any(n == 'rosclaw' or n.startswith('rosclaw.') for n in sys.modules)
assert not any(n == 'requests' or n.startswith('requests.') or
               n == 'urllib3' or n.startswith('urllib3.') for n in sys.modules)
root = Path(os.environ['ROSCLAW_CANDIDATE_SRC']).resolve(strict=True)
assert sys.path[0] == str(root)
assert os.environ['JOBLIB_MULTIPROCESSING'] == '0'
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
            'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    assert os.environ[key] == '1'

storage_names = ['PracticeCatalog', 'FallbackSync', 'PracticeLayout']
practice_names = ['PracticeConfig', 'PracticeCoordinator', 'PracticeRecorder',
                  'PracticeSession', 'PracticeSummary', 'UnifiedTimeline',
                  'TimelineChannel', 'TimelineEntry', 'EpisodeRecorder']
practice_modules = ['config', 'coordinator', 'recorder', 'config', 'config',
                    'timeline', 'timeline', 'timeline', 'episode_recorder']
http_roots = ('rosclaw.practice.storage.fallback_sync', 'requests', 'urllib3',
              'chardet', 'idna', 'certifi')
forbidden = ('pyrealsense2', 'rclpy', 'rospy', 'cyclonedds', 'serial', 'can',
             'mujoco', 'torch', 'transformers', 'pyseekdb')
denials = []
import_attempts = []

class ImportObserver:
    def find_spec(self, fullname, path=None, target=None):
        import_attempts.append(fullname)
        if any(fullname == p or fullname.startswith(p + '.') for p in forbidden):
            denials.append(('import', fullname))
            raise PermissionError('forbidden import: ' + fullname)
        return None

def audit(event, args):
    if (event.startswith('socket.') or event.startswith('subprocess.') or
            event in ('os.system', 'os.fork', 'os.forkpty', 'os.posix_spawn',
                      'pty.spawn')):
        denials.append((event, repr(args)))
        raise PermissionError(event)
    if event == 'open' and isinstance(args[0], (str, bytes)):
        path = os.fsdecode(args[0])
        if path == '/dev' or path.startswith('/dev/') or path.startswith('/sys/'):
            denials.append((event, path))
            raise PermissionError(path)

sys.meta_path.insert(0, ImportObserver())
sys.addaudithook(audit)

def origins():
    for name, module in tuple(sys.modules.items()):
        if name == 'rosclaw' or name.startswith('rosclaw.'):
            path = root.joinpath(*name.split('.'))
            expected = path / '__init__.py' if hasattr(module, '__path__') else path.with_suffix('.py')
            expected = expected.resolve(strict=True)
            assert Path(module.__file__).resolve(strict=True) == expected, name
            assert Path(module.__spec__.origin).resolve(strict=True) == expected, name
            if hasattr(module, '__path__') and name != 'rosclaw':
                assert [Path(p).resolve(strict=True) for p in module.__path__] == [expected.parent]

def unknowns(package):
    for name in ('NotAStorageExport', '_missing_export', '__not_an_export__'):
        before = tuple(import_attempts)
        try:
            getattr(package, name)
        except AttributeError as error:
            assert str(error) == f"module {package.__name__!r} has no attribute {name!r}"
        else:
            raise AssertionError(name)
        assert tuple(import_attempts) == before
        assert name not in vars(package)
        assert not hasattr(package, name)

def no_http():
    for name in tuple(sys.modules) + tuple(import_attempts):
        assert not any(name == p or name.startswith(p + '.') for p in http_roots), name
    # Generic encoding detection is not HTTP. Still require ordinary installed
    # package provenance for every loaded member, including native extensions.
    import sysconfig
    from importlib.machinery import all_suffixes
    # Before Python 3.14, -S can leave sys.prefix at the base install.
    # Locate pyvenv.cfg from the executable spelling, not its symlink target;
    # neither a module's directory nor an arbitrary sys.path entry proves this.
    executable = Path(os.path.abspath(sys.executable))
    assert executable.is_file(), executable
    configs = [p / 'pyvenv.cfg' for p in (executable.parent, executable.parent.parent)
               if (p / 'pyvenv.cfg').is_file()]
    assert len(configs) <= 1, configs
    if configs:
        config = configs[0]
        prefix = config.parent.resolve(strict=True)
        settings = {}
        for line in config.read_text(encoding='utf-8').splitlines():
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip().lower()
                assert key not in settings, key
                settings[key] = value.strip()
        base_executable = Path(sys._base_executable).resolve(strict=True)
        assert Path(settings['home']).resolve(strict=True) == base_executable.parent
        assert tuple(map(int, settings['version'].split('.')[:2])) == sys.version_info[:2]
        assert Path(sys.prefix).resolve(strict=True) in (
            prefix, Path(sys.base_prefix).resolve(strict=True))
        assert Path(sys.exec_prefix).resolve(strict=True) in (
            prefix, Path(sys.base_exec_prefix).resolve(strict=True))
        # Explicit vars select paths without mutating sysconfig or sys.prefix.
        install_paths = sysconfig.get_paths(
            scheme='venv', vars={'base': str(prefix), 'platbase': str(prefix)})
        assert Path(install_paths['scripts']).resolve(strict=True) == executable.parent.resolve(strict=True)
    else:
        assert sys.prefix == sys.base_prefix
        assert sys.exec_prefix == sys.base_exec_prefix
        assert executable.resolve(strict=True) == Path(sys._base_executable).resolve(strict=True)
        install_paths = sysconfig.get_paths()
        assert Path(install_paths['scripts']).resolve(strict=True) == executable.parent.resolve(strict=True)
    installed = {Path(install_paths[key]).resolve(strict=True)
                 for key in ('purelib', 'platlib')}
    search_directories = {Path(p).resolve(strict=True) for p in sys.path
                          if p and Path(p).is_dir()}
    assert installed <= search_directories, (installed, search_directories)
    for name, module in tuple(sys.modules.items()):
        if name == 'charset_normalizer' or name.startswith('charset_normalizer.'):
            assert module is not None and module.__spec__ is not None, name
            actual = Path(module.__file__).resolve(strict=True)
            assert Path(module.__spec__.origin).resolve(strict=True) == actual, name
            relative = Path(*name.split('.'))
            if hasattr(module, '__path__'):
                candidates = {base / relative / '__init__.py' for base in installed}
                assert [Path(p).resolve(strict=True) for p in module.__path__] == [actual.parent], name
            else:
                candidates = {base / (str(relative) + suffix)
                              for base in installed for suffix in all_suffixes()}
            assert actual in candidates, (name, actual)

def identity(package, name, module):
    value = getattr(package, name)
    assert value is getattr(module, name)
    assert value.__module__ == module.__name__
    assert vars(package)[name] is value
    before = tuple(import_attempts)
    assert getattr(package, name) is value
    assert tuple(import_attempts) == before

# The scenario is a literal suffix in this same file, not exec/eval.
"""


def _cold_source(target):
    return (
        "\ntarget = "
        + repr(target)
        + r"""
try:
    module = importlib.import_module(target)
    import rosclaw
    import rosclaw.practice as practice
    import rosclaw.practice.storage as storage
    assert rosclaw.practice is practice
    assert practice.storage is storage
    assert storage.__all__ == storage_names
    assert set(storage_names) <= set(dir(storage))
    assert 'FallbackSync' not in vars(storage)
    unknowns(storage)
    no_http()
    if target == 'rosclaw.practice.storage':
        assert not any(name in vars(storage) for name in storage_names)
        assert 'rosclaw.practice.storage.catalog' not in sys.modules
        assert 'rosclaw.practice.storage.layout' not in sys.modules
    else:
        leaf = target.rsplit('.', 1)[1]
        assert getattr(storage, leaf) is module
    from rosclaw.practice.storage import PracticeCatalog, PracticeLayout
    catalog = importlib.import_module('rosclaw.practice.storage.catalog')
    layout = importlib.import_module('rosclaw.practice.storage.layout')
    assert PracticeCatalog is catalog.PracticeCatalog
    assert PracticeLayout is layout.PracticeLayout
    identity(storage, 'PracticeCatalog', catalog)
    identity(storage, 'PracticeLayout', layout)
    assert storage.catalog is catalog and storage.layout is layout
    assert storage.__all__ == storage_names
    unknowns(storage)
    no_http()
    origins()
finally:
    assert not denials, repr(denials)
"""
    )


def _practice_source(mode):
    if mode == "from":
        statement = r"""
from rosclaw.practice import (
    PracticeConfig, PracticeCoordinator, PracticeRecorder, PracticeSession,
    PracticeSummary, UnifiedTimeline, TimelineChannel, TimelineEntry, EpisodeRecorder,
)
"""
    elif mode == "star":
        statement = "\nfrom rosclaw.practice import *\n"
    else:
        raise ValueError(mode)
    return (
        r"""
import rosclaw.practice as practice
assert practice.__all__ == practice_names
assert not any(n in vars(practice) for n in practice_names)
"""
        + statement
        + r"""
try:
    assert practice.__all__ == practice_names
    for name, leaf in zip(practice_names, practice_modules, strict=True):
        module = importlib.import_module('rosclaw.practice.' + leaf)
        assert globals()[name] is getattr(module, name)
        assert getattr(practice, leaf) is module
        identity(practice, name, module)
    import rosclaw.practice.storage as storage
    assert practice.storage is storage
    assert storage.__all__ == storage_names
    assert 'FallbackSync' not in vars(storage)
    unknowns(practice)
    unknowns(storage)
    no_http()
    origins()
finally:
    assert not denials, repr(denials)
"""
    )


def _fallback_source(mode):
    if mode == "explicit":
        statement = "\n    from rosclaw.practice.storage import FallbackSync\n"
    elif mode == "star":
        statement = "\n    from rosclaw.practice.storage import *\n"
    else:
        raise ValueError(mode)
    return (
        r"""
import rosclaw.practice.storage as storage
assert storage.__all__ == storage_names
assert 'FallbackSync' not in vars(storage)
no_http()
compatibility_completed = False
try:
"""
        + statement
        + r"""
    fallback = importlib.import_module('rosclaw.practice.storage.fallback_sync')
    import requests
    assert FallbackSync is fallback.FallbackSync
    assert fallback.requests is requests
    assert fallback.requests.post is requests.post
    assert callable(requests.post)
    assert storage.fallback_sync is fallback
    identity(storage, 'FallbackSync', fallback)
    if 'PracticeCatalog' in globals():
        for name, leaf in (('PracticeCatalog', 'catalog'), ('PracticeLayout', 'layout')):
            module = importlib.import_module('rosclaw.practice.storage.' + leaf)
            assert globals()[name] is getattr(module, name)
            identity(storage, name, module)
    assert storage.__all__ == storage_names
    unknowns(storage)
    origins()
    compatibility_completed = True
    print('STORAGE_CONTROL_V1 compatibility=complete', flush=True)
finally:
    # Full denials survive even when a dependency catches PermissionError.
    # Emit acceptance evidence only from the original strict assertion's
    # exception path, after every compatibility/origin assertion completed.
    print('explicit HTTP definition stage denials:', repr(denials), flush=True)
    if not compatibility_completed:
        print('STORAGE_CONTROL_V1 compatibility=incomplete', flush=True)
    try:
        assert not denials, repr(denials)
    except AssertionError:
        if (compatibility_completed and denials and
                all(event == 'socket.__new__' for event, detail in denials)):
            print('STORAGE_CONTROL_V1 effects=rejected:socket.__new__', flush=True)
        raise
"""
    )


def _run_reviewed_source(source):
    """Authored bounded transport; not invoked during this source-only task.

    Each child has 90s wall / 60s CPU / 2GiB address space / 1MiB per
    output-file bounds. Timeout propagates as failure after subprocess.run
    kills/reaps its owned child. No retry, environment repair or status rewrite.
    Output uses files, not unbounded PIPE capture. Root still supplies external
    containment/observation, including dependency shared-extension review.
    """
    import os
    import subprocess
    import sys
    import sysconfig
    import tempfile

    if os.name != "posix":
        raise RuntimeError("BLOCKED: POSIX resource limits required")
    root = Path(__file__).resolve().parents[2] / "src"
    root = root.resolve(strict=True)
    for relative in (
        "rosclaw/__init__.py",
        "rosclaw/practice/__init__.py",
        "rosclaw/practice/storage/__init__.py",
        "rosclaw/practice/coordinator.py",
    ):
        if not (root / relative).is_file():
            raise RuntimeError("BLOCKED: complete legitimate product source required")
    paths = [str(root)]
    for key in ("purelib", "platlib"):
        path = sysconfig.get_path(key)
        if path and path not in paths:
            paths.append(path)
    bootstrap = (
        "import resource\nimport sys\n"
        "resource.setrlimit(resource.RLIMIT_CPU, (60, 60))\n"
        "resource.setrlimit(resource.RLIMIT_AS, (2147483648, 2147483648))\n"
        "resource.setrlimit(resource.RLIMIT_CORE, (0, 0))\n"
        "resource.setrlimit(resource.RLIMIT_FSIZE, (1048576, 1048576))\n"
    )
    with tempfile.TemporaryDirectory(prefix="storage-import-") as temporary:
        directory = Path(temporary)
        script = directory / "scenario.py"
        script.write_text(
            bootstrap + "sys.path[:0] = " + repr(paths) + "\n" + _common_source() + source,
            encoding="utf-8",
        )
        environment = {
            "HOME": temporary,
            "TMPDIR": temporary,
            "TMP": temporary,
            "TEMP": temporary,
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "ROSCLAW_CANDIDATE_SRC": str(root),
            "JOBLIB_MULTIPROCESSING": "0",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        }
        # An owned, genuinely absent prefix prevents reads of existing pyc too.
        pycache = directory / "missing-pycache"
        assert not pycache.exists() and not pycache.is_symlink()
        argv = [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-X",
            "pycache_prefix=" + str(pycache),
            str(script),
        ]
        stdout = directory / "stdout"
        stderr = directory / "stderr"
        with stdout.open("wb") as out, stderr.open("wb") as err:
            result = subprocess.run(
                argv,
                cwd=directory,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                timeout=90,
                check=False,
                close_fds=True,
            )
        assert stdout.stat().st_size < 1048576, "stdout limit reached"
        assert stderr.stat().st_size < 1048576, "stderr limit reached"
        result.stdout = stdout.read_text(errors="replace")
        result.stderr = stderr.read_text(errors="replace")
        # Only the two exact reviewed HTTP suffixes use the negative protocol.
        # All five positive scenarios retain their original zero-exit assertion.
        if source not in (_fallback_source("explicit"), _fallback_source("star")):
            assert result.returncode == 0, (
                result.returncode,
                stdout.read_text(errors="replace"),
                stderr.read_text(errors="replace"),
            )
        return result


@pytest.mark.parametrize(
    "target",
    [
        "rosclaw.practice.storage",
        "rosclaw.practice.storage.catalog",
        "rosclaw.practice.storage.layout",
    ],
)
def test_cold_storage_without_unrequested_http(target):
    _run_reviewed_source(_cold_source(target))


@pytest.mark.parametrize("mode", ["from", "star"])
def test_original_nine_practice_exports_without_unrequested_http(mode):
    _run_reviewed_source(_practice_source(mode))


@pytest.mark.parametrize("mode", ["explicit", "star"])
def test_requested_fallback_compatibility_and_strict_effects(mode):
    result = _run_reviewed_source(_fallback_source(mode))
    lines = result.stdout.splitlines()
    complete = "STORAGE_CONTROL_V1 compatibility=complete"
    incomplete = "STORAGE_CONTROL_V1 compatibility=incomplete"
    rejected = "STORAGE_CONTROL_V1 effects=rejected:socket.__new__"
    # These are reviewed child checkpoints, not proof against malicious code;
    # Root's independent origin/effects observation is still mandatory.
    compatibility = (
        "complete" if complete in lines else "incomplete" if incomplete in lines else "unconfirmed"
    )
    evidence = {
        "mode": mode,
        "compatibility": compatibility,
        "effects": "rejected:socket.__new__" if rejected in lines else "unconfirmed",
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    print("explicit HTTP negative-control evidence:", evidence)
    assert result.returncode == 1, evidence
    assert lines.count(complete) == 1 and incomplete not in lines, evidence
    assert lines.count(rejected) == 1, evidence
    denial_lines = [
        line for line in lines if line.startswith("explicit HTTP definition stage denials:")
    ]
    assert len(denial_lines) == 1, evidence
    assert "socket.__new__" in denial_lines[0], evidence
    assert lines.index(complete) < lines.index(denial_lines[0]) < lines.index(rejected), evidence
    assert lines[-1] == rejected, evidence
    # Require the real uncaught strict assertion, not an arbitrary nonzero exit.
    assert "Traceback (most recent call last):" in result.stderr, evidence
    assert "assert not denials, repr(denials)" in result.stderr, evidence
    assert result.stderr.splitlines()[-1].startswith("AssertionError: ["), evidence
    assert "socket.__new__" in result.stderr.splitlines()[-1], evidence
    print("accepted negative control: compatibility complete; strict effects rejected")


def test_generic_encoding_dependency_is_not_http():
    """Real encoding API, not a fake namespace or an alternate classifier."""
    _run_reviewed_source(r"""
try:
    import charset_normalizer
    import charset_normalizer.constant
    assert charset_normalizer.from_bytes(b'plain encoding sample').best() is not None
    assert 'charset_normalizer.constant' in sys.modules
    before = tuple(import_attempts)
    no_http()
    assert tuple(import_attempts)[:len(before)] == before
    assert 'requests' not in sys.modules and 'urllib3' not in sys.modules
    origins()
finally:
    assert not denials, repr(denials)
""")


def test_scientific_encoding_dependency_is_not_http():
    """Exercise the genuine f2py consumer; no compiler or Fortran parsing."""
    _run_reviewed_source(r"""
try:
    from numpy.f2py import crackfortran
    import charset_normalizer
    assert crackfortran.charset_normalizer is charset_normalizer
    assert 'charset_normalizer.constant' in sys.modules
    no_http()
    origins()
finally:
    assert not denials, repr(denials)
""")


def test_encoding_presence_does_not_hide_forbidden_http_attempts():
    """Resolve genuine root specs under the retained observer, without running
    HTTP modules (urllib3 definition itself can attempt a socket). Each root
    gets a fresh child so a prior rejection cannot mask a later classifier bug.
    This checks attempted-import classification, not HTTP compatibility; the
    original explicit fallback controls still check actual imports and effects.
    """
    for target in (
        "requests",
        "urllib3",
        "rosclaw.practice.storage.fallback_sync",
        "chardet",
        "idna",
        "certifi",
    ):
        _run_reviewed_source(
            "\nhttp_control_target = "
            + repr(target)
            + r"""
try:
    import importlib.util
    import charset_normalizer
    no_http()
    assert http_control_target not in sys.modules
    before = tuple(import_attempts)
    spec = importlib.util.find_spec(http_control_target)
    if spec is None:
        # Absence is still an observed resolution attempt, not a skipped root.
        assert import_attempts[len(before):].count(http_control_target) == 1, import_attempts
        assert http_control_target not in sys.modules
    else:
        assert spec.origin, http_control_target
        assert Path(spec.origin).resolve(strict=True).is_file()
    assert http_control_target in import_attempts[len(before):], import_attempts
    assert http_control_target not in sys.modules
    observed = tuple(import_attempts)
    try:
        no_http()
    except AssertionError as error:
        assert error.args == (http_control_target,), error.args
    else:
        raise AssertionError('retained classifier accepted HTTP attempt')
    assert tuple(import_attempts) == observed
    assert 'charset_normalizer' in sys.modules
    origins()
finally:
    assert not denials, repr(denials)
"""
        )
