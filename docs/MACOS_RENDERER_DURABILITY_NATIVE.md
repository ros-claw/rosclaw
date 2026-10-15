# macOS renderer and durability source repair

This is SOFTWARE SOURCE ONLY. Real macOS, Ubuntu remote CI, actual MuJoCo/GL,
camera fidelity and actual vision engine tests are NOT_RUN locally. Those frozen
integration tests and remote CI remain required; inert boundary tests are not
physics, pixel correctness, renderer availability or runtime acceptance evidence.

## Renderer binding

The two producer entrypoints select only CGL on Darwin and retain egl then osmesa
on other platforms (including Linux). Their isolated workers overwrite inherited
MUJOCO_GL and PYOPENGL_PLATFORM before importing MuJoCo. Darwin CGL uses
MUJOCO_GL=cgl with PYOPENGL_PLATFORM=darwin, not the unregistered cgl plugin name.
The oracle independently checks its platform-specific allowlist and pins the
producer-declared renderer, with no fallback to another renderer. Unrelated
environment entries are preserved. Model/state capture, physics, rendering bodies,
scientific comparisons and error handling are unchanged.

Official static MuJoCo 3.13.0 gl_context source dispatches Darwin to mujoco.cgl;
the supplied CGL source creates/frees Apple offscreen contexts. The supplied
PyOpenGL plugin registration supports darwin, egl and osmesa, not cgl. These are
API facts, not evidence that this host or any remote runner can render.

## Durability observation

Only the six original tests' /proc/self/fd call expressions are replaced. The
single observer reads Linux descriptor symlinks or uses Darwin fcntl F_GETPATH
(command 50, MAXPATHLEN 1024 bytes), decodes the NUL-terminated filesystem path,
and propagates errors. It never guesses from cwd or skips a barrier. The existing
cross-process fail/retry script contains the same helper AST before first use.
Production store.py and storage/durable.py remain frozen. All original assertions,
real fsync calls, injection messages, ancestor ordering and subprocess flags remain.

## Inert regression coverage

Ten unparametrized tests exercise actual camera/GIF producer entrypoints through
an inert subprocess boundary, Linux ordered fallback, actual independent oracle
request/results with inert subprocess output, platform rejection, and inherited
foreign environment replacement. Both complete worker strings execute only until
an import interceptor stops at MuJoCo, verifying environment values at that exact
boundary; no engine or OpenGL is imported or initialized. Portable observations
check real open file/directory identity and an injected Darwin F_GETPATH boundary,
including error propagation. Original targeted directory failure and deep ancestor
retry tests are also invoked unchanged. The authorized runner separately runs the
full frozen original durability selection, including genuine cross-process retry.

These tests intentionally do not create a renderer or assert scientific pixel
correctness. They do not grant real Darwin/Ubuntu execution credit. Root review
of helper effects, final diff and remote CI remains necessary.
