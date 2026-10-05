Prepared episode launchers should call `validate_episode_executable(path,
expected_sha256)` before writing their clock or starting the worker. A content
manifest alone does not establish that an executable copied between workspaces
can run: identical bytes at mode 664 caused the saved-scene FCL process to fail
before birth during a local supervision experiment.

The helper validates the declared regular file, pinned SHA256 and current
operator execute permission without changing files or starting a process.
Preparation must additionally perform a harmless real startup fixture to test
the loader and dependencies. It is an explicit launcher API; existing launchers
are not automatically changed. This check provides no filesystem isolation,
protection against changes after admission, or physical execution evidence.
