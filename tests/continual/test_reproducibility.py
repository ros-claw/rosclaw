from __future__ import annotations

from rosclaw.continual.reproducibility import NumericalRuntimeContract


def test_single_threaded_contract_fails_closed_on_missing_environment() -> None:
    contract = NumericalRuntimeContract.single_threaded_cpu(random_seed=17)

    report = contract.verify_environment({})

    assert not report.passed
    assert "OMP_NUM_THREADS" in report.mismatches
    assert "PYTHONHASHSEED" in report.mismatches


def test_subprocess_environment_satisfies_the_exact_contract() -> None:
    contract = NumericalRuntimeContract.single_threaded_cpu(random_seed=17)

    environment = contract.subprocess_environment({"PATH": "/bin"})
    report = contract.verify_environment(environment)

    assert report.passed
    assert environment["OPENBLAS_NUM_THREADS"] == "1"
    assert environment["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert environment["PATH"] == "/bin"


def test_runtime_hash_binds_seed_and_onnx_settings() -> None:
    first = NumericalRuntimeContract.single_threaded_cpu(random_seed=17)
    second = NumericalRuntimeContract.single_threaded_cpu(random_seed=18)

    assert first.contract_hash != second.contract_hash
    assert first.onnx_session_settings()["providers"] == ["CPUExecutionProvider"]
    assert first.onnx_session_settings()["execution_mode"] == "ORT_SEQUENTIAL"
