"""Media verification closes the real PIL-owned file on every decode path."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from rosclaw.task_kernel.verifier_plugins import TrajectoryVerifier


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("valid", []),
        ("blank", ["MEDIA_BLANK"]),
        ("one_frame", ["MEDIA_INSUFFICIENT_FRAMES"]),
        ("seek_failure", ["MEDIA_UNDECODABLE"]),
        ("convert_failure", ["MEDIA_UNDECODABLE"]),
    ],
)
def test_real_image_fd_closed_and_verdict_preserved(tmp_path: Path, monkeypatch, mode, expected):
    path = tmp_path / "media.gif"
    frames = []
    for i in range(1 if mode == "one_frame" else 2):
        pixels = np.arange(256, dtype=np.uint8).reshape(16, 16)
        if mode == "blank":
            pixels = np.full((16, 16), i * 255, dtype=np.uint8)
        elif i:
            pixels = np.flipud(pixels)
        frames.append(Image.fromarray(pixels))
    try:
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=10, loop=0)
    finally:
        for frame in frames:
            frame.close()

    actual_open = Image.open
    images = []
    handles = []

    def observed_open(*args, **kwargs):
        image = actual_open(*args, **kwargs)
        images.append(image)
        handles.append(image.fp)
        if mode in {"seek_failure", "convert_failure"}:

            def fail(*args, **kwargs):
                raise OSError("private decode fault after actual file open")

            monkeypatch.setattr(image, "seek" if mode == "seek_failure" else "convert", fail)
        return image

    monkeypatch.setattr(Image, "open", observed_open)
    try:
        verdict = TrajectoryVerifier._media_failures(path)
        assert len(verdict) == len(expected)
        for prefix in expected:
            assert any(item.startswith(prefix) for item in verdict), verdict
        assert handles and all(handle.closed for handle in handles)
    finally:
        # Retain the image to distinguish explicit close from garbage collection.
        # Also release the FD if this test is run against the old RED source.
        for image in images:
            image.close()


def test_corrupt_file_stays_media_undecodable(tmp_path: Path):
    path = tmp_path / "corrupt.gif"
    path.write_bytes(b"not a GIF")
    assert TrajectoryVerifier._media_failures(path) == [
        "MEDIA_UNDECODABLE: corrupt.gif 不是可解码的图像",
    ]
