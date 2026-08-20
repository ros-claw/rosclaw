"""Evidence-bound 1080p promotional pack for the G1 football curriculum.

The pack deliberately separates physics claims from visual transformations.
Right-foot, moving-ball, and coupled-relay clips are downstream renders of
saved SIM trajectories.  The left-foot clip is an explicitly labelled image
symmetry augmentation; it must never be interpreted as left-foot physics.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rosclaw.simforge.g1_hat_trick_video import _escape_filtergraph_option

_WIDTH = 1920
_HEIGHT = 1080
_DEFAULT_FPS = 30


@dataclass(frozen=True)
class G1PromoSource:
    source_id: str
    manifest_path: str
    manifest_hash: str
    video_path: str
    video_hash: str
    width: int
    height: int
    duration_sec: float
    strict_physics_source: bool
    simultaneous_two_body_physics: bool
    candidate_only: bool


@dataclass(frozen=True)
class G1PromoArtifact:
    artifact_id: str
    output_path: str
    video_hash: str
    width: int
    height: int
    fps: int
    duration_sec: float
    clip_count: int
    contains_symmetry_augmented_left_foot: bool
    visualization_only: bool = True


@dataclass(frozen=True)
class G1PromoPackResult:
    output_dir: str
    manifest_path: str
    report_path: str
    sources: tuple[G1PromoSource, ...]
    artifacts: tuple[G1PromoArtifact, ...]
    left_foot_physics_claimed: bool = False
    activation_ceiling: str = "SIM_ONLY"
    pixels_used_for_scoring: bool = False
    schema_version: str = "rosclaw.simforge.g1_promo_pack.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "sources": [asdict(source) for source in self.sources],
            "artifacts": [asdict(artifact) for artifact in self.artifacts],
            "claims": {
                "actual_left_foot_physics": False,
                "left_foot_is_symmetry_augmented_visualization": True,
                "coupled_relay_uses_simultaneous_two_body_physics": True,
                "moving_ball_clips_use_strict_replay_evidence": True,
                "pixels_used_for_task_scoring": False,
                "real_hardware": False,
            },
        }


@dataclass(frozen=True)
class _Clip:
    source_id: str
    start_sec: float
    duration_sec: float
    title: str
    subtitle: str
    footer: str
    mirror: bool = False


def render_g1_promo_pack(
    *,
    precision_manifest_path: Path,
    coupled_manifest_path: Path,
    moving_manifest_path: Path,
    output_dir: Path,
    source_checkout: Path,
    fps: int = _DEFAULT_FPS,
) -> G1PromoPackResult:
    """Create a long master reel and two focused 1080p cut-downs."""

    root = output_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if root == checkout or checkout in root.parents:
        raise ValueError("promo pack must be written outside the source checkout")
    if not 24 <= fps <= 60:
        raise ValueError("promo pack fps must be in [24, 60]")
    if root.exists():
        raise FileExistsError("promo pack output directory already exists")
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise RuntimeError("ffmpeg and ffprobe are required for promo pack export")

    precision, precision_data = _load_source(
        source_id="precision",
        manifest_path=precision_manifest_path,
        checkout=checkout,
        ffprobe=ffprobe,
    )
    coupled, coupled_data = _load_source(
        source_id="coupled",
        manifest_path=coupled_manifest_path,
        checkout=checkout,
        ffprobe=ffprobe,
    )
    moving, moving_data = _load_source(
        source_id="moving",
        manifest_path=moving_manifest_path,
        checkout=checkout,
        ffprobe=ffprobe,
    )
    _require_source_contracts(
        precision=precision,
        precision_data=precision_data,
        coupled=coupled,
        coupled_data=coupled_data,
        moving=moving,
        moving_data=moving_data,
    )

    root.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="rosclaw-g1-promo-") as temporary:
        temp = Path(temporary)
        continuous = precision_data["clips"][1]
        precision_start = float(precision_data["clips"][0]["duration_sec"])
        precision_duration = float(continuous["duration_sec"])
        moving_offsets = _clip_offsets(moving_data["clips"])
        coupled_offsets = _clip_offsets(coupled_data["clips"])

        master_clips = (
            _Clip(
                source_id="precision",
                start_sec=precision_start,
                duration_sec=precision_duration,
                title="RIGHT-FOOT LONG-RANGE STRIKE",
                subtitle="7.5 m · LEARNED CONTACT ACTOR · UPPER-LEFT CORNER",
                footer="STRICT TRAJECTORY REPLAY · DEVELOPMENT CANDIDATE · SIM ONLY",
            ),
            _Clip(
                source_id="precision",
                start_sec=precision_start,
                duration_sec=precision_duration,
                title="LEFT-FOOT SYMMETRY STUDY",
                subtitle="MIRRORED POSE AUGMENTATION · UPPER-RIGHT VISUAL",
                footer="VISUAL AUGMENTATION ONLY · NOT LEFT-FOOT PHYSICS EVIDENCE",
                mirror=True,
            ),
            _moving_clip(moving_data, moving_offsets, 0, "OFFSET BALL · INCOMING INTERCEPT"),
            _moving_clip(moving_data, moving_offsets, 1, "LATERAL BALL · FAST INTERCEPT"),
            _moving_clip(moving_data, moving_offsets, 2, "FRICTION EDGE · ADAPTIVE CONTACT"),
            _coupled_clip(
                coupled_data,
                coupled_offsets,
                4,
                "TWO G1s · PASS → ONE-TOUCH FINISH",
            ),
        )
        source_paths = {
            "precision": Path(precision.video_path),
            "coupled": Path(coupled.video_path),
            "moving": Path(moving.video_path),
        }
        master = root / "rosclaw-g1-all-star-combo-1080p.mp4"
        _render_sequence(
            ffmpeg=ffmpeg,
            output=master,
            temp=temp / "master",
            source_paths=source_paths,
            clips=master_clips,
            title="ROSClaw · G1 ALL-STAR FOOTBALL COMBO",
            subtitle="TWO FEET · MOVING BALLS · TWO-G1 ONE-TOUCH RELAY",
            fps=fps,
        )

        dual = root / "rosclaw-g1-dual-foot-pose-study-1080p.mp4"
        _render_dual_foot(
            ffmpeg=ffmpeg,
            source=Path(precision.video_path),
            output=dual,
            start_sec=precision_start,
            duration_sec=precision_duration,
            temp=temp / "dual",
            fps=fps,
        )

        relay_clips = tuple(
            _coupled_clip(coupled_data, coupled_offsets, index, title)
            for index, title in (
                (0, "EARLY BALL · ONLINE PHASE HOLD"),
                (2, "HIGH TARGET · ONE-TOUCH PRECISION"),
                (4, "LATE BALL · ONLINE PHASE ADVANCE"),
            )
        )
        relay = root / "rosclaw-g1-two-player-relay-highlights-1080p.mp4"
        _render_sequence(
            ffmpeg=ffmpeg,
            output=relay,
            temp=temp / "relay",
            source_paths=source_paths,
            clips=relay_clips,
            title="ROSClaw · TWO-G1 ONE-TOUCH RELAY",
            subtitle="ONE WORLD · ONE BALL · THREE TIMING CHALLENGES",
            fps=fps,
        )

    artifacts = (
        _artifact("all-star-combo", master, ffprobe, fps, len(master_clips), True),
        _artifact("dual-foot-study", dual, ffprobe, fps, 2, True),
        _artifact("two-player-relay", relay, ffprobe, fps, len(relay_clips), False),
    )
    manifest = root / "rosclaw-g1-promo-pack.json"
    report = root / "宣传视频说明.md"
    result = G1PromoPackResult(
        output_dir=str(root),
        manifest_path=str(manifest),
        report_path=str(report),
        sources=(precision, moving, coupled),
        artifacts=artifacts,
    )
    manifest.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report.write_text(_report(result), encoding="utf-8")
    return result


def _load_source(
    *,
    source_id: str,
    manifest_path: Path,
    checkout: Path,
    ffprobe: str,
) -> tuple[G1PromoSource, dict[str, Any]]:
    manifest = manifest_path.expanduser().resolve()
    if manifest == checkout or checkout in manifest.parents:
        raise ValueError("promo source manifests must remain outside the source checkout")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data.get("visualization_only") is not True:
        raise ValueError(f"{source_id} source is not a visualization-only manifest")
    video = Path(str(data["output_path"])).expanduser().resolve()
    if video == checkout or checkout in video.parents:
        raise ValueError("promo source videos must remain outside the source checkout")
    video_hash = _file_hash(video)
    if video_hash != data.get("video_hash"):
        raise ValueError(f"{source_id} source video hash mismatch")
    width, height, duration = _probe(ffprobe, video)
    return (
        G1PromoSource(
            source_id=source_id,
            manifest_path=str(manifest),
            manifest_hash=_file_hash(manifest),
            video_path=str(video),
            video_hash=video_hash,
            width=width,
            height=height,
            duration_sec=duration,
            strict_physics_source=source_id in {"moving", "coupled"},
            simultaneous_two_body_physics=bool(data.get("simultaneous_two_body_physics", False)),
            candidate_only=bool(data.get("candidate_only", False)),
        ),
        data,
    )


def _require_source_contracts(
    *,
    precision: G1PromoSource,
    precision_data: dict[str, Any],
    coupled: G1PromoSource,
    coupled_data: dict[str, Any],
    moving: G1PromoSource,
    moving_data: dict[str, Any],
) -> None:
    if precision.width != _WIDTH or precision.height != _HEIGHT:
        raise ValueError("precision promo source must be native 1080p")
    if len(precision_data.get("clips", ())) < 2:
        raise ValueError("precision promo source is missing its continuous clip")
    if not (
        precision_data.get("source_evidence_passed") is True
        or precision_data.get("candidate_only") is True
    ):
        raise ValueError("precision promo source has no declared evidence status")
    if not coupled.simultaneous_two_body_physics or len(coupled_data.get("clips", ())) != 5:
        raise ValueError("coupled promo source must contain five simultaneous relay clips")
    _require_sibling_evidence(
        manifest=Path(coupled.manifest_path),
        name="g1-coupled-showcase.json",
        expected_hash=coupled_data.get("evidence_report_hash"),
    )
    if len(moving_data.get("clips", ())) < 3:
        raise ValueError("moving-ball promo source must contain three challenge clips")
    _require_sibling_evidence(
        manifest=Path(moving.manifest_path),
        name="g1-self-aware-showcase.json",
        expected_hash=moving_data.get("showcase_evidence_hash"),
    )


def _require_sibling_evidence(*, manifest: Path, name: str, expected_hash: Any) -> None:
    evidence = manifest.parent / name
    if _file_hash(evidence) != expected_hash:
        raise ValueError(f"{name} does not match its video manifest")
    data = json.loads(evidence.read_text(encoding="utf-8"))
    if data.get("passed") is not True:
        raise ValueError(f"{name} is not passing evidence")


def _moving_clip(
    data: dict[str, Any],
    offsets: tuple[float, ...],
    index: int,
    title: str,
) -> _Clip:
    source = data["clips"][index]
    return _Clip(
        source_id="moving",
        start_sec=offsets[index],
        duration_sec=float(source["duration_sec"]),
        title=title,
        subtitle=str(source["title"]),
        footer="STRICT CPU MUJOCO REPLAY · MOVING-BALL CLOSED LOOP · SIM ONLY",
    )


def _coupled_clip(
    data: dict[str, Any],
    offsets: tuple[float, ...],
    index: int,
    title: str,
) -> _Clip:
    source = data["clips"][index]
    return _Clip(
        source_id="coupled",
        start_sec=offsets[index],
        duration_sec=float(source["duration_sec"]),
        title=title,
        subtitle=str(source["title"]),
        footer="STRICT CPU MUJOCO REPLAY · TWO LIVE G1s · ONE PHYSICAL BALL · SIM ONLY",
    )


def _clip_offsets(clips: list[dict[str, Any]]) -> tuple[float, ...]:
    values: list[float] = []
    offset = 0.0
    for clip in clips:
        values.append(offset)
        offset += float(clip["duration_sec"])
    return tuple(values)


def _render_sequence(
    *,
    ffmpeg: str,
    output: Path,
    temp: Path,
    source_paths: dict[str, Path],
    clips: tuple[_Clip, ...],
    title: str,
    subtitle: str,
    fps: int,
) -> None:
    temp.mkdir(parents=True)
    parts = [
        _title_card(
            ffmpeg=ffmpeg,
            output=temp / "000-title.mp4",
            title=title,
            subtitle=subtitle,
            footer="ROSCLAW GROWTH ENGINE · PHYSICAL-AI DEVELOPMENT SHOWCASE · SIM ONLY",
            fps=fps,
            temp=temp,
        )
    ]
    for index, clip in enumerate(clips, start=1):
        part = temp / f"{index:03d}-{clip.source_id}.mp4"
        _source_clip(
            ffmpeg=ffmpeg,
            source=source_paths[clip.source_id],
            output=part,
            clip=clip,
            fps=fps,
            temp=temp,
            index=index,
        )
        parts.append(part)
    concat = temp / "concat.txt"
    concat.write_text(
        "".join(f"file '{_concat_path(path)}'\n" for path in parts),
        encoding="utf-8",
    )
    _run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(output),
        ],
        "promo sequence concat",
    )


def _title_card(
    *,
    ffmpeg: str,
    output: Path,
    title: str,
    subtitle: str,
    footer: str,
    fps: int,
    temp: Path,
) -> Path:
    files = _label_files(temp, "title", title, subtitle, footer)
    font = _font_option()
    filters = (
        "drawbox=x=120:y=250:w=1680:h=420:color=0x081827@0.94:t=fill,"
        f"drawtext={font}textfile={_escape_filtergraph_option(str(files[0]))}:"
        "expansion=none:x=(w-text_w)/2:y=335:fontsize=66:fontcolor=white,"
        f"drawtext={font}textfile={_escape_filtergraph_option(str(files[1]))}:"
        "expansion=none:x=(w-text_w)/2:y=455:fontsize=34:fontcolor=0x65F59A,"
        f"drawtext={font}textfile={_escape_filtergraph_option(str(files[2]))}:"
        "expansion=none:x=(w-text_w)/2:y=565:fontsize=24:fontcolor=0x8DD8FF,"
        "fade=t=in:st=0:d=0.35,fade=t=out:st=2.25:d=0.35"
    )
    _run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c=0x02070d:s={_WIDTH}x{_HEIGHT}:r={fps}:d=2.6",
            "-vf",
            filters,
            *_encode_args(fps),
            str(output),
        ],
        "promo title card",
    )
    return output


def _source_clip(
    *,
    ffmpeg: str,
    source: Path,
    output: Path,
    clip: _Clip,
    fps: int,
    temp: Path,
    index: int,
) -> None:
    labels = _label_files(temp, f"clip-{index}", clip.title, clip.subtitle, clip.footer)
    font = _font_option()
    mirror = "hflip," if clip.mirror else ""
    fade_out = max(0.0, clip.duration_sec - 0.25)
    filters = (
        f"fps={fps},scale={_WIDTH}:{_HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={_WIDTH}:{_HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=0x02070d,{mirror}"
        "drawbox=x=0:y=0:w=iw:h=220:color=0x040913@1.0:t=fill,"
        "drawbox=x=0:y=ih-110:w=iw:h=110:color=0x040913@1.0:t=fill,"
        f"drawtext={font}textfile={_escape_filtergraph_option(str(labels[0]))}:"
        "expansion=none:x=48:y=24:fontsize=48:fontcolor=white,"
        f"drawtext={font}textfile={_escape_filtergraph_option(str(labels[1]))}:"
        "expansion=none:x=48:y=102:fontsize=30:fontcolor=0x65F59A,"
        f"drawtext={font}textfile={_escape_filtergraph_option(str(labels[2]))}:"
        "expansion=none:x=48:y=h-72:fontsize=25:fontcolor=0x8DD8FF,"
        f"fade=t=in:st=0:d=0.25,fade=t=out:st={fade_out:.6f}:d=0.25"
    )
    _run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{clip.start_sec:.6f}",
            "-t",
            f"{clip.duration_sec:.6f}",
            "-i",
            str(source),
            "-vf",
            filters,
            "-an",
            *_encode_args(fps),
            str(output),
        ],
        f"promo source clip {index}",
    )


def _render_dual_foot(
    *,
    ffmpeg: str,
    source: Path,
    output: Path,
    start_sec: float,
    duration_sec: float,
    temp: Path,
    fps: int,
) -> None:
    temp.mkdir(parents=True)
    labels = _label_files(
        temp,
        "dual",
        "ROSClaw · G1 DUAL-FOOT POSE STUDY",
        "RIGHT: STRICT TRAJECTORY REPLAY      LEFT: SYMMETRY-AUGMENTED VISUAL",
        "LEFT PANEL IS NOT LEFT-FOOT PHYSICS EVIDENCE · SIM ONLY",
    )
    font = _font_option()
    fade_out = max(0.0, duration_sec - 0.30)
    graph = (
        f"[0:v]fps={fps},split=2[right0][left0];"
        "[right0]crop=iw:ih-257:0:177,scale=960:412[right];"
        "[left0]hflip,crop=iw:ih-257:0:177,scale=960:412[left];"
        "[right][left]hstack=inputs=2,pad=1920:1080:0:300:color=0x02070d,"
        "drawbox=x=0:y=0:w=iw:h=225:color=0x040913@0.94:t=fill,"
        "drawbox=x=0:y=250:w=iw:h=52:color=0x040913@0.94:t=fill,"
        "drawbox=x=0:y=712:w=iw:h=368:color=0x040913@0.94:t=fill,"
        f"drawtext={font}textfile={_escape_filtergraph_option(str(labels[0]))}:"
        "expansion=none:x=(w-text_w)/2:y=42:fontsize=54:fontcolor=white,"
        f"drawtext={font}textfile={_escape_filtergraph_option(str(labels[1]))}:"
        "expansion=none:x=(w-text_w)/2:y=132:fontsize=28:fontcolor=0x65F59A,"
        f"drawtext={font}text='RIGHT FOOT':expansion=none:x=340:y=254:fontsize=28:fontcolor=0xFFD166,"
        f"drawtext={font}text='LEFT FOOT · VISUAL AUGMENT':expansion=none:x=1220:y=254:fontsize=28:fontcolor=0xFFD166,"
        f"drawtext={font}text='ONE LEARNED MOTION · TWO SYMMETRIC POSE VIEWS':"
        "expansion=none:x=(w-text_w)/2:y=790:fontsize=38:fontcolor=white,"
        f"drawtext={font}textfile={_escape_filtergraph_option(str(labels[2]))}:"
        "expansion=none:x=(w-text_w)/2:y=875:fontsize=26:fontcolor=0x8DD8FF,"
        f"fade=t=in:st=0:d=0.30,fade=t=out:st={fade_out:.6f}:d=0.30[out]"
    )
    _run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start_sec:.6f}",
            "-t",
            f"{duration_sec:.6f}",
            "-i",
            str(source),
            "-filter_complex",
            graph,
            "-map",
            "[out]",
            "-an",
            *_encode_args(fps),
            str(output),
        ],
        "dual-foot pose study",
    )


def _label_files(
    root: Path,
    stem: str,
    title: str,
    subtitle: str,
    footer: str,
) -> tuple[Path, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    paths = (
        root / f"{stem}-title.txt",
        root / f"{stem}-subtitle.txt",
        root / f"{stem}-footer.txt",
    )
    for path, value in zip(paths, (title, subtitle, footer), strict=True):
        path.write_text(value, encoding="utf-8")
    return paths


def _font_option() -> str:
    font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    return f"fontfile={_escape_filtergraph_option(str(font))}:" if font.is_file() else ""


def _encode_args(fps: int) -> list[str]:
    return [
        "-r",
        str(fps),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "17",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ]


def _artifact(
    artifact_id: str,
    path: Path,
    ffprobe: str,
    fps: int,
    clip_count: int,
    augmented: bool,
) -> G1PromoArtifact:
    width, height, duration = _probe(ffprobe, path)
    if width != _WIDTH or height != _HEIGHT:
        raise ValueError(f"{artifact_id} was not exported at 1080p")
    return G1PromoArtifact(
        artifact_id=artifact_id,
        output_path=str(path),
        video_hash=_file_hash(path),
        width=width,
        height=height,
        fps=fps,
        duration_sec=duration,
        clip_count=clip_count,
        contains_symmetry_augmented_left_foot=augmented,
    )


def _probe(ffprobe: str, path: Path) -> tuple[int, int, float]:
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height:format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(completed.stdout)
    stream = data["streams"][0]
    return int(stream["width"]), int(stream["height"]), float(data["format"]["duration"])


def _run(command: list[str], label: str) -> None:
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(f"{label} failed ({completed.returncode}): {completed.stderr[-2000:]}")


def _concat_path(path: Path) -> str:
    return str(path).replace("'", "'\\''")


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _report(result: G1PromoPackResult) -> str:
    artifacts = "\n".join(
        f"- `{Path(item.output_path).name}`：{item.duration_sec:.2f} 秒，"
        f"{item.width}×{item.height}，SHA-256 `{item.video_hash.removeprefix('sha256:')}`"
        for item in result.artifacts
    )
    return f"""# ROSClaw G1 足球宣传视频包说明

证据域：`SIM_ONLY`

用途：研发成果可视化；视频像素不参与任务评分或模型晋升。

## 成品

{artifacts}

## 组合内容

- 远距离右脚射门：来自保存的连续助跑—触球—恢复轨迹；其来源是开发候选，视频不改变候选门控结论。
- 左脚姿态：由同一条右脚轨迹做画面对称增强，只用于展示左右脚数据增强方向，**不是左脚物理仿真证据**。
- 三种移动球：来自三条分别执行、分别严格复跑的 CPU MuJoCo 轨迹，包含不同来球速度、横向漂移与摩擦条件。
- 双 G1 传射：来自同一个 MuJoCo 世界中的两台 G1 和一颗共享物理球；展示提前、标称与延迟来球三种闭环相位响应。

## 可准确宣传的表述

“ROSClaw 在 SIM_ONLY 的 CPU MuJoCo 环境中，将长距离射门、移动球处理与同场双 G1 一脚传射组织成可复核的足球技能组合；其中双 G1 与移动球片段绑定严格回放轨迹。”

不要表述为真机成绩、sim-to-real 已解决，或真实左脚策略已经训练完成。
"""


__all__ = [
    "G1PromoArtifact",
    "G1PromoPackResult",
    "G1PromoSource",
    "render_g1_promo_pack",
]
