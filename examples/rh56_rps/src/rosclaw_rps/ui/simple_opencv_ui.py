"""Simple OpenCV overlay UI for the RPS demo (Chinese + emoji/graphical)."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np

from ..gesture_schema import CameraFrame


# Mapping from internal labels to user-facing Chinese names.
_GESTURE_NAMES = {
    "rock": "石头",
    "paper": "布",
    "scissors": "剪刀",
    "ready": "OK",
    "count3": "三",
    "count2": "二",
    "count1": "一",
    "win": "赢",
    "lose": "输",
    "draw": "平",
    "error": "安全位",
    "unknown": "未识别",
    "left_ready": "左手就绪",
    "left_thumb_up": "左手大拇指",
    "left_pinkie": "左手小拇指",
    "left_orchid": "左手兰花指",
    "left_draw": "左手张开",
    "left_error": "左手安全位",
}

_GESTURE_ICONS = {
    "rock": "✊",
    "paper": "✋",
    "scissors": "✌️",
    "ready": "OK",
    "win": "赢",
    "lose": "输",
    "draw": "平",
    "error": "!",
    "unknown": "?",
    "left_thumb_up": "👍",
    "left_pinkie": "🤙",
    "left_orchid": "🌸",
    "left_draw": "✋",
}

# Fallback display when the color-emoji font cannot be loaded.
_GESTURE_TEXT = {
    "rock": "石头",
    "paper": "布",
    "scissors": "剪刀",
    "ready": "OK",
    "win": "赢",
    "lose": "输",
    "draw": "平",
    "error": "!",
    "unknown": "?",
    "left_thumb_up": "大拇指",
    "left_pinkie": "小拇指",
    "left_orchid": "兰花指",
    "left_draw": "张开",
    "left_ready": "就绪",
    "left_error": "安全位",
}

_RESULT_TEXT = {
    "robot_win": "机器人赢",
    "robot_lose": "你赢",
    "draw": "平局",
    "invalid": "无效",
}

_RESULT_SUBTEXT = {
    "robot_win": "（你输）",
    "robot_lose": "（机器人输）",
    "draw": "",
    "invalid": "未识别到手势",
}

_RESULT_COLORS = {
    "robot_win": (0, 165, 255),
    "robot_lose": (0, 0, 255),
    "draw": (0, 255, 255),
    "invalid": (128, 128, 128),
}

_COUNTDOWN_TEXT = {
    "3": "三",
    "2": "二",
    "1": "一",
    "shoot": "出",
}

_DEFAULT_FONTS = {
    "zh": [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        "/usr/share/fonts/truetype/arphic/ukai.ttc",
    ],
    "emoji": [],
}


def _load_font(paths, size):
    for p in paths:
        if Path(p).exists():
            try:
                from PIL import ImageFont

                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return None


class _FontCache:
    """Lazy-load Chinese and emoji fonts; fall back to None if not found."""

    def __init__(self):
        self._zh: Optional[object] = None
        self._emoji: Optional[object] = None
        self._big_zh: Optional[object] = None
        self._zh_size = 26
        # NotoColorEmoji only accepts specific fixed sizes on this system.
        self._emoji_size = 109
        self._big_size = 64

    @property
    def zh(self):
        if self._zh is None:
            self._zh = _load_font(_DEFAULT_FONTS["zh"], self._zh_size)
        return self._zh

    @property
    def emoji(self):
        if self._emoji is None:
            self._emoji = _load_font(_DEFAULT_FONTS["emoji"], self._emoji_size)
        return self._emoji

    @property
    def big_zh(self):
        if self._big_zh is None:
            self._big_zh = _load_font(_DEFAULT_FONTS["zh"], self._big_size)
        return self._big_zh


_FONT_CACHE = _FontCache()


def _emoji_font_available() -> bool:
    """Return True if the color-emoji font loaded at the required fixed size."""
    return _FONT_CACHE.emoji is not None


def _gesture_display(label: str) -> tuple[str, str]:
    """Return (chinese_name, icon) for a gesture label."""
    name = _GESTURE_NAMES.get(label, label)
    icon = _GESTURE_ICONS.get(label, "")
    return name, icon


class SimpleOpenCVUI:
    """Render state and countdown onto a camera frame."""

    def __init__(self, window_name: str = "RH56 Rock Paper Scissors"):
        self.window_name = window_name
        self._blink = 0
        self._has_zh = _FONT_CACHE.zh is not None
        self._has_emoji = _FONT_CACHE.emoji is not None

    @staticmethod
    def _draw_region(
        canvas: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        draw_fn: Callable,
    ) -> None:
        """Crop a BGR region, draw on it with PIL, and paste it back."""
        import cv2
        from PIL import Image

        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(canvas.shape[1], x1 + w), min(canvas.shape[0], y1 + h)
        if x2 <= x1 or y2 <= y1:
            return
        crop_bgr = canvas[y1:y2, x1:x2].copy()
        pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        draw_fn(pil, x2 - x1, y2 - y1)
        rgb = np.array(pil)
        canvas[y1:y2, x1:x2] = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def _draw_text_pil(
        self,
        draw,
        text: str,
        pos: tuple[int, int],
        font: Optional[object] = None,
        color: tuple[int, int, int] = (0, 255, 0),
        anchor: str = "lt",
    ) -> tuple[int, int]:
        """Draw text with PIL; returns bounding box size (w, h)."""
        from PIL import ImageFont

        f = font or _FONT_CACHE.zh or ImageFont.load_default()
        draw.text(pos, text, font=f, fill=color, anchor=anchor)
        bbox = draw.textbbox(pos, text, font=f, anchor=anchor)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    def render(
        self,
        frame: Optional[CameraFrame],
        round_id: str = "",
        countdown_label: str = "",
        robot_committed: bool = False,
        robot_choice: str = "",
        hand_label: str = "",
        human_label: str = "",
        result: str = "",
        bodysense: str = "",
        waiting_for_start: bool = False,
        referee: Optional[str] = None,
        camera_lost: bool = False,
    ) -> np.ndarray:
        import cv2

        if frame is None or frame.color is None:
            canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            canvas = frame.color.copy()
        h, w = canvas.shape[:2]

        if camera_lost:
            cv2.rectangle(canvas, (0, 0), (w, h), (0, 0, 0), -1)
            cv2.putText(
                canvas,
                "No camera feed",
                (w // 2 - 140, h // 2 - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                canvas,
                "Check RealSense node / USB",
                (w // 2 - 200, h // 2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200, 200, 200),
                2,
            )
            return canvas

        # Right-side info panel.
        panel_w = 240
        panel_h = 260
        panel_x = max(0, w - panel_w - 10)
        panel_y = 10

        # Draw panel background with OpenCV (fast), then text with PIL on top.
        cv2.rectangle(
            canvas,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            (20, 20, 20),
            -1,
        )
        cv2.rectangle(
            canvas,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            (80, 80, 80),
            2,
        )

        def panel_text(pil, pw, ph):
            from PIL import ImageDraw

            draw = ImageDraw.Draw(pil)
            x = 12
            y = 18
            line_h = 32

            rid = round_id.replace("round_", "") if round_id else "--"
            self._draw_text_pil(draw, f"第 {rid} 轮", (x, y), color=(0, 255, 0))
            y += line_h

            commit_text = "机器人已选" if robot_committed else "机器人未选"
            self._draw_text_pil(draw, commit_text, (x, y), color=(200, 200, 200))
            y += line_h

            if hand_label:
                hname = _GESTURE_NAMES.get(hand_label, hand_label)
                self._draw_text_pil(draw, f"机器人手：{hname}", (x, y), color=(0, 200, 255))
                y += line_h

            if robot_choice:
                name = _GESTURE_NAMES.get(robot_choice, robot_choice)
                self._draw_text_pil(draw, f"机器人出：{name}", (x, y), color=(0, 200, 255))
                y += line_h

            if human_label:
                name = _GESTURE_NAMES.get(human_label, human_label)
                self._draw_text_pil(draw, f"你：{name}", (x, y), color=(255, 200, 0))
                y += line_h

            if result:
                rtext = _RESULT_TEXT.get(result, result)
                color = (0, 255, 0)
                if result == "robot_win":
                    color = (0, 165, 255)
                elif result == "robot_lose":
                    color = (255, 0, 0)
                elif result == "draw":
                    color = (0, 255, 255)
                elif result == "invalid":
                    color = (128, 128, 128)
                self._draw_text_pil(draw, f"结果：{rtext}", (x, y), color=color)
                y += line_h

            if bodysense:
                if bodysense == "verified":
                    btext, bcolor = "已验证", (0, 255, 0)
                else:
                    btext, bcolor = f"{bodysense}", (0, 0, 255)
                self._draw_text_pil(draw, f"本体感：{btext}", (x, y), color=bcolor)
                y += line_h

            if referee:
                rname = _GESTURE_NAMES.get(referee, referee)
                self._draw_text_pil(draw, f"左手裁判：{rname}", (x, y), color=(200, 200, 200))

        self._draw_region(canvas, panel_x, panel_y, panel_w, panel_h, panel_text)

        # Prominent top banner for the round result so it can't be missed.
        if result and not countdown_label:
            rtext = _RESULT_TEXT.get(result, result)
            rsub = _RESULT_SUBTEXT.get(result, "")
            rcolor = _RESULT_COLORS.get(result, (0, 255, 0))
            banner_h = 90
            banner = canvas[:banner_h, :].copy()
            cv2.rectangle(banner, (0, 0), (w, banner_h), (0, 0, 0), -1)
            cv2.rectangle(banner, (0, 0), (w, banner_h), rcolor, 3)

            def result_text(pil, pw, ph):
                from PIL import ImageDraw

                draw = ImageDraw.Draw(pil)
                big = _FONT_CACHE.big_zh
                if self._has_zh and big:
                    self._draw_text_pil(draw, f"{rtext}", (pw // 2, ph // 2 - 10), font=big, color=rcolor, anchor="mm")
                else:
                    self._draw_text_pil(draw, f"{rtext}", (pw // 2, ph // 2 - 10), color=rcolor, anchor="mm")
                if rsub:
                    self._draw_text_pil(draw, rsub, (pw // 2, ph // 2 + 22), color=(200, 200, 200), anchor="mm")

            self._draw_region(banner, 0, 0, w, banner_h, result_text)
            canvas[:banner_h, :] = banner

        # Countdown in the center.
        if countdown_label:
            label_norm = countdown_label.lower().replace("!", "").replace("shoot", "shoot")
            display = _COUNTDOWN_TEXT.get(label_norm, countdown_label)
            overlay_w, overlay_h = 220, 180
            ox = (w - overlay_w) // 2
            oy = (h - overlay_h) // 2
            cv2.rectangle(canvas, (ox, oy), (ox + overlay_w, oy + overlay_h), (0, 0, 0), -1)
            cv2.rectangle(canvas, (ox, oy), (ox + overlay_w, oy + overlay_h), (0, 0, 255), 3)

            def countdown_text(pil, pw, ph):
                from PIL import ImageDraw

                draw = ImageDraw.Draw(pil)
                cx, cy = pw // 2, ph // 2
                big_font = _FONT_CACHE.big_zh
                if self._has_zh and big_font:
                    self._draw_text_pil(
                        draw,
                        display,
                        (cx, cy),
                        font=big_font,
                        color=(0, 0, 255),
                        anchor="mm",
                    )
                else:
                    self._draw_text_pil(
                        draw,
                        display,
                        (cx, cy),
                        color=(0, 0, 255),
                        anchor="mm",
                    )

            self._draw_region(canvas, ox, oy, overlay_w, overlay_h, countdown_text)

        # Waiting-for-start overlay: bottom banner so the camera feed stays visible.
        if waiting_for_start:
            self._blink += 1
            alpha = 1.0 if self._blink % 60 < 30 else 0.6
            overlay_h = 80
            overlay_y = h - overlay_h - 10
            overlay = canvas[overlay_y : overlay_y + overlay_h, :].copy()
            cv2.rectangle(overlay, (0, 0), (w, overlay_h), (0, 0, 0), -1)

            def waiting_text(pil, pw, ph):
                from PIL import ImageDraw

                draw = ImageDraw.Draw(pil)
                color = (int(255 * alpha), int(255 * alpha), int(255 * alpha))
                if self._has_zh and _FONT_CACHE.big_zh:
                    self._draw_text_pil(
                        draw,
                        "按空格开始游戏",
                        (pw // 2, ph // 2),
                        font=_FONT_CACHE.big_zh,
                        color=color,
                        anchor="mm",
                    )
                else:
                    self._draw_text_pil(
                        draw,
                        "Press SPACE",
                        (pw // 2, ph // 2),
                        color=color,
                        anchor="mm",
                    )

            self._draw_region(overlay, 0, 0, w, overlay_h, waiting_text)
            # Blend the banner back with slight transparency so the feed is still readable.
            cv2.addWeighted(overlay, 0.85, canvas[overlay_y : overlay_y + overlay_h, :], 0.15, 0,
                            canvas[overlay_y : overlay_y + overlay_h, :])

        return canvas

    def show(self, canvas: np.ndarray) -> None:
        import cv2

        cv2.imshow(self.window_name, canvas)

    def wait_key(self, delay_ms: int = 1) -> int:
        import cv2

        return cv2.waitKey(delay_ms) & 0xFF

    def close(self) -> None:
        import cv2

        try:
            cv2.destroyWindow(self.window_name)
        except cv2.error:
            pass
