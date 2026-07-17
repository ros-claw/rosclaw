"""Memory 2.0 bilingual tokenizer (§6.4).

Rules:

* Chinese text is segmented with jieba (falls back to bigrams when jieba is
  unavailable);
* English is word-tokenized;
* Error codes are preserved verbatim: ``USB_TIMEOUT``, ``EIO``, ``-110``;
* Device names are preserved: ``RH56``, ``D435i``, ``Jetson``, ``CH340``,
  ``FTDI``;
* Code symbols are preserved: ``snake_case``, ``dotted.path``.
"""

from __future__ import annotations

import re

try:
    import jieba

    _HAS_JIEBA = True
except ImportError:  # pragma: no cover - exercised on minimal installs
    jieba = None  # type: ignore[assignment]
    _HAS_JIEBA = False

# Tokens that must survive tokenization verbatim (case-insensitive lookup,
# emitted in their canonical form).
_PROTECTED_TOKENS = {
    "usb_timeout": "USB_TIMEOUT",
    "eio": "EIO",
    "-110": "-110",
    "enoent": "ENOENT",
    "rh56": "RH56",
    "d435i": "D435i",
    "d405": "D405",
    "jetson": "Jetson",
    "ch340": "CH340",
    "ftdi": "FTDI",
    "mcap": "MCAP",
    "modbus": "Modbus",
    "rs485": "RS485",
}

# Code symbols: snake_case, dotted.path, CamelCase.
_CODE_SYMBOL_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+|[A-Za-z_][A-Za-z0-9_]*_[A-Za-z0-9_]+"
)
# Numeric error codes like -110, -5.
_ERROR_CODE_RE = re.compile(r"-\d{1,4}\b")
# Chinese runs.
_CJK_RE = re.compile(r"[一-鿿]+")
# English words.
_WORD_RE = re.compile(r"[A-Za-z0-9]+")

# Cross-lingual bridge (§6.4): robotics-domain zh ↔ en synonyms so a Chinese
# query recalls English documents and vice versa.  Kept deliberately small and
# auditable — generic translation is out of scope.
_CROSS_LINGUAL_MAP: dict[str, list[str]] = {
    "剪刀": ["scissors"],
    "石头": ["rock"],
    "布": ["paper"],
    "失败": ["failed", "failure"],
    "成功": ["success", "succeeded"],
    "过流": ["overcurrent", "current"],
    "电流": ["current"],
    "温度": ["temperature"],
    "温升": ["thermal", "temperature"],
    "串口": ["serial", "uart"],
    "相机": ["camera"],
    "超时": ["timeout"],
    "手指": ["finger"],
    "拇指": ["thumb"],
    "食指": ["index"],
    "小指": ["little"],
    "无帧": ["no_frames", "frames"],
    "急停": ["estop", "emergency"],
    "碰撞": ["collision"],
    "恢复": ["recovery"],
    "干预": ["intervention"],
    "关节": ["joint"],
    "到位": ["reached"],
    "手势": ["gesture"],
    "电机": ["motor"],
    "异常": ["error", "abnormal"],
    "左手": ["left"],
    "右手": ["right"],
    "scissors": ["剪刀"],
    "rock": ["石头", "fist"],
    "paper": ["布"],
    "failed": ["失败"],
    "failure": ["失败"],
    "overcurrent": ["过流", "电流"],
    "current": ["电流"],
    "temperature": ["温度"],
    "serial": ["串口"],
    "camera": ["相机"],
    "timeout": ["超时"],
    "finger": ["手指"],
    "thumb": ["拇指"],
    "index": ["食指"],
    "gesture": ["手势"],
    "joint": ["关节"],
    "recovery": ["恢复"],
    "motor": ["电机"],
    "left": ["左手"],
    "right": ["右手"],
}


def expand_tokens(tokens: list[str]) -> list[str]:
    """Return tokens plus their cross-lingual equivalents (deduped, order-stable)."""
    expanded: list[str] = []
    seen: set[str] = set()
    for token in tokens + [t for t in tokens for t in _CROSS_LINGUAL_MAP.get(t, [])]:
        if token not in seen:
            seen.add(token)
            expanded.append(token)
    return expanded


def _protect(text: str) -> str:
    """Lowercase a token for protected-set lookup."""
    return text.lower()


def tokenize(text: str) -> list[str]:
    """Tokenize mixed Chinese/English text into search terms."""
    if not text:
        return []
    tokens: list[str] = []

    # 1. Error codes first (they contain a leading dash that splits words).
    for match in _ERROR_CODE_RE.finditer(text):
        tokens.append(match.group(0))
    text = _ERROR_CODE_RE.sub(" ", text)

    # 2. Code symbols (snake_case, dotted.path) — keep whole AND parts.
    for match in _CODE_SYMBOL_RE.finditer(text):
        symbol = match.group(0)
        canonical = _PROTECTED_TOKENS.get(_protect(symbol), symbol)
        tokens.append(canonical)
        for part in re.split(r"[._]", symbol):
            if part:
                tokens.append(_PROTECTED_TOKENS.get(_protect(part), part.lower()))
    text = _CODE_SYMBOL_RE.sub(" ", text)

    # 3. Protected device/error tokens that remain.
    for word in _WORD_RE.findall(text):
        canonical = _PROTECTED_TOKENS.get(_protect(word))
        if canonical is not None:
            tokens.append(canonical)

    # 4. Chinese segmentation.
    for cjk_run in _CJK_RE.findall(text):
        if _HAS_JIEBA:
            tokens.extend(term for term in jieba.lcut(cjk_run) if term.strip())
        else:
            # Fallback: overlapping bigrams (single char for len-1 runs).
            if len(cjk_run) == 1:
                tokens.append(cjk_run)
            else:
                tokens.extend(cjk_run[i : i + 2] for i in range(len(cjk_run) - 1))

    # 5. Remaining English words (lowercased, skip stop-ish short tokens).
    for word in _WORD_RE.findall(text):
        if _protect(word) in _PROTECTED_TOKENS:
            continue  # already emitted in canonical form
        if len(word) > 2:
            tokens.append(word.lower())

    return tokens


def token_set(text: str, *, expand: bool = True) -> set[str]:
    """Unique token set for overlap scoring (cross-lingual by default)."""
    tokens = tokenize(text)
    if expand:
        tokens = expand_tokens(tokens)
    return set(tokens)
