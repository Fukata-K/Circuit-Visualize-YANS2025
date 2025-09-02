from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Color:
    r: int | float
    g: int | float
    b: int | float

    def __post_init__(self):
        object.__setattr__(self, "r", self._process(self.r))
        object.__setattr__(self, "g", self._process(self.g))
        object.__setattr__(self, "b", self._process(self.b))

    @staticmethod
    def _process(value: int | float) -> int:
        # 0–255 にクリップ
        return max(0, min(255, int(round(float(value)))))

    def to_hex(self) -> str:
        return f"#{self.r:02x}{self.g:02x}{self.b:02x}"

    @classmethod
    def from_hex(cls, hex_str: str) -> Color:
        s = hex_str.strip().lstrip("#")
        if len(s) == 3:  # #abc -> #aabbcc
            s = "".join(ch * 2 for ch in s)
        if len(s) != 6 or any(c not in "0123456789abcdefABCDEF" for c in s):
            raise ValueError(f"Invalid hex color: {hex_str!r}")
        r, g, b = (int(s[i : i + 2], 16) for i in (0, 2, 4))
        return cls(r, g, b)

    @classmethod
    def average(cls, c1: Color, c2: Color) -> Color:
        """2色の単純平均"""
        return cls((c1.r + c2.r) / 2, (c1.g + c2.g) / 2, (c1.b + c2.b) / 2)

    @classmethod
    def add(cls, c1: Color, c2: Color) -> Color:
        """チャネル毎に加算 (255 超えはクリップ)"""
        return cls(c1.r + c2.r, c1.g + c2.g, c1.b + c2.b)


# カラーパレット
BLACK = Color.from_hex("#262626")
WHITE = Color.from_hex("#F7F7F7")
GRAY = Color.from_hex("#A6A6A6")
RED = Color.from_hex("#ED6517")
GREEN = Color.from_hex("#63A375")
BLUE = Color.from_hex("#295FA5")

RBMIX = Color.average(RED, BLUE)
BGCOLOR = BLACK
FONT = BLACK
NODE = WHITE
EDGE = WHITE
BORDER = WHITE
