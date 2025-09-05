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

    @staticmethod
    def _srgb_to_linear(c_0_255: int | float) -> float:
        """sRGB(0–255) -> 線形化(0–1)"""
        cs = c_0_255 / 255.0
        # WCAG 2.x のしきい値 0.03928
        return cs / 12.92 if cs <= 0.03928 else ((cs + 0.055) / 1.055) ** 2.4

    def relative_luminance(self) -> float:
        """相対輝度 (0–1), WCAG 定義"""
        R = self._srgb_to_linear(self.r)
        G = self._srgb_to_linear(self.g)
        B = self._srgb_to_linear(self.b)
        return 0.2126 * R + 0.7152 * G + 0.0722 * B

    def pick_text_color_from(self, c1: Color, c2: Color) -> Color:
        """
        この色を背景としたときに候補 c1, c2 のどちらが可読性が高いかをコントラスト比で判定して返す.
        """
        Lbg = self.relative_luminance()

        def luminance(c: Color) -> float:
            return c.relative_luminance()

        def contrast(L1: float, L2: float) -> float:
            Lmax, Lmin = (L1, L2) if L1 >= L2 else (L2, L1)
            return (Lmax + 0.05) / (Lmin + 0.05)

        cr1 = contrast(Lbg, luminance(c1))
        cr2 = contrast(Lbg, luminance(c2))

        return c1 if cr1 >= cr2 else c2


# カラーパレット
BLACK = Color.from_hex("#393636")
WHITE = Color.from_hex("#F7F7F7")
GRAY = Color.from_hex("#A6A6A6")
RED = Color.from_hex("#ED6517")
GREEN = Color.from_hex("#63A375")
BLUE = Color.from_hex("#295FA5")

RBMIX = Color.average(RED, BLUE)
BGCOLOR = WHITE
NODE = WHITE
EDGE = GRAY
BORDER = BLACK
