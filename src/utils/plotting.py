from __future__ import annotations

from functools import lru_cache
from typing import Any


_CJK_FONT_CANDIDATES = (
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "Source Han Sans SC",
    "Source Han Sans CN",
    "WenQuanYi Zen Hei",
    "Microsoft YaHei",
    "SimHei",
    "PingFang SC",
    "Heiti SC",
    "Sarasa Gothic SC",
)

_SANS_SERIF_FALLBACKS = (
    "DejaVu Sans",
    "Ubuntu",
    "Arial",
)


@lru_cache(maxsize=1)
def _detect_cjk_font_name() -> str | None:
    try:
        from matplotlib import font_manager

        available = {entry.name for entry in font_manager.fontManager.ttflist}
    except Exception:
        return None

    for candidate in _CJK_FONT_CANDIDATES:
        if candidate in available:
            return candidate
    return None


def configure_report_matplotlib(matplotlib_module: Any) -> bool:
    """为实验报告图表设置稳定的字体与负号渲染策略。"""
    cjk_font = _detect_cjk_font_name()

    matplotlib_module.rcParams["axes.unicode_minus"] = False
    matplotlib_module.rcParams["font.family"] = "sans-serif"
    if cjk_font:
        matplotlib_module.rcParams["font.sans-serif"] = [cjk_font, *_SANS_SERIF_FALLBACKS]
        return True

    matplotlib_module.rcParams["font.sans-serif"] = list(_SANS_SERIF_FALLBACKS)
    return False


def pick_plot_text(chinese_text: str, ascii_text: str, *, force_ascii: bool = False) -> str:
    """在无中文字体环境下自动回退到 ASCII 文案，避免图表出现方块乱码。"""
    if force_ascii:
        return ascii_text
    return chinese_text if _detect_cjk_font_name() else ascii_text
