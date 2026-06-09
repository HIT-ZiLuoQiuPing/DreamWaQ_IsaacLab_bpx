"""Robot asset configurations."""

__all__ = ["BPX_CFG", "BPX_PLAY_CFG"]


def __getattr__(name: str):
    if name in __all__:
        from .bpx import BPX_CFG, BPX_PLAY_CFG

        return {"BPX_CFG": BPX_CFG, "BPX_PLAY_CFG": BPX_PLAY_CFG}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
