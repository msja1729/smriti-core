"""SMRITI: a memory firewall for AI agents. Start with `from smriti import Smriti`."""

__all__ = ["Smriti", "Context", "DENIED", "UNSEAL_FAILED"]


def __getattr__(name: str):
    # Lazy, so `import smriti.gateway...` does not load the public API's dependencies.
    if name in __all__:
        from smriti import api
        return getattr(api, name)
    raise AttributeError(f"module 'smriti' has no attribute {name!r}")
