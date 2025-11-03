import importlib as _importlib

from lambast.version import version as __version__

submodules = ["detection_methods", "generate_data", "utils"]

__all__ = submodules + ["__version__"]


def __getattr__(name):
    if name in submodules:
        return _importlib.import_module(f'lambast.{name}')
    else:
        try:
            return globals()[name]
        except KeyError:
            pass
