"""
LAMBAST
=======
Los Alamos Model Bias Assessment and Statistical Toolkit

Subpackages
-----------
detection_methods
    Library of methods for out of distribution or change point detection.
generate_data
    Library of methods to generate distributions
utils
    General utility functions
"""
import importlib as _importlib

from lambast.version import version as __version__

submodules = ["detection_methods", "generate_data", "utils"]
optional = ["__qualname__", "__date__", "__author__", "__credits__"]

__all__ = submodules + ["__version__"]


def __getattr__(name):
    if name in submodules:
        return _importlib.import_module(f'lambast.{name}')
    else:
        try:
            return globals()[name]
        except KeyError:
            if name in optional:
                pass
            else:
                raise ModuleNotFoundError(
                    f"Module {name} does not exist in LAMBAST")
