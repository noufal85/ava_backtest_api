"""Custom strategies — auto-discovered and registered by strategy builder.

Any .py file dropped into this directory is automatically imported,
triggering the @register decorator without any manual __init__ edits.
"""
import importlib
import pkgutil
import src.strategies.custom as _pkg

for _mod_info in pkgutil.iter_modules(_pkg.__path__):
    importlib.import_module(f"src.strategies.custom.{_mod_info.name}")
