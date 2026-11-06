"""Legacy import alias for archived experiments; use advice_audit for new work."""
import importlib
import importlib.abc
import importlib.util
import sys

class _AliasLoader(importlib.abc.Loader):
    def __init__(self, target):
        self.module = importlib.import_module(target)
        self.spec = self.module.__spec__
    def create_module(self, spec):
        return self.module
    def get_code(self, fullname):
        return self.spec.loader.get_code(self.module.__name__)
    def get_source(self, fullname):
        return self.spec.loader.get_source(self.module.__name__)
    def exec_module(self, module):
        module.__spec__ = self.spec

class _LegacyFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("daci."):
            destination = "advice_audit." + fullname[len("daci."):]
            loader = _AliasLoader(destination)
            return importlib.util.spec_from_loader(fullname, loader, origin=loader.spec.origin)
        return None

sys.meta_path.insert(0, _LegacyFinder())
