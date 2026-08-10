"""Tilelang IR analysis & visitors."""
from . import _ffi_api
from tvm import IRModule
from tvm.tir import PrimFunc

def CheckStatic(mod: IRModule | PrimFunc) -> bool:
    """CheckStatic

    Returns
    -------
    bool
        Whether the function is completely static.
    """
    if isinstance(mod, IRModule):
        items = mod.functions_items()
        assert len(items) == 1, "Temporarily only support single function module"
        return _ffi_api.CheckStatic(items[0][1])
    return _ffi_api.CheckStatic(mod)

def DumpTileGraph(mod: IRModule | PrimFunc):
    if isinstance(mod, IRModule):
        items = mod.functions_items()
        assert len(items) == 1, "Temporarily only support single function module"
        return _ffi_api.DumpTileGraph(items[0][1])
    return _ffi_api.DumpTileGraph(mod)

