def import_model_by_backend(tensorflow_cmd, pytorch_cmd):
    import sys
    for _backend in sys.modules["external"].backend:
        if _backend == "tensorflow":
            exec(tensorflow_cmd)
        elif _backend == "pytorch":
            exec(pytorch_cmd)
            break


import sys
for _backend in sys.modules["external"].backend:
    if _backend == "tensorflow":
        pass
    elif _backend == "pytorch":
        from .lightgcn.LightGCN import LightGCN
        from .dgcf.DGCF import DGCF
        from .svd_gcn import SVDGCN
        from .ultragcn import UltraGCN
