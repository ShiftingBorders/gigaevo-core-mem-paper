import numpy as np
import jax.numpy as jnp

def reconcstruct_from_tensoralpha(tensoralpha_res):
    dim = 3
    factors = [tensoralpha_res[0,:,:].T for r in range(dim)]
    letters = ''.join(chr(97 + i) for i in range(dim))
    spec = ','.join(f"{chr(97 + i)}r" for i in range(dim)) + f"->{letters}r"
    and_per_r = jnp.einsum(spec, *factors, optimize=True).astype(jnp.uint8)
    T = (jnp.sum(and_per_r, axis=-1) % 2).astype(jnp.uint8)
    return T


class Data:
    def __init__(self, name, tensor, rank):
        self.name = name
        self.data = tensor
        self.m = rank

def build_context() -> list[Data]:
    l = []
    name = "binary_addition.npz"
    name = "benchmarks_gadgets.npz"
    with np.load(name) as f:
        for file in f.files:
            rank = f[file].shape[1]
            if rank < 200:
                l.append(Data(file, reconcstruct_from_tensoralpha(f[file]), rank))
    return l