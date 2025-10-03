import numpy as np
import jax.numpy as jnp

def reconcstruct_from_tensoralpha(tensoralpha_res):
    dim = 3
    factors = [tensoralpha_res[0,:,:].T for r in range(dim)]
    letters = ''.join(chr(97 + i) for i in range(dim))
    spec = ','.join(f"{chr(97 + i)}r" for i in range(dim)) + f"->{letters}r"
    and_per_r = jnp.einsum(spec, *factors, optimize=True).astype(jnp.uint8)
    T = (jnp.sum(and_per_r, axis=-1) & jnp.uint8(1)).astype(jnp.uint8)
    return T


class Data:
    def __init__(self, name, tensor, rank):
        self.name = name
        self.tensor = tensor
        self.sota_rank = rank

def build_context() -> list[Data]:
    l = []
    name = "binary_addition.npz"
    name = "data/benchmarks_gadgets.npz"
    with np.load(name) as f:
        for file in f.files:
            rank = f[file].shape[1]
            n = f[file].shape[2]
            if rank*n < 500:
                l.append(Data(file, reconcstruct_from_tensoralpha(f[file]), rank))
    print(len(l))
    return l[:]