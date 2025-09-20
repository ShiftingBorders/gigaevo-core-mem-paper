from __future__ import annotations
import jax
import jax.numpy as jnp
from jax.nn import initializers, sigmoid
import optax
from dataclasses import dataclass
from typing import Tuple, List, Any, Dict
from helper import generate_symmetric_bool_tensor

def entrypoint(context: List[jnp.ndarray]) -> List[int]:
    def smooth_reconstruction(factors: jnp.ndarray, dim: int) -> jnp.ndarray:
        letters = ''.join(chr(97 + i) for i in range(dim))
        spec = ','.join(f"{chr(97+i)}r" for i in range(dim)) + "->" + letters + "r"
        p = sigmoid(factors)
        and_per_r = jnp.einsum(spec, *([p] * dim))
        prod_term = jnp.prod(1.0 - 2.0 * and_per_r, axis=-1)
        return 0.5 * (1.0 - prod_term)

    def get_optimizer(lr, **kwargs):
        return optax.adam(lr)
    
    def make_step(target: jnp.ndarray, lr: float):
        opt = get_optimizer(lr)

        def loss_fn(f: jnp.ndarray) -> jnp.ndarray:
            P = smooth_reconstruction(f, dim=target.ndim)
            return bce_loss(target.astype(jnp.float32), P)

        @jax.jit
        def step(f: jnp.ndarray, state: optax.OptState) -> Tuple[jnp.ndarray, optax.OptState]:
            grads = jax.grad(loss_fn)(f)
            updates, state = opt.update(grads, state, f)
            f = optax.apply_updates(f, updates)
            return f, state

        return opt, step
    
    def bce_loss(target: jnp.ndarray, P: jnp.ndarray) -> jnp.ndarray:
        eps = 1e-6
        return jnp.mean(-target * jnp.log(P + eps) - (1.0 - target) * jnp.log(1.0 - P + eps))

    def reconstruct_gf2_binary(factors: jnp.ndarray, dim: int) -> jnp.ndarray:
        n, R = factors.shape
        letters = ''.join(chr(97 + i) for i in range(dim))
        spec = ','.join(f"{chr(97+i)}r" for i in range(dim)) + "->" + letters + "r"
        b = (sigmoid(factors) >= 0.5).astype(jnp.uint8)
        and_per_r = jnp.einsum(spec, *([b] * dim)).astype(jnp.uint8)
        return (jnp.sum(and_per_r, axis=-1) & jnp.uint8(1)).astype(jnp.uint8)

    def search_min_rank(T: jnp.ndarray,
                        r_max: int | None = None,
                        per_rank_steps: int = 1000,
                        lr: float = 3e-2,
                        restarts: int = 1,
                        seed: int = 0) -> Dict[str, Any]:
        n = T.shape[0]
        dim = T.ndim
        cap = r_max or min(n, 32)
        base_key = jax.random.PRNGKey(seed)
        best = {"rank": None, "residual": T.size, "factors": None}

        for r in range(1, cap + 1):
            for s in range(restarts):
                key = jax.random.fold_in(base_key, (r << 8) + s)
                F = jax.random.normal(key, (n, r)) * 0.01
                opt, step = make_step(T, lr)
                state = opt.init(F)
                for _ in range(per_rank_steps):
                    F, state = step(F, state)

                norms = jnp.sqrt(jnp.sum(jnp.square(sigmoid(F) - 0.5), axis=0))
                keep = norms > 1e-3
                if keep.sum() == 0:
                    continue
                if keep.sum() < r:
                    F = F[:, keep]
                    r = int(F.shape[1])

                T_hat = reconstruct_gf2_binary(F, dim)
                residual = int(jnp.sum(T_hat ^ T.astype(jnp.uint8)))

                if residual < best["residual"] or (residual == best["residual"] and (best["rank"] is None or r < best["rank"])):
                    best = {"rank": r, "residual": residual, "factors": F}

                if residual == 0:
                    return {"rank": r, "residual": 0, "factors": F}
        return best

    results: List[Dict[str, Any]] = []
    for idx, T in enumerate(context):
        res = search_min_rank(T.data, r_max=T.m, per_rank_steps=800, lr=3e-2, restarts=2, seed=idx)
        results.append({"rank": res["rank"], "residual": res["residual"]})

    return results
