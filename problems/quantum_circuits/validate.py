import numpy as np
from helper import get_residual_num, reconstruct_from_binary_factors



def validate(
    payload: tuple[list[object], list[dict[str, np.ndarray]]],
    W_EXACT: float = 100.0,      # bonus for residual==0
    W_UNDER_SOTA: float = 500.0,   # extra bonus if rank < sota
    W_AT_SOTA: float = 300.0,       # blonus if rank == sota
) -> dict[str, float]:
    context, result = payload
    score = 0
    normed_steps = 0
    for con, res in zip(context, result):
        T_rec = reconstruct_from_binary_factors(res["factors"])
        residual = get_residual_num(con.tensor, T_rec)
        rank = res["factors"].shape[-1]
        score += 5 - residual/500
        if residual == 0:
            score += W_EXACT
            if rank == con.sota_rank:
                score += W_AT_SOTA
            if rank < con.sota_rank:
                score += W_UNDER_SOTA
        normed_steps += res["steps"] / 100_000
    return {"fitness": score, "normed_steps": normed_steps, "is_valid": 1}
