import math


def lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Cosine learning rate schedule with linear warmup.

    (Warm-up)        t < T_w:  alpha_t = (t / T_w) * alpha_max
    (Cosine anneal)  T_w <= t <= T_c:  alpha_t = alpha_min + 0.5 * (1 + cos((t - T_w) / (T_c - T_w) * pi)) * (alpha_max - alpha_min)
    (Post-anneal)    t > T_c:  alpha_t = alpha_min
    """
    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate
    elif it <= cosine_cycle_iters:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 0.5 * (1 + math.cos(progress * math.pi)) * (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate
