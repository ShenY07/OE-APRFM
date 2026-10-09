import time


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(
            f"The executed time of {func.__name__}: {end_time - start_time:.2e} seconds."
        )
        return result

    return wrapper


# @timer
# def map_psi_fn_b(x):
#     def psi_fn_b(x):
#         funcs = [
#             lambda x: jnp.ones_like(x),
#             lambda x: 0.5 * (1.0 - jnp.sin(2.0 * jnp.pi * jnp.abs(x))),
#             lambda x: jnp.zeros_like(x),
#         ]
#         value = jnp.where(
#             jnp.abs(x) < 3 / 4,
#             funcs[0](x),
#             jnp.where(
#                 jnp.abs(x) > 5 / 4,
#                 funcs[-1](x),
#                 funcs[1](x),
#             ),
#         )
#         return value

#     return jax.vmap(psi_fn_b)(x)
