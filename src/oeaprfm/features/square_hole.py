"""Restrict collocation to the square-hole domain and expose its inflow normals."""
import jax.numpy as jnp


def restrict_square_hole(sample, half_width=1/3):
    x,y,t = sample.pts_int
    keep = ((jnp.abs(x)>half_width)|(jnp.abs(y)>half_width)).reshape(-1)
    sample.pts_int = tuple(a[keep] for a in (x,y,t))
    n = max(sample.collocation_sizes['boundary'][:2])
    edge = jnp.linspace(-half_width,half_width,n+2)[1:-1,None]
    fixed = jnp.full_like(edge,half_width)
    # Directions point from the void into the physical domain.
    sample.pts_hole = (
        jnp.concatenate((-fixed,fixed,edge,edge)),
        jnp.concatenate((edge,edge,-fixed,fixed)),
        jnp.concatenate(tuple(jnp.full_like(edge,a) for a in (jnp.pi,0,1.5*jnp.pi,0.5*jnp.pi))),
    )
