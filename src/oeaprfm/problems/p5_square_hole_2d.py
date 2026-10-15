"""P5 manufactured square-hole problem matching the OE-APNN checkpoints."""
from oeaprfm.problems.p3_manufactured_2d import get_config as square_config


def get_config(epsilon=1.0):
    config = square_config(epsilon)
    config.problem = 'p5'
    config.model.bdy_cond.f_circle = lambda x,y: config.model.exact_solution(x,y,0.0)
    return config
