"""Validate the five nonredundant JSC experiment definitions."""

import importlib

import jax.numpy as jnp
import numpy as np

MODULES = [
    "configuration.p1_manufactured_1d",
    "configuration.p2_heterogeneous_1d",
    "configuration.p3_manufactured_2d",
    "configuration.p4_circular_hole_2d",
    "configuration.p5_heterogeneous_2d",
]


def configs(epsilon=1.0):
    return [
        importlib.import_module(name).get_config(epsilon) for name in MODULES
    ]


def test_shared_protocol_is_frozen_and_independent():
    loaded = configs()
    for index, config in enumerate(loaded):
        model, protocol = config.model, config.protocol
        assert tuple(protocol.seeds) == (11, 23, 37)
        assert protocol.precision == "float64"
        assert protocol.evaluation_is_independent
        assert protocol.row_scaling == "unit_l2"
        assert protocol.rcond == 1.0e-12
        assert len(protocol.epsilon_values) == (3 if index < 3 else 2)
        expected_features = 64 if index < 2 or index == 4 else 128
        assert set(model.Jn.values()) == {expected_features}
        np.testing.assert_allclose(
            model.activation(jnp.array([-1.0, 1.0])),
            jnp.tanh(jnp.array([-1.0, 1.0])),
        )


def test_problem_physics():
    p1, p2, p3, p4, p5 = configs(1.0e-3)
    assert p1.model.source(0.2, 0.4) == -400.0
    np.testing.assert_allclose(p1.model.exact_solution(0.2, 0.4), 0.8)
    expected = 0.01 + (jnp.tanh(6.5 - 11 * 0.5) + jnp.tanh(11 * 0.5 - 4.5)) / 2
    np.testing.assert_allclose(p2.model.coeff.scattering(0.5), expected)
    np.testing.assert_allclose(p3.model.exact_solution(0.0, 0.0, 0.3), 1.0)
    np.testing.assert_allclose(p4.model.source(0.0, 0.0, 0.3), 0.0)
    np.testing.assert_allclose(p4.model.bdy_cond.f_circle(0.5, 0.0), 0.8)
    np.testing.assert_allclose(p5.model.coeff.scattering(0.2, 0.0), 1.0)
    np.testing.assert_allclose(p5.model.coeff.scattering(0.2, 1.0), 2.0)
    np.testing.assert_allclose(p5.model.coeff.absorption(0.2, 0.7), 0.1)
    assert p5.protocol.reference == "oe_sn_refined"
    np.testing.assert_allclose(p5.model.source(0.5, 0.5, 1.2), 0.0)
    np.testing.assert_allclose(p5.model.bdy_cond.f_l(0.5), 1.1)
    np.testing.assert_allclose(p5.model.bdy_cond.f_b(0.5), 1.0)
    np.testing.assert_allclose(p5.model.bdy_cond.f_t(0.5), 1.2)
    np.testing.assert_allclose(
        p5.model.source(0.30, 0.50, 1.2),
        configs(1.0)[-1].model.source(0.30, 0.50, 1.2),
    )


def test_epsilon_is_applied_without_changing_discretization():
    module = importlib.import_module(MODULES[0])
    configs_by_epsilon = [module.get_config(value) for value in (1.0, 1.0e-16)]
    assert [config.model.knudsen_number for config in configs_by_epsilon] == [
        1.0,
        1.0e-16,
    ]
    assert configs_by_epsilon[0].model.Jn == configs_by_epsilon[1].model.Jn
    assert (
        configs_by_epsilon[0].model.collocation_sizes
        == configs_by_epsilon[1].model.collocation_sizes
    )


def test_p5_uses_the_convergent_reference_sweep():
    config = importlib.import_module(MODULES[-1]).get_config()
    assert tuple(config.protocol.epsilon_values) == (1.0, 1.0e-3)
    np.testing.assert_allclose(config.mesh.strides.x, 0.5)
    np.testing.assert_allclose(config.mesh.strides.y, 0.5)
    np.testing.assert_allclose(config.mesh.strides.theta, 0.5 * jnp.pi)
    assert tuple(config.model.collocation_sizes.interior) == (32, 32, 16)
    assert config.model.num_quads == 16
    assert config.model.num_unknowns["j/r"] == 1024
    assert tuple(config.problem_data.angular_components) == (
        "j1",
        "r1",
        "j2",
        "r2",
    )
    assert not config.problem_data.adaptive_sampling


def test_p5_v2_uses_the_same_partition_basis_for_assembly_and_reconstruction():
    from constraints import continuous_2d_odd_even_v2 as constraint
    from modules import function_space_v2

    # A mismatched POU changes the represented function once P5 has multiple
    # spatial patches, even if the coefficient vector is unchanged.
    x = jnp.array([0.9])
    np.testing.assert_allclose(constraint.psi(x), function_space_v2.psi(x))
    np.testing.assert_allclose(
        constraint.dpsi(x), -jnp.pi * jnp.cos(2 * jnp.pi * x)
    )
