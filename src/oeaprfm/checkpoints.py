"""Save fitted RF coefficients, feature parameters and reconstruction settings."""

import importlib
import json
from pathlib import Path
import threading

import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict
from jax import random

_LOCK = threading.RLock()


def _seed(seed):
    from oeaprfm.features import function_space, function_space_v2
    function_space.seedXV = seed
    function_space.seedX = seed
    function_space_v2.seedX = seed


def _features(constructor, variables):
    bound = constructor.bind(variables)
    # Access forces setup of the reconstruction and its feature spaces.
    names = ('j1', 'r1', 'j2', 'r2') if 'quadrant' in type(constructor).__module__ else ('j', 'r')
    arrays = {}
    for name in names:
        feature = getattr(bound, '_model_' + name)
        for i, params in enumerate(feature._params_list):
            for key, value in flatten_dict(params).items():
                arrays['feature/' + name + '/' + str(i) + '/' + '/'.join(key)] = np.asarray(value)
    return arrays


def save_model(directory, constructor, variables, record):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = dict(
        format='oe-aprfm-v1', problem=record['problem'], epsilon=record['epsilon'],
        seed=record['seed'], constructor=type(constructor).__module__ + ':' + type(constructor).__name__,
        domain=constructor.domain, strides=constructor.strides,
        Jn=constructor.Jn, scale=constructor.scale, activation='tanh',
        dtype=str(np.asarray(constructor.coefficients).dtype),
        jax_version=jax.__version__, problem_definition=record.get('problem_definition',record['problem']),
    )
    arrays = _features(constructor, variables)
    arrays['coefficients'] = np.asarray(constructor.coefficients)
    np.savez_compressed(directory / 'model.npz', **arrays)
    (directory / 'config.json').write_text(json.dumps(config, indent=2, default=lambda value: value.to_dict() if hasattr(value, "to_dict") else float(value))+'\n')
    (directory / 'metrics.json').write_text(json.dumps(record, indent=2)+'\n')


def load_model(directory):
    """Return a batched predictor; verify regenerated features against saved weights."""
    directory = Path(directory)
    config = json.loads((directory/'config.json').read_text())
    if config['format'] != 'oe-aprfm-v1':
        raise ValueError('Unsupported RF checkpoint format')
    with np.load(directory/'model.npz', allow_pickle=False) as data:
        arrays = {key:data[key] for key in data.files}
    if arrays['coefficients'].dtype == np.float64:
        jax.config.update('jax_enable_x64', True)
    module, name = config['constructor'].split(':')
    if not module.startswith('oeaprfm.features.'):
        raise ValueError('Unsupported reconstruction module')
    cls = getattr(importlib.import_module(module),name)
    dimension = 3 if 'theta' in config['domain'] else 2
    with _LOCK:
        _seed(config['seed'])
        model = cls(domain=config['domain'],strides=config['strides'],Jn=config['Jn'],
                    scale=config['scale'],init_rng=random.key(config['seed']),kn=config['epsilon'],
                    activation=jnp.tanh,coefficients=jnp.asarray(arrays['coefficients']))
        dummy = tuple(jnp.zeros((1,)) for _ in range(dimension))
        variables = model.init(random.key(config['seed']),*dummy)
        current = _features(model,variables)
        if set(current) != set(arrays)-{'coefficients'}:
            raise ValueError('Saved feature structure differs from reconstruction')
        for key,value in current.items():
            if not np.array_equal(value,arrays[key]):
                raise ValueError('Random feature state differs: '+key)
        # Compile a scalar kernel now, while the checkpoint seed is active.
        kernel = jax.jit(lambda *coords:model.apply(variables,*coords))
        kernel(*dummy).block_until_ready()
    def predict(*coords):
        values = np.broadcast_arrays(*[np.asarray(c) for c in coords])
        shape = values[0].shape
        flat = [jnp.asarray(c.reshape(-1,1)) for c in values]
        # vmap traces the already-compiled scalar kernel with its captured weights.
        with _LOCK:
            _seed(config["seed"])
            return np.asarray(jax.vmap(kernel)(*flat)).reshape(shape)
    return predict
