"""Checkpoint reconstruction must survive loading another seed and reject corruption."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from oeaprfm.checkpoints import save_model, load_model
from oeaprfm.features import function_space
from oeaprfm.features.reconstruction import OEField1D

jax.config.update("jax_enable_x64", True)


def test_roundtrip_independent_seeds(tmp_path):
    predictors=[];expected=[]
    x=jnp.array([[.1],[.7]]);v=jnp.array([[.6],[-.3]])
    for seed in (7,17):
        function_space.seedXV=seed
        model=OEField1D(domain={'x':(0.,1.),'v':(0.,1.)},strides={'x':1.,'v':1.},Jn={'j':4,'r':4},scale=1.,init_rng=jax.random.key(seed),kn=.001,activation=jnp.tanh,coefficients=jnp.arange(8.))
        params=model.init(jax.random.key(seed),x[0],v[0])
        expected.append(np.asarray(jax.vmap(lambda a,b:model.apply(params,a,b))(x,v)))
        directory=tmp_path/str(seed)
        save_model(directory,model,params,dict(problem='p1',seed=seed,epsilon=.001))
        predictors.append(load_model(directory))
    for predictor,values in zip(predictors,expected):
        np.testing.assert_allclose(predictor(x[:,0],v[:,0]),values,rtol=1e-12,atol=1e-12)
    path=tmp_path/'7'/'model.npz'
    with np.load(path) as d:arrays={k:d[k] for k in d.files}
    key=next(k for k in arrays if k.startswith('feature/'))
    arrays[key]=arrays[key]+1
    np.savez_compressed(path,**arrays)
    with pytest.raises(ValueError,match='feature state differs'):
        load_model(path.parent)


def test_square_hole_matches_apnn():
    from oeaprfm.features.sampling import Sample2D
    from oeaprfm.features.square_hole import restrict_square_hole
    from oeaprfm.problems.p5_square_hole_2d import get_config
    c=get_config(.001)
    sample=Sample2D(dict(c.mesh.domain),{'interior':(12,12,8),'boundary':(8,8,8)},'uniform')
    restrict_square_hole(sample)
    x,y,_=sample.pts_int
    assert np.all((np.abs(x)>1/3)|(np.abs(y)>1/3))
    x,y,normal=map(np.asarray,sample.pts_hole)
    # Moving along every inflow direction leaves the excluded square.
    xx=x+1e-5*np.cos(normal);yy=y+1e-5*np.sin(normal)
    assert np.all((np.abs(xx)>1/3)|(np.abs(yy)>1/3))
    np.testing.assert_allclose(c.model.bdy_cond.f_circle(x,y),np.exp(-x-y))
