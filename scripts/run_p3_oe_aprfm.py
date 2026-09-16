"""Run one P3 OE-APRFM realization and measure f/rho errors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault("JAX_ENABLE_X64", "true")
import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from jax import random, vmap

import modules.function_space as function_space
import modules.function_space_v2 as function_space_v2
from configuration.p3_manufactured_2d import get_config
from configuration.p4_circular_hole_2d import get_config as get_p4_config
from configuration.p5_heterogeneous_2d import get_config as get_p5_config
from constraints.continuous_2d_odd_even import (
    OddEvenDecompositionPointwiseBoundaryConstraint2D,
    OddEvenDecompositionPointwiseInteriorConstraint2D,
)
from constraints.continuous_2d_odd_even_v2 import (
    OddEvenDecompositionPointwiseBoundaryConstraint2D as V2BoundaryConstraint2D,
    OddEvenDecompositionPointwiseInteriorConstraint2D as V2InteriorConstraint2D,
)
from modules.collocation_sampling import Sample2D
from modules.collocation_sampling_v2_alternative import SampleHoleCircle
from modules.solution_construction import OddEvenDecompositionConstructor2D
from modules.solution_construction_v2 import OddEvenDecompositionConstructor2D as V2Constructor2D
from solver.streaming_least_squares import StreamingQRLeastSquares

jax.config.update("jax_enable_x64", True)


def run(epsilon, seed, output_dir, problem="p3", *, angular_representation="four_component", p5_constant_inflow=False, partitions=None, features=None, scale=1.0, rcond=1e-12, block_weights=None, collocation=None, tag=""):
    config_functions = {"p3": get_config, "p4": get_p4_config, "p5": get_p5_config}
    config = get_p5_config(epsilon, constant_inflow=p5_constant_inflow) if problem == "p5" else config_functions[problem](epsilon); model = config.model
    if partitions is None:
        partitions = (2, 2, 1) if problem == "p5" else (1, 1, 1)
    if angular_representation not in ("legacy_two_component", "four_component"):
        raise ValueError(f"unknown angular representation: {angular_representation}")
    use_v2 = angular_representation == "four_component"
    if features is None:
        # Match the frozen P3/P4 table budget: both representations have 512
        # linear unknowns on one spatial patch (4*128 == 2*256).
        features = 128 if use_v2 else 256
    if collocation is None and problem == "p5":
        collocation = (32, 32, 16)
    if collocation is not None:
        model.collocation_sizes={"interior":tuple(collocation),"boundary":tuple(collocation)}
    patch_count=int(np.prod(partitions)); model.Jn={"j":features,"r":features,"f":features}; model.Mp={"j":patch_count,"r":patch_count,"f":patch_count}; model.num_unknowns={"j/r":2*features*patch_count,"f":features*patch_count}; model.scale=scale
    config.mesh.strides={axis:(config.mesh.domain[axis][1]-config.mesh.domain[axis][0])/count for axis,count in zip(("x","y","theta"),partitions)}
    domain, strides = dict(config.mesh.domain), dict(config.mesh.strides)
    key = random.key(seed); function_space.seedX = seed; function_space_v2.seedX = seed
    sample_class = SampleHoleCircle if problem == "p4" else Sample2D
    sample = sample_class(domain, dict(model.collocation_sizes), model.sample_mode)
    common = dict(domain=domain, strides=strides, Jn=dict(model.Jn), scale=float(model.scale), init_rng=key, kn=epsilon, activation=model.activation)
    start = perf_counter()
    interior_class = V2InteriorConstraint2D if use_v2 else OddEvenDecompositionPointwiseInteriorConstraint2D
    boundary_class = V2BoundaryConstraint2D if use_v2 else OddEvenDecompositionPointwiseBoundaryConstraint2D
    interior = interior_class(**common, num_quads=int(model.num_quads), coeff_fns=dict(model.coeff))
    boundary = boundary_class(**common)
    dummy = (jnp.zeros(1), jnp.zeros(1), jnp.zeros(1))
    ip, bp = interior.init(key, *dummy), boundary.init(key, *dummy)
    interior_fn = jax.jit(vmap(lambda x,y,t: interior.apply(ip,x,y,t)))
    boundary_fn = jax.jit(vmap(lambda x,y,t: boundary.apply(bp,x,y,t)))
    trace_fn = None if not use_v2 else jax.jit(vmap(lambda x,y,t: boundary.apply(bp,x,y,t,method=boundary.trace)))
    feature_seconds = perf_counter()-start
    start = perf_counter()
    num_columns = (4 if use_v2 else 2) * features * patch_count
    least_squares = StreamingQRLeastSquares(num_columns, rcond=rcond)
    weights=(1.0,1.0,1.0,1.0) if block_weights is None else block_weights
    for points, value_fn, coordinate_index in zip((sample.pts_left,sample.pts_right,sample.pts_lower,sample.pts_upper),(model.bdy_cond.f_l,model.bdy_cond.f_r,model.bdy_cond.f_b,model.bdy_cond.f_t),(1,1,0,0)):
        x,y,t=points
        coordinate = points[coordinate_index]
        for begin in range(0, x.shape[0], 256):
            stop = min(begin + 256, x.shape[0])
            evaluator = trace_fn if use_v2 else boundary_fn
            block = np.asarray(evaluator(x[begin:stop], y[begin:stop], t[begin:stop]))
            block_rhs = np.asarray(vmap(value_fn)(coordinate[begin:stop])).reshape(-1)
            least_squares.add(block, block_rhs, row_factors=np.sqrt(weights[0]))
    if problem == "p4":
        circle_x, circle_y, normal_angle = sample.pts_hole
        angular_offsets = jnp.linspace(-0.49 * jnp.pi, 0.49 * jnp.pi, 32)
        circle_x = jnp.repeat(circle_x, angular_offsets.size, axis=0)
        circle_y = jnp.repeat(circle_y, angular_offsets.size, axis=0)
        circle_theta = (jnp.repeat(normal_angle, angular_offsets.size, axis=0) + jnp.tile(angular_offsets, normal_angle.shape[0])[:, None]) % (2 * jnp.pi)
        for begin in range(0, circle_x.shape[0], 256):
            stop = min(begin + 256, circle_x.shape[0])
            evaluator = trace_fn if use_v2 else boundary_fn
            block = np.asarray(evaluator(circle_x[begin:stop], circle_y[begin:stop], circle_theta[begin:stop]))
            block_rhs = np.asarray(vmap(model.bdy_cond.f_circle)(circle_x[begin:stop], circle_y[begin:stop])).reshape(-1)
            least_squares.add(block, block_rhs, row_factors=np.sqrt(weights[0]))
    ix,iy,it=sample.pts_int
    # V2 supplies both independent antipodal pairs at each first-quadrant
    # angle.  The legacy space needs the mirrored second pair explicitly.
    if not use_v2:
        ix=jnp.concatenate((ix,ix)); iy=jnp.concatenate((iy,iy)); it=jnp.concatenate((it,jnp.pi-it))
    for begin in range(0, ix.shape[0], 128):
        stop = min(begin + 128, ix.shape[0])
        bx, by, bt = ix[begin:stop], iy[begin:stop], it[begin:stop]
        block = np.asarray(interior_fn(bx, by, bt)).reshape(-1, num_columns)
        qp=np.asarray(vmap(model.source)(bx,by,bt)).reshape(-1)
        qm=np.asarray(vmap(model.source)(bx,by,(bt+jnp.pi)%(2*jnp.pi))).reshape(-1)
        qe,qo=.5*(qp+qm),.5*(qp-qm)
        # The macro balance is <v·grad j> + sigma_a <r> = <q_even>.
        # Use the angular mean in the macro equation and only its fluctuation
        # in the micro equation.  This matters for P5's isotropic source.
        angular_nodes=np.linspace(0.0,2.0*np.pi,int(model.num_quads),endpoint=False)
        q_average=np.zeros_like(qe)
        for angle in angular_nodes:
            q_average += np.asarray(
                vmap(model.source)(bx,by,jnp.full_like(bt,angle))
            ).reshape(-1)/angular_nodes.size
        if use_v2:
            q2p=np.asarray(vmap(model.source)(bx,by,-bt)).reshape(-1)
            q2m=np.asarray(vmap(model.source)(bx,by,jnp.pi-bt)).reshape(-1)
            even1, odd1=qe, qo
            even2, odd2=.5*(q2p+q2m),.5*(q2p-q2m)
            block_rhs=np.column_stack((2.0*q_average,epsilon**2*((even1-q_average)+(even2-q_average)),epsilon*(odd1+odd2),epsilon**2*(even1-even2),epsilon*(odd1-odd2))).reshape(-1)
            row_factors=np.ones(5*(stop-begin))
        else:
            block_rhs=np.column_stack((q_average,epsilon**2*(qe-q_average),epsilon*qo)).reshape(-1)
            row_factors=np.tile(np.sqrt(np.asarray(weights[1:])), stop-begin)
        least_squares.add(block, block_rhs, row_factors=row_factors)
    assembly_seconds=perf_counter()-start
    start=perf_counter(); coefficients,diagnostics=least_squares.solve(); solve_seconds=perf_counter()-start
    constructor_class=V2Constructor2D if use_v2 else OddEvenDecompositionConstructor2D
    constructor=constructor_class(**common,coefficients=jnp.asarray(coefficients.reshape(-1)))
    cp=constructor.init(key,*dummy); approx=jax.jit(vmap(lambda x,y,t: constructor.apply(cp,x,y,t)))
    nx,ny,na=65,65,64
    x=np.linspace(float(domain["x"][0]),float(domain["x"][1]),nx)
    y=np.linspace(float(domain["y"][0]),float(domain["y"][1]),ny)
    # Periodic midpoint rule avoids assigning finite quadrature weight to the
    # four measure-zero seams of the quadrant-wise V2 reconstruction.
    theta=(np.arange(na)+.5)*(2*np.pi/na)
    numerical=np.empty((nx,ny,na)); start=perf_counter()
    for i in range(nx):
        yy,tt=np.meshgrid(y,theta,indexing="ij"); xx=np.full_like(yy,x[i])
        numerical[i]=np.asarray(approx(jnp.asarray(xx.reshape(-1,1)),jnp.asarray(yy.reshape(-1,1)),jnp.asarray(tt.reshape(-1,1)))).reshape(ny,na)
    rho = numerical.mean(axis=2)
    if problem == "p3":
        reference=np.exp(-x[:,None,None]-y[None,:,None])+np.zeros((1,1,na)); reference_rho=reference.mean(axis=2); mask=np.ones((nx,ny),dtype=bool)
    elif problem == "p4":
        xx,yy=np.meshgrid(x,y,indexing="ij"); reference_rho=1/(1+xx**2+yy**2); mask=xx**2+yy**2>=.25; reference=np.broadcast_to(reference_rho[:,:,None],numerical.shape)
    elif problem == "p5":
        reference_dir=Path("results/references_p5_boundary_constant") if p5_constant_inflow else Path("results/references_p5_boundary")
        candidates=[reference_dir/f"p5_parity_ref_eps_{epsilon:.0e}_level_{level}.npz" for level in ("C","B","A")]
        reference_path=next((path for path in candidates if path.exists()),None)
        if reference_path is None:
            raise FileNotFoundError(f"numerical P5 reference is missing for epsilon={epsilon}")
        with np.load(reference_path) as data:
            rx,ry,rt,rf,rr=data["x"],data["y"],data["theta"],data["f"],data["rho"]
        order=np.argsort(rt); rt=rt[order]; rf=rf[:,:,order]
        rt=np.concatenate((rt[-1:]-2*np.pi,rt,rt[:1]+2*np.pi)); rf=np.concatenate((rf[:,:,-1:],rf,rf[:,:,:1]),axis=2)
        phase_points=np.stack(np.meshgrid(x,y,theta,indexing="ij"),axis=-1)
        reference=RegularGridInterpolator((rx,ry,rt),rf,bounds_error=False,fill_value=None)(phase_points)
        density_points=np.stack(np.meshgrid(x,y,indexing="ij"),axis=-1)
        reference_rho=RegularGridInterpolator((rx,ry),rr,bounds_error=False,fill_value=None)(density_points)
        mask=np.ones((nx,ny),dtype=bool)
    else:
        mask=np.ones((nx,ny),dtype=bool)
    phase_mask=np.broadcast_to(mask[:,:,None],numerical.shape)
    error_f=float(np.linalg.norm((numerical-reference)[phase_mask])/np.linalg.norm(reference[phase_mask])); error_rho=float(np.linalg.norm((rho-reference_rho)[mask])/np.linalg.norm(reference_rho[mask])); evaluation_seconds=perf_counter()-start
    rho_correlation=float(np.corrcoef(rho[mask],reference_rho[mask])[0,1])
    record={"problem":problem,"method":"oe_aprfm","epsilon":epsilon,"seed":seed,"relative_l2_f":error_f,"relative_l2_rho":error_rho,"residual_half":diagnostics["normalized_residual_rms"],"empirical_stability_ratio":np.hypot(error_f,error_rho)/diagnostics["normalized_residual_rms"],"condition_number":diagnostics["condition_number"],"largest_singular_value":diagnostics["largest_singular_value"],"smallest_effective_singular_value":diagnostics["smallest_effective_singular_value"],"singular_value_threshold":diagnostics["singular_value_threshold"],"coefficient_norm":float(np.linalg.norm(coefficients)),"rank":diagnostics["rank"],"num_rows":diagnostics["num_rows"],"num_columns":diagnostics["num_columns"],"num_angular":int(model.num_quads),"oversampling_ratio":diagnostics["num_rows"]/diagnostics["num_columns"],"partitions":partitions,"strides":dict(config.mesh.strides),"features_per_patch":features,"scale":scale,"rcond":rcond,"block_weights":block_weights,"collocation":collocation if collocation is not None else dict(model.collocation_sizes),"feature_seconds":feature_seconds,"assembly_seconds":assembly_seconds,"solve_seconds":solve_seconds,"evaluation_seconds":evaluation_seconds,"total_seconds":feature_seconds+assembly_seconds+solve_seconds+evaluation_seconds,"min_rho_h":float(rho.min()),"max_rho_h":float(rho.max()),"min_f_h":float(numerical.min()),"correlation_rho":rho_correlation,"negative_fraction_f":float(np.mean(numerical < 0.0)),"negative_fraction_rho":float(np.mean(rho < 0.0))}
    record["angular_representation"] = angular_representation
    record['residual_definition'] = 'complete_normalized_training_residual_v2'
    record['boundary_implementation'] = ('independent_pair_traces_v2' if angular_representation == 'four_component' else 'legacy_two_component')
    root = Path(__file__).resolve().parents[1]
    record['source_sha256'] = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ('scripts/run_p3_oe_aprfm.py',
                     'src/solver/streaming_least_squares.py',
                     'src/constraints/continuous_2d_odd_even_v2.py',
                     'src/modules/function_space_v2.py')
    }
    output_dir.mkdir(parents=True,exist_ok=True); suffix=f"_{tag}" if tag else ""; stem=f"{problem}_oe_aprfm_eps_{epsilon:.0e}_seed_{seed}{suffix}"; (output_dir/f"{stem}.json").write_text(json.dumps(record,indent=2)+"\n"); np.savez_compressed(output_dir/f"{stem}.npz",x=x,y=y,theta=theta,f=numerical,rho=rho,reference_f=reference,reference_rho=reference_rho,error_f=np.abs(numerical-reference),error_rho=np.abs(rho-reference_rho),mask=mask); return record


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--problem",choices=("p3","p4","p5"),default="p3"); parser.add_argument("--angular-representation",choices=("legacy_two_component","four_component"),default="four_component"); parser.add_argument("--p5-constant-inflow",action="store_true"); parser.add_argument("--epsilon",type=float,required=True); parser.add_argument("--seed",type=int,required=True); parser.add_argument("--output-dir",type=Path,default=Path("results/raw")); parser.add_argument("--partitions",type=int,nargs=3); parser.add_argument("--features",type=int); parser.add_argument("--scale",type=float,default=1.0); parser.add_argument("--rcond",type=float,default=1e-12); parser.add_argument("--block-weights",type=float,nargs=4); parser.add_argument("--collocation",type=int,nargs=3); parser.add_argument("--tag",default=""); args=parser.parse_args(); print(json.dumps(run(args.epsilon,args.seed,args.output_dir,args.problem,angular_representation=args.angular_representation,p5_constant_inflow=args.p5_constant_inflow,partitions=None if args.partitions is None else tuple(args.partitions),features=args.features,scale=args.scale,rcond=args.rcond,block_weights=None if args.block_weights is None else tuple(args.block_weights),collocation=None if args.collocation is None else tuple(args.collocation),tag=args.tag),indent=2))
