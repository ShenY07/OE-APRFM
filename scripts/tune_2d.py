"""Run one indexed 2D OE-RFM tuning candidate."""
import argparse
import itertools
from pathlib import Path
from run_p3_oe_aprfm import run

parser=argparse.ArgumentParser(); parser.add_argument("--problem",choices=("p3","p4","p5"),required=True); parser.add_argument("--epsilon",type=float,required=True); parser.add_argument("--index",type=int,required=True); args=parser.parse_args()
combinations=list(itertools.product((((1,1,1),128),((2,1,1),96),((2,2,1),64)),(0.5,1.0,2.0)))
(partitions,features),scale=combinations[args.index]
run(args.epsilon,11,Path(f"results/tuning/{args.problem}"),args.problem,partitions=partitions,features=features,scale=scale,tag=f"t{args.index:03d}")
