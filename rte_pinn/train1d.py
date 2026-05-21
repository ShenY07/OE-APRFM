import numpy as np
import os
import time
import torch
import torch.optim as optim
from torch.optim import lr_scheduler

from configuration.rte1d_setting import get_config
from modules.networks import ResNet_1d, Xavier_initi
from modules.data_pipeline1d import Data_Loader
from constraints.rte_eqn import OERTE1D


cfg = get_config()
kn = cfg.rte.kn
device_ids = cfg.model.device_ids
device = torch.device(
    "cuda:{:d}".format(device_ids[0]) if torch.cuda.is_available() else "cpu"
)
print(f"Using device: {device}")
model_config = cfg.model

j_nn = ResNet_1d(
    input_size=model_config.fn_j.input_size,
    hidden_sizes=model_config.fn_j.hidden_sizes,
    output_size=model_config.fn_j.output_size,
    device=device,
).to(device)
r_nn = ResNet_1d(
    input_size=model_config.fn_r.input_size,
    hidden_sizes=model_config.fn_r.hidden_sizes,
    output_size=model_config.fn_r.output_size,
    device=device,
).to(device)


model_ckpt_path = "./ckpts/"
os.makedirs(model_ckpt_path, exist_ok=True)
records_path = "./records/"
os.makedirs(records_path, exist_ok=True)

# Load pretrained parameters if available, otherwise initialize with Xavier
for model, name in [
    (j_nn, f"j_nn_ex2_params_{kn:.1e}.pt"),
    (r_nn, f"r_nn_ex2_params_{kn:.1e}.pt"),
]:
    ckpt_file = os.path.join(model_ckpt_path, name)

    if os.path.isfile(ckpt_file):
        model.load_state_dict(torch.load(ckpt_file, map_location=device))
        print(f"[Model Parameters] Loaded pretrained weights: {ckpt_file}")
    else:
        print(
            f"[Model Parameters] Pretrained weights not found, initializing with Xavier: {ckpt_file}"
        )
        Xavier_initi(model)

rte_constraint = OERTE1D(cfg)

data = np.load(f"./data/rte1d_ex2_ref_{kn:.1e}.npz")
mesh_x = data["mesh_x"]
mesh_v = data["mesh_v"]
# 生成所有 (x, v) 组合，保证 shape 匹配
mesh_xx, mesh_vv = np.meshgrid(mesh_x, mesh_v, indexing="ij")
mesh_xx = torch.from_numpy(mesh_xx).float().reshape(-1, 1)
mesh_vv = torch.from_numpy(mesh_vv).float().reshape(-1, 1)
F_ref = data["F"].T.reshape(-1, 1)  # 保证 shape 一致

adam_optimizer = optim.Adam(
    list(j_nn.parameters()) + list(r_nn.parameters()), lr=cfg.model.Adam.lr
)
# scheduler = lr_scheduler.StepLR(
#     adam_optimizer,
#     step_size=cfg.model.Adam.step_size,
#     gamma=cfg.model.Adam.gamma,
# )

dataloader = Data_Loader(cfg)
Iter = cfg.model.iteration_steps
regularizers = cfg.model.regularizers
optimizer = adam_optimizer

time_start = time.time()
print("Start training...")
for it in range(Iter):
    data = next(dataloader)
    train_interior = (
        data["interior"]["x"].to(device),
        data["interior"]["v"].to(device),
    )
    train_bc = (
        data["boundary_left"]["x"].to(device),
        data["boundary_left"]["v"].to(device),
        data["boundary_right"]["x"].to(device),
        data["boundary_right"]["v"].to(device),
    )

    optimizer.zero_grad()
    res1, res2, res3 = rte_constraint.residual((j_nn, r_nn), train_interior)
    res_bcl, res_bcr = rte_constraint.bc((j_nn, r_nn), train_bc[:2], train_bc[2:])

    if it == 0:
        results_adam = {
            "loss_1": np.zeros(Iter),
            "loss_2": np.zeros(Iter),
            "loss_3": np.zeros(Iter),
            "loss_bc": np.zeros(Iter),
            "total_loss": np.zeros(Iter),
            "error_f": np.zeros(Iter),
        }

    loss_1 = torch.mean(res1**2)
    loss_2 = torch.mean(res2**2)
    loss_3 = torch.mean(res3**2)
    loss_bc = torch.mean(res_bcl**2) + torch.mean(res_bcr**2)
    loss = (
        regularizers[0] * loss_1
        + regularizers[1] * loss_2
        + regularizers[2] * loss_3
        + regularizers[3] * loss_bc
    )

    j_app = (
        rte_constraint
        .model_j(j_nn, [mesh_xx.to(device), mesh_vv.to(device)])
        .detach()
        .cpu()
        .numpy()
    )
    r_app = (
        rte_constraint
        .model_r(r_nn, [mesh_xx.to(device), mesh_vv.to(device)])
        .detach()
        .cpu()
        .numpy()
    )
    f_app = kn * j_app + r_app
    err_f = np.linalg.norm(f_app - F_ref) / np.linalg.norm(F_ref)

    results_adam["loss_1"][it] = loss_1.item()
    results_adam["loss_2"][it] = loss_2.item()
    results_adam["loss_3"][it] = loss_3.item()
    results_adam["loss_bc"][it] = loss_bc.item()
    results_adam["total_loss"][it] = loss.item()
    results_adam["error_f"][it] = err_f

    loss.backward()
    optimizer.step()
    # scheduler.step()

    if it % 200 == 0:
        print(
            f"Iter: {it + 1}, lr:{optimizer.param_groups[0]['lr']:.2e}, Loss: {loss.item():.4e}, Rel L2 Error f: {err_f:.4e}"
        )
        print(
            f"Loss1: {loss_1.item():.4e}, Loss2: {loss_2.item():.4e}, Loss3: {loss_3.item():.4e}, Loss_bc: {loss_bc.item():.4e}, "
        )

    if it == Iter - 1 or (it + 1) % 5000 == 0:
        np.savez(
            os.path.join(records_path, f"rte1d_oer_te_results_adam_{kn:.1e}.npz"),
            **results_adam,
        )

save_param_path_j = os.path.join(model_ckpt_path, f"j_nn_ex2_params_{kn:.1e}.pt")
save_param_path_r = os.path.join(model_ckpt_path, f"r_nn_ex2_params_{kn:.1e}.pt")

torch.save(j_nn.state_dict(), save_param_path_j)
torch.save(r_nn.state_dict(), save_param_path_r)
print(f"Saved trained model parameters to {model_ckpt_path}.")
time_end = time.time()
print(f"Training completed in {time_end - time_start:.2f} seconds.")
