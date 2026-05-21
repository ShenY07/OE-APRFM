import numpy as np
import os
import time
import torch
import torch.optim as optim

from configuration.rte2d_setting import get_config
from modules.networks import ResNet_2d, Xavier_initi
from modules.data_pipeline2d import Data_Loader
from constraints.rte_eqn import OERTE2D


cfg = get_config()
kn = cfg.rte.kn
freq = cfg.rte.freq
device_ids = cfg.model.device_ids
device = torch.device(
    "cuda:{:d}".format(device_ids[0]) if torch.cuda.is_available() else "cpu"
)
print(f"Using device: {device}")
model_config = cfg.model

j1_nn = ResNet_2d(
    input_size=model_config.fn_j1.input_size,
    hidden_sizes=model_config.fn_j1.hidden_sizes,
    output_size=model_config.fn_j1.output_size,
    device=device,
    freq=freq,
).to(device)
r1_nn = ResNet_2d(
    input_size=model_config.fn_r1.input_size,
    hidden_sizes=model_config.fn_r1.hidden_sizes,
    output_size=model_config.fn_r1.output_size,
    device=device,
    freq=freq,
).to(device)
j2_nn = ResNet_2d(
    input_size=model_config.fn_j2.input_size,
    hidden_sizes=model_config.fn_j2.hidden_sizes,
    output_size=model_config.fn_j2.output_size,
    device=device,
    freq=freq,
).to(device)
r2_nn = ResNet_2d(
    input_size=model_config.fn_r2.input_size,
    hidden_sizes=model_config.fn_r2.hidden_sizes,
    output_size=model_config.fn_r2.output_size,
    device=device,
    freq=freq,
).to(device)


model_ckpt_path = "./ckpts1/"
os.makedirs(model_ckpt_path, exist_ok=True)
records_path = "./records1/"
os.makedirs(records_path, exist_ok=True)

# Load pretrained parameters if available, otherwise initialize with Xavier
for model, name in [
    (j1_nn, f"j1_nn_ex5_params_{kn:.1e}.pt"),
    (r1_nn, f"r1_nn_ex5_params_{kn:.1e}.pt"),
    (j2_nn, f"j2_nn_ex5_params_{kn:.1e}.pt"),
    (r2_nn, f"r2_nn_ex5_params_{kn:.1e}.pt"),
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

rte_constraint = OERTE2D(cfg)


adam_optimizer = optim.Adam(
    list(j1_nn.parameters())
    + list(r1_nn.parameters())
    + list(j2_nn.parameters())
    + list(r2_nn.parameters()),
    lr=cfg.model.Adam.lr,
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

print(f"Kn = {kn:.1e}, Training Iterations = {Iter}.")

time_start = time.time()
print("Start training...")
for it in range(Iter):
    data = next(dataloader)
    train_interior = (
        data["interior"]["x"].to(device),
        data["interior"]["y"].to(device),
        data["interior"]["theta"].to(device),
    )
    train_bc = (
        data["boundary_left"]["x"].to(device),
        data["boundary_left"]["y"].to(device),
        data["boundary_left"]["theta"].to(device),
        data["boundary_right"]["x"].to(device),
        data["boundary_right"]["y"].to(device),
        data["boundary_right"]["theta"].to(device),
        data["boundary_bottom"]["x"].to(device),
        data["boundary_bottom"]["y"].to(device),
        data["boundary_bottom"]["theta"].to(device),
        data["boundary_top"]["x"].to(device),
        data["boundary_top"]["y"].to(device),
        data["boundary_top"]["theta"].to(device),
    )

    optimizer.zero_grad()
    res1, res2, res3, res4, res5 = rte_constraint.residual(
        (j1_nn, r1_nn, j2_nn, r2_nn), train_interior
    )
    res_bcl1, res_bcl2, res_bcr1, res_bcr2, res_bcb1, res_bcb2, res_bct1, res_bct2 = (
        rte_constraint.bc(
            (j1_nn, r1_nn, j2_nn, r2_nn),
            train_bc[:3],
            train_bc[3:6],
            train_bc[6:9],
            train_bc[9:],
        )
    )

    if it == 0:
        results_adam = {
            "loss_1": np.zeros(Iter),
            "loss_2": np.zeros(Iter),
            "loss_3": np.zeros(Iter),
            "loss_4": np.zeros(Iter),
            "loss_5": np.zeros(Iter),
            "loss_bc": np.zeros(Iter),
            "total_loss": np.zeros(Iter),
            "error_f": np.zeros(Iter),
        }

    loss_1 = torch.mean(res1**2)
    loss_2 = torch.mean(res2**2)
    loss_3 = torch.mean(res3**2)
    loss_4 = torch.mean(res4**2)
    loss_5 = torch.mean(res5**2)
    loss_bc = (
        torch.mean(res_bcl1**2)
        + torch.mean(res_bcl2**2)
        + torch.mean(res_bcr1**2)
        + torch.mean(res_bcr2**2)
        + torch.mean(res_bcb1**2)
        + torch.mean(res_bcb2**2)
        + torch.mean(res_bct1**2)
        + torch.mean(res_bct2**2)
    )
    loss = (
        regularizers[0] * loss_1
        + regularizers[1] * loss_2
        + regularizers[2] * loss_3
        + regularizers[3] * loss_4
        + regularizers[4] * loss_5
        + regularizers[5] * loss_bc
    )

    results_adam["loss_1"][it] = loss_1.item()
    results_adam["loss_2"][it] = loss_2.item()
    results_adam["loss_3"][it] = loss_3.item()
    results_adam["loss_4"][it] = loss_4.item()
    results_adam["loss_5"][it] = loss_5.item()
    results_adam["loss_bc"][it] = loss_bc.item()
    results_adam["total_loss"][it] = loss.item()

    loss.backward()
    optimizer.step()
    # scheduler.step()

    if it % 200 == 0:
        print(
            f"Iter: {it + 1}, lr:{optimizer.param_groups[0]['lr']:.2e}, Loss: {loss.item():.4e}"
        )
        print(
            f"Loss1: {loss_1.item():.4e}, Loss2: {loss_2.item():.4e}, Loss3: {loss_3.item():.4e}, "
        )
        print(
            f"Loss4: {loss_4.item():.4e}, Loss5: {loss_5.item():.4e}, Loss_bc: {loss_bc.item():.4e}. "
        )

    if it == Iter - 1 or (it + 1) % 5000 == 0:
        np.savez(
            os.path.join(records_path, f"rte2d_oe_rte_ex5_results_adam_{kn:.1e}.npz"),
            **results_adam,
        )

save_param_path_j1 = os.path.join(model_ckpt_path, f"j1_nn_ex5_params_{kn:.1e}.pt")
save_param_path_r1 = os.path.join(model_ckpt_path, f"r1_nn_ex5_params_{kn:.1e}.pt")
save_param_path_j2 = os.path.join(model_ckpt_path, f"j2_nn_ex5_params_{kn:.1e}.pt")
save_param_path_r2 = os.path.join(model_ckpt_path, f"r2_nn_ex5_params_{kn:.1e}.pt")

torch.save(j1_nn.state_dict(), save_param_path_j1)
torch.save(r1_nn.state_dict(), save_param_path_r1)
torch.save(j2_nn.state_dict(), save_param_path_j2)
torch.save(r2_nn.state_dict(), save_param_path_r2)
print(f"Saved trained model parameters to {model_ckpt_path}.")
time_end = time.time()
print(
    f"For kn = {kn:.1e}, the training completed in {time_end - time_start:.2f} seconds."
)
