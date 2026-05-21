import torch
import numpy as np


class OERTE1D:
    def __init__(self, config):
        self.device = torch.device(
            f"cuda:{config.model.device_ids[0]}" if torch.cuda.is_available(
            ) else "cpu"
        )

        self.kn = config.rte.kn
        self.sigma_a = config.rte.sigma_a
        self.sigma_s = config.rte.sigma_s

        self.xmin, self.xmax = config.domain["x"]
        self.vmin, self.vmax = config.domain["v"]

        self.bdy_left = config.rte.bdy_left
        self.bdy_right = config.rte.bdy_right
        self.source = config.rte.source

        self.num_vquads = config.rte.num_vquads
        vquads, wquads = np.polynomial.legendre.leggauss(self.num_vquads)
        vquads = 0.5 * (vquads + 1) * (self.vmax - self.vmin) + self.vmin
        wquads = 0.5 * (self.vmax - self.vmin) * wquads
        self.vquads = torch.tensor(vquads, dtype=torch.float32).to(self.device)
        self.wquads = torch.tensor(wquads, dtype=torch.float32).to(self.device)

    # ---- 网络输出，统一使用 inputs=(x,v) ----
    def model_r(self, net, inputs):
        x, v = inputs
        xv1 = torch.cat([x, v], dim=-1)
        xv2 = torch.cat([x, -v], dim=-1)
        return 0.5 * (net(xv1) + net(xv2))

    def model_j(self, net, inputs):
        x, v = inputs
        xv1 = torch.cat([x, v], dim=-1)
        xv2 = torch.cat([x, -v], dim=-1)
        return 0.5 * (net(xv1) - net(xv2))

    # ---- v 积分 ----
    def v_average(self, g, inputs):
        x, _ = inputs
        Nx = x.shape[0]
        Nv = self.num_vquads

        x_rep = x[:, None, :].repeat(1, Nv, 1)
        v_rep = self.vquads[None, :, None].repeat(Nx, 1, 1)

        g_val = g((x_rep, v_rep))  # [Nx, Nv, 1]
        avg = torch.sum(g_val * self.wquads[None, :, None], dim=1) / (
            self.vmax - self.vmin
        )

        return avg

    def avg_vdj(self, net_j, x):
        """
        compute < v * ∂_x j >
        """
        Nx = x.shape[0]
        Nv = self.num_vquads

        x_rep = x[:, None, :].repeat(1, Nv, 1)  # [Nx, Nv, 1]
        v_rep = self.vquads[None, :, None].repeat(Nx, 1, 1)

        x_rep.requires_grad_(True)

        j_rep = self.model_j(net_j, (x_rep, v_rep))  # j(x,v_quad)

        dj_dx_rep = torch.autograd.grad(
            j_rep,
            x_rep,
            grad_outputs=torch.ones_like(j_rep),
            retain_graph=True,
            create_graph=True,
        )[0]

        vdj_rep = v_rep * dj_dx_rep

        avg_vdj = torch.sum(vdj_rep * self.wquads[None, :, None], dim=1) / (
            self.vmax - self.vmin
        )

        return avg_vdj

    # ---- 残差 ----

    def residual(self, sol, inputs):
        net_j, net_r = sol
        x, v = inputs
        x.requires_grad_(True)

        j = self.model_j(net_j, inputs)
        r = self.model_r(net_r, inputs)

        Q_even = 0.5 * (self.source(x, v) + self.source(x, -v))
        Q_odd = 0.5 * (self.source(x, v) - self.source(x, -v))

        dj_dx = torch.autograd.grad(
            j, x, grad_outputs=torch.ones_like(j), retain_graph=True, create_graph=True
        )[0]

        dr_dx = torch.autograd.grad(
            r, x, grad_outputs=torch.ones_like(r), retain_graph=True, create_graph=True
        )[0]

        vdj = v * dj_dx
        vdr = v * dr_dx

        avg_vdj = self.avg_vdj(net_j, x)
        avg_r = self.v_average(lambda inp: self.model_r(net_r, inp), inputs)
        avg_Q = self.v_average(
            lambda inp: (
                0.5 * (self.source(inp[0], inp[1]) +
                       self.source(inp[0], -inp[1]))
            ),
            inputs,
        )

        res1 = avg_vdj + self.sigma_a(x) * avg_r - avg_Q
        res2 = (
            self.kn**2 * (vdj - avg_vdj)
            + (self.sigma_s(x) + self.kn**2 * self.sigma_a(x)) * (r - avg_r)
            - self.kn**2 * (Q_even - avg_Q)
        )
        res3 = (
            (self.sigma_s(x) + self.kn**2 * self.sigma_a(x)) *
            j + vdr - self.kn * Q_odd
        )

        return res1, res2, res3

    # ---- 边界条件 ----
    def bc(self, sol, inputs_left, inputs_right):
        net_j, net_r = sol
        x_left, v_left = inputs_left
        x_right, v_right = inputs_right

        r_left = self.model_r(net_r, inputs_left)
        j_left = self.model_j(net_j, inputs_left)
        r_right = self.model_r(net_r, inputs_right)
        j_right = self.model_j(net_j, inputs_right)

        psi_left = r_left + self.kn * j_left
        psi_right = r_right - self.kn * j_right

        bc_left = psi_left - self.bdy_left(v_left)
        bc_right = psi_right - self.bdy_right(v_right)

        return bc_left, bc_right


class OERTE2D:
    def __init__(self, config):
        self.device = torch.device(
            f"cuda:{config.model.device_ids[0]}" if torch.cuda.is_available(
            ) else "cpu"
        )

        self.kn = config.rte.kn
        self.freq = config.rte.freq
        self.sigma_a = config.rte.sigma_a
        self.sigma_s = config.rte.sigma_s

        self.xmin, self.xmax = config.domain["x"]
        self.ymin, self.ymax = config.domain["y"]
        self.thetamin, self.thetamax = config.domain["theta"]

        self.bdy_left = config.rte.bdy_left
        self.bdy_right = config.rte.bdy_right
        self.bdy_bottom = config.rte.bdy_bottom
        self.bdy_top = config.rte.bdy_top
        self.source = config.rte.source

        self.num_vquads = config.rte.num_vquads
        vquads, wquads = np.polynomial.legendre.leggauss(self.num_vquads)
        vquads = 0.5 * (vquads + 1.0) * (self.thetamax -
                                         self.thetamin) + self.thetamin
        wquads = 0.5 * (self.thetamax - self.thetamin) * wquads
        self.vquads = torch.tensor(vquads, dtype=torch.float32).to(self.device)
        self.wquads = torch.tensor(wquads, dtype=torch.float32).to(self.device)

    def model_r1(self, net_r1, inputs):
        x, y, theta = inputs
        r11 = net_r1(torch.cat([x, y, -theta], dim=-1))
        r12 = net_r1(torch.cat([x, y, -theta + torch.pi], dim=-1))
        return 0.5 * (r11 + r12)

    def model_j1(self, net_j1, inputs):
        x, y, theta = inputs
        j11 = net_j1(torch.cat([x, y, -theta], dim=-1))
        j12 = net_j1(torch.cat([x, y, -theta + torch.pi], dim=-1))
        return 0.5 * (j11 - j12)

    def model_r2(self, net_r2, inputs):
        x, y, theta = inputs
        r21 = net_r2(torch.cat([x, y, theta], dim=-1))
        r22 = net_r2(torch.cat([x, y, theta + torch.pi], dim=-1))
        return 0.5 * (r21 + r22)

    def model_j2(self, net_j2, inputs):
        x, y, theta = inputs
        j21 = net_j2(torch.cat([x, y, theta], dim=-1))
        j22 = net_j2(torch.cat([x, y, theta + torch.pi], dim=-1))
        return 0.5 * (j21 - j22)

    def v_average(self, g, inputs):
        x, y, _ = inputs
        Nx = x.shape[0]
        Nv = self.num_vquads

        x_rep = x[:, None, :].repeat(1, Nv, 1)
        y_rep = y[:, None, :].repeat(1, Nv, 1)
        theta_rep = self.vquads[None, :, None].repeat(Nx, 1, 1)

        g_val = g((x_rep, y_rep, theta_rep))  # [Nx, Nv, 1]
        avg = torch.sum(g_val * self.wquads[None, :, None], dim=1) / (
            self.thetamax - self.thetamin
        )

        return avg

    def avg_vdj1(self, model_j, net_j, x, y):
        """
        compute < v * ∂_x j >
        """
        Nx = x.shape[0]
        Nv = self.num_vquads

        x_rep = x[:, None, :].repeat(1, Nv, 1)  # [Nx, Nv, 1]
        y_rep = y[:, None, :].repeat(1, Nv, 1)
        theta_rep = self.vquads[None, :, None].repeat(Nx, 1, 1)

        x_rep.requires_grad_(True)
        y_rep.requires_grad_(True)

        j_rep = model_j(net_j, (x_rep, y_rep, theta_rep))  # j(x,v_quad)

        dj_dx_rep = torch.autograd.grad(
            j_rep,
            x_rep,
            grad_outputs=torch.ones_like(j_rep),
            retain_graph=True,
            create_graph=True,
        )[0]

        dj_dy_rep = torch.autograd.grad(
            j_rep,
            y_rep,
            grad_outputs=torch.ones_like(j_rep),
            retain_graph=True,
            create_graph=True,
        )[0]

        vdj_rep = torch.cos(theta_rep) * dj_dx_rep - \
            torch.sin(theta_rep) * dj_dy_rep

        avg_vdj = torch.sum(vdj_rep * self.wquads[None, :, None], dim=1) / (
            self.thetamax - self.thetamin
        )

        return avg_vdj

    def avg_vdj2(self, model_j, net_j, x, y):
        """
        compute < v * ∂_x j >
        """
        Nx = x.shape[0]
        Nv = self.num_vquads

        x_rep = x[:, None, :].repeat(1, Nv, 1)  # [Nx, Nv, 1]
        y_rep = y[:, None, :].repeat(1, Nv, 1)
        theta_rep = self.vquads[None, :, None].repeat(Nx, 1, 1)

        x_rep.requires_grad_(True)
        y_rep.requires_grad_(True)

        j_rep = model_j(net_j, (x_rep, y_rep, theta_rep))  # j(x,v_quad)

        dj_dx_rep = torch.autograd.grad(
            j_rep,
            x_rep,
            grad_outputs=torch.ones_like(j_rep),
            retain_graph=True,
            create_graph=True,
        )[0]

        dj_dy_rep = torch.autograd.grad(
            j_rep,
            y_rep,
            grad_outputs=torch.ones_like(j_rep),
            retain_graph=True,
            create_graph=True,
        )[0]

        vdj_rep = torch.cos(theta_rep) * dj_dx_rep + \
            torch.sin(theta_rep) * dj_dy_rep

        avg_vdj = torch.sum(vdj_rep * self.wquads[None, :, None], dim=1) / (
            self.thetamax - self.thetamin
        )

        return avg_vdj

    def residual(self, sol, inputs):
        net_j1, net_r1, net_j2, net_r2 = sol
        x, y, theta = inputs
        xi, eta = torch.cos(theta), torch.sin(theta)
        x.requires_grad_(True)
        y.requires_grad_(True)

        j1 = self.model_j1(net_j1, inputs)
        r1 = self.model_r1(net_r1, inputs)
        j2 = self.model_j2(net_j2, inputs)
        r2 = self.model_r2(net_r2, inputs)
        Q1_even = 0.5 * (
            self.source(x, y, -theta) + self.source(x, y, -theta + torch.pi)
        )
        Q1_odd = 0.5 * (
            self.source(x, y, -theta) - self.source(x, y, -theta + torch.pi)
        )
        Q2_even = 0.5 * (self.source(x, y, theta) +
                         self.source(x, y, theta + torch.pi))
        Q2_odd = 0.5 * (self.source(x, y, theta) -
                        self.source(x, y, theta + torch.pi))

        dj1_dx = torch.autograd.grad(
            j1,
            x,
            grad_outputs=torch.ones_like(j1),
            retain_graph=True,
            create_graph=True,
        )[0]
        dj1_dy = torch.autograd.grad(
            j1,
            y,
            grad_outputs=torch.ones_like(j1),
            retain_graph=True,
            create_graph=True,
        )[0]
        dr1_dx = torch.autograd.grad(
            r1,
            x,
            grad_outputs=torch.ones_like(r1),
            retain_graph=True,
            create_graph=True,
        )[0]
        dr1_dy = torch.autograd.grad(
            r1,
            y,
            grad_outputs=torch.ones_like(r1),
            retain_graph=True,
            create_graph=True,
        )[0]
        dj2_dx = torch.autograd.grad(
            j2,
            x,
            grad_outputs=torch.ones_like(j2),
            retain_graph=True,
            create_graph=True,
        )[0]
        dj2_dy = torch.autograd.grad(
            j2,
            y,
            grad_outputs=torch.ones_like(j2),
            retain_graph=True,
            create_graph=True,
        )[0]
        dr2_dx = torch.autograd.grad(
            r2,
            x,
            grad_outputs=torch.ones_like(r2),
            retain_graph=True,
            create_graph=True,
        )[0]
        dr2_dy = torch.autograd.grad(
            r2,
            y,
            grad_outputs=torch.ones_like(r2),
            retain_graph=True,
            create_graph=True,
        )[0]

        vdj1 = xi * dj1_dx - eta * dj1_dy
        vdj2 = xi * dj2_dx + eta * dj2_dy
        vdr1 = xi * dr1_dx - eta * dr1_dy
        vdr2 = xi * dr2_dx + eta * dr2_dy

        avg_vdj1 = self.avg_vdj1(self.model_j1, net_j1, x, y)
        avg_r1 = self.v_average(lambda inp: self.model_r1(net_r1, inp), inputs)
        avg_vdj2 = self.avg_vdj2(self.model_j2, net_j2, x, y)
        avg_r2 = self.v_average(lambda inp: self.model_r2(net_r2, inp), inputs)
        avg_Q1 = self.v_average(
            lambda inp: (
                0.5
                * (
                    self.source(inp[0], inp[1], -inp[2])
                    + self.source(inp[0], inp[1], -inp[2] + torch.pi)
                )
            ),
            inputs,
        )
        avg_Q2 = self.v_average(
            lambda inp: (
                0.5
                * (
                    self.source(inp[0], inp[1], inp[2])
                    + self.source(inp[0], inp[1], inp[2] + torch.pi)
                )
            ),
            inputs,
        )

        res1 = (
            avg_vdj1
            + self.sigma_a(x) * avg_r1
            + avg_vdj2
            + self.sigma_a(x) * avg_r2
            - avg_Q1
            - avg_Q2
        )
        res2 = (
            self.kn**2 * (vdj1 - avg_vdj1)
            + (self.sigma_s(x) + self.kn**2 * self.sigma_a(x)) * (r1 - avg_r1)
            - self.kn**2 * (Q1_even - avg_Q1)
            + self.kn**2 * (vdj2 - avg_vdj2)
            + (self.sigma_s(x) + self.kn**2 * self.sigma_a(x)) * (r2 - avg_r2)
            - self.kn**2 * (Q2_even - avg_Q2)
        )
        res3 = (
            (self.sigma_s(x) + self.kn**2 * self.sigma_a(x)) * j1
            + vdr1
            - self.kn * Q1_odd
            + (self.sigma_s(x) + self.kn**2 * self.sigma_a(x)) * j2
            + vdr2
            - self.kn * Q2_odd
        )
        res4 = (
            self.kn**2 * vdj1
            + (self.sigma_s(x) + self.kn**2 * self.sigma_a(x)) * r1
            - self.kn**2 * Q1_even
            - self.kn**2 * vdj2
            - (self.sigma_s(x) + self.kn**2 * self.sigma_a(x)) * r2
            + self.kn**2 * Q2_even
        )
        res5 = (
            (self.sigma_s(x) + self.kn**2 * self.sigma_a(x)) * j1
            + vdr1
            - self.kn * Q1_odd
            - (self.sigma_s(x) + self.kn**2 * self.sigma_a(x)) * j2
            - vdr2
            + self.kn * Q2_odd
        )

        return res1, res2, res3, res4, res5

    def bc(self, sol, inputs_left, inputs_right, inputs_bottom, inputs_top):
        net_j1, net_r1, net_j2, net_r2 = sol
        x_left, y_left, theta_left = inputs_left
        x_right, y_right, theta_right = inputs_right
        x_bottom, y_bottom, theta_bottom = inputs_bottom
        x_top, y_top, theta_top = inputs_top

        r1_left = self.model_r1(net_r1, inputs_left)
        j1_left = self.model_j1(net_j1, inputs_left)
        r2_left = self.model_r2(net_r2, inputs_left)
        j2_left = self.model_j2(net_j2, inputs_left)
        r1_right = self.model_r1(net_r1, inputs_right)
        j1_right = self.model_j1(net_j1, inputs_right)
        r2_right = self.model_r2(net_r2, inputs_right)
        j2_right = self.model_j2(net_j2, inputs_right)
        r1_bottom = self.model_r1(net_r1, inputs_bottom)
        j1_bottom = self.model_j1(net_j1, inputs_bottom)
        r2_bottom = self.model_r2(net_r2, inputs_bottom)
        j2_bottom = self.model_j2(net_j2, inputs_bottom)
        r1_top = self.model_r1(net_r1, inputs_top)
        j1_top = self.model_j1(net_j1, inputs_top)
        r2_top = self.model_r2(net_r2, inputs_top)
        j2_top = self.model_j2(net_j2, inputs_top)

        pts_left1 = r1_left + self.kn * j1_left  # [-pi/2,0]
        pts_left2 = r2_left + self.kn * j2_left  # [0,pi/2]
        pts_right1 = r1_right - self.kn * j1_right  # [pi/2,pi]
        pts_right2 = r2_right - self.kn * j2_right  # [-pi, -pi/2]
        pts_bottom1 = r2_bottom + self.kn * j2_bottom  # [0, pi/2]
        pts_bottom2 = r1_bottom - self.kn * j1_bottom  # [pi/2, pi]
        pts_top2 = r2_top - self.kn * j2_top  # [-pi, -pi/2]
        pts_top1 = r1_top + self.kn * j1_top  # [-pi/2, 0]

        bc1_left = pts_left1 - self.bdy_left(y_left)
        bc2_left = pts_left2 - self.bdy_left(y_left)
        bc1_right = pts_right1 - self.bdy_right(y_right)
        bc2_right = pts_right2 - self.bdy_right(y_right)
        bc1_bottom = pts_bottom1 - self.bdy_bottom(x_bottom)
        bc2_bottom = pts_bottom2 - self.bdy_bottom(x_bottom)
        bc1_top = pts_top1 - self.bdy_top(x_top)
        bc2_top = pts_top2 - self.bdy_top(x_top)

        return (
            bc1_left,
            bc2_left,
            bc1_right,
            bc2_right,
            bc1_bottom,
            bc2_bottom,
            bc1_top,
            bc2_top,
        )


class OERTEHole:
    def __init__(self, config):
        self.device = torch.device(
            f"cuda:{config.model.device_ids[0]}" if torch.cuda.is_available(
            ) else "cpu"
        )

        self.kn = config.rte.kn
        self.freq = config.rte.freq
        self.sigma_a = config.rte.sigma_a
        self.sigma_s = config.rte.sigma_s

        self.xmin, self.xmax = config.domain["x"]
        self.ymin, self.ymax = config.domain["y"]
        self.thetamin, self.thetamax = config.domain["theta"]

        self.bdy_left = config.rte.bdy_left
        self.bdy_right = config.rte.bdy_right
        self.bdy_bottom = config.rte.bdy_bottom
        self.bdy_top = config.rte.bdy_top
        self.bdy_hole_left = config.rte.bdy_hole_left
        self.bdy_hole_right = config.rte.bdy_hole_right
        self.bdy_hole_bottom = config.rte.bdy_hole_bottom
        self.bdy_hole_top = config.rte.bdy_hole_top
        self.source = config.rte.source

        self.num_vquads = config.rte.num_vquads
        vquads, wquads = np.polynomial.legendre.leggauss(self.num_vquads)
        vquads = 0.5 * (vquads + 1.0) * (self.thetamax -
                                         self.thetamin) + self.thetamin
        wquads = 0.5 * (self.thetamax - self.thetamin) * wquads
        self.vquads = torch.tensor(vquads, dtype=torch.float32).to(self.device)
        self.wquads = torch.tensor(wquads, dtype=torch.float32).to(self.device)

    def model_r1(self, net_r1, inputs):
        x, y, theta = inputs
        r11 = net_r1(torch.cat([x, y, -theta], dim=-1))
        r12 = net_r1(torch.cat([x, y, -theta + torch.pi], dim=-1))
        return 0.5 * (r11 + r12)

    def model_j1(self, net_j1, inputs):
        x, y, theta = inputs
        j11 = net_j1(torch.cat([x, y, -theta], dim=-1))
        j12 = net_j1(torch.cat([x, y, -theta + torch.pi], dim=-1))
        return 0.5 * (j11 - j12)

    def model_r2(self, net_r2, inputs):
        x, y, theta = inputs
        r21 = net_r2(torch.cat([x, y, theta], dim=-1))
        r22 = net_r2(torch.cat([x, y, theta + torch.pi], dim=-1))
        return 0.5 * (r21 + r22)

    def model_j2(self, net_j2, inputs):
        x, y, theta = inputs
        j21 = net_j2(torch.cat([x, y, theta], dim=-1))
        j22 = net_j2(torch.cat([x, y, theta + torch.pi], dim=-1))
        return 0.5 * (j21 - j22)

    def v_average(self, g, inputs):
        x, y, _ = inputs
        Nx = x.shape[0]
        Nv = self.num_vquads

        x_rep = x[:, None, :].repeat(1, Nv, 1)
        y_rep = y[:, None, :].repeat(1, Nv, 1)
        theta_rep = self.vquads[None, :, None].repeat(Nx, 1, 1)

        g_val = g((x_rep, y_rep, theta_rep))  # [Nx, Nv, 1]
        avg = torch.sum(g_val * self.wquads[None, :, None], dim=1) / (
            self.thetamax - self.thetamin
        )

        return avg

    def avg_vdj1(self, model_j, net_j, x, y):
        """
        compute < v * ∂_x j >
        """
        Nx = x.shape[0]
        Nv = self.num_vquads

        x_rep = x[:, None, :].repeat(1, Nv, 1)  # [Nx, Nv, 1]
        y_rep = y[:, None, :].repeat(1, Nv, 1)
        theta_rep = self.vquads[None, :, None].repeat(Nx, 1, 1)

        x_rep.requires_grad_(True)
        y_rep.requires_grad_(True)

        j_rep = model_j(net_j, (x_rep, y_rep, theta_rep))  # j(x,v_quad)

        dj_dx_rep = torch.autograd.grad(
            j_rep,
            x_rep,
            grad_outputs=torch.ones_like(j_rep),
            retain_graph=True,
            create_graph=True,
        )[0]

        dj_dy_rep = torch.autograd.grad(
            j_rep,
            y_rep,
            grad_outputs=torch.ones_like(j_rep),
            retain_graph=True,
            create_graph=True,
        )[0]

        vdj_rep = torch.cos(theta_rep) * dj_dx_rep - \
            torch.sin(theta_rep) * dj_dy_rep

        avg_vdj = torch.sum(vdj_rep * self.wquads[None, :, None], dim=1) / (
            self.thetamax - self.thetamin
        )

        return avg_vdj

    def avg_vdj2(self, model_j, net_j, x, y):
        """
        compute < v * ∂_x j >
        """
        Nx = x.shape[0]
        Nv = self.num_vquads

        x_rep = x[:, None, :].repeat(1, Nv, 1)  # [Nx, Nv, 1]
        y_rep = y[:, None, :].repeat(1, Nv, 1)
        theta_rep = self.vquads[None, :, None].repeat(Nx, 1, 1)

        x_rep.requires_grad_(True)
        y_rep.requires_grad_(True)

        j_rep = model_j(net_j, (x_rep, y_rep, theta_rep))  # j(x,v_quad)

        dj_dx_rep = torch.autograd.grad(
            j_rep,
            x_rep,
            grad_outputs=torch.ones_like(j_rep),
            retain_graph=True,
            create_graph=True,
        )[0]

        dj_dy_rep = torch.autograd.grad(
            j_rep,
            y_rep,
            grad_outputs=torch.ones_like(j_rep),
            retain_graph=True,
            create_graph=True,
        )[0]

        vdj_rep = torch.cos(theta_rep) * dj_dx_rep + \
            torch.sin(theta_rep) * dj_dy_rep

        avg_vdj = torch.sum(vdj_rep * self.wquads[None, :, None], dim=1) / (
            self.thetamax - self.thetamin
        )

        return avg_vdj

    def residual(self, sol, inputs):
        net_j1, net_r1, net_j2, net_r2 = sol
        x, y, theta = inputs
        xi, eta = torch.cos(theta), torch.sin(theta)
        x.requires_grad_(True)
        y.requires_grad_(True)

        j1 = self.model_j1(net_j1, inputs)
        r1 = self.model_r1(net_r1, inputs)
        j2 = self.model_j2(net_j2, inputs)
        r2 = self.model_r2(net_r2, inputs)
        Q1_even = 0.5 * (
            self.source(x, y, -theta) + self.source(x, y, -theta + torch.pi)
        )
        Q1_odd = 0.5 * (
            self.source(x, y, -theta) - self.source(x, y, -theta + torch.pi)
        )
        Q2_even = 0.5 * (self.source(x, y, theta) +
                         self.source(x, y, theta + torch.pi))
        Q2_odd = 0.5 * (self.source(x, y, theta) -
                        self.source(x, y, theta + torch.pi))

        dj1_dx = torch.autograd.grad(
            j1,
            x,
            grad_outputs=torch.ones_like(j1),
            retain_graph=True,
            create_graph=True,
        )[0]
        dj1_dy = torch.autograd.grad(
            j1,
            y,
            grad_outputs=torch.ones_like(j1),
            retain_graph=True,
            create_graph=True,
        )[0]
        dr1_dx = torch.autograd.grad(
            r1,
            x,
            grad_outputs=torch.ones_like(r1),
            retain_graph=True,
            create_graph=True,
        )[0]
        dr1_dy = torch.autograd.grad(
            r1,
            y,
            grad_outputs=torch.ones_like(r1),
            retain_graph=True,
            create_graph=True,
        )[0]
        dj2_dx = torch.autograd.grad(
            j2,
            x,
            grad_outputs=torch.ones_like(j2),
            retain_graph=True,
            create_graph=True,
        )[0]
        dj2_dy = torch.autograd.grad(
            j2,
            y,
            grad_outputs=torch.ones_like(j2),
            retain_graph=True,
            create_graph=True,
        )[0]
        dr2_dx = torch.autograd.grad(
            r2,
            x,
            grad_outputs=torch.ones_like(r2),
            retain_graph=True,
            create_graph=True,
        )[0]
        dr2_dy = torch.autograd.grad(
            r2,
            y,
            grad_outputs=torch.ones_like(r2),
            retain_graph=True,
            create_graph=True,
        )[0]

        vdj1 = xi * dj1_dx - eta * dj1_dy
        vdj2 = xi * dj2_dx + eta * dj2_dy
        vdr1 = xi * dr1_dx - eta * dr1_dy
        vdr2 = xi * dr2_dx + eta * dr2_dy

        avg_vdj1 = self.avg_vdj1(self.model_j1, net_j1, x, y)
        avg_r1 = self.v_average(lambda inp: self.model_r1(net_r1, inp), inputs)
        avg_vdj2 = self.avg_vdj2(self.model_j2, net_j2, x, y)
        avg_r2 = self.v_average(lambda inp: self.model_r2(net_r2, inp), inputs)
        avg_Q1 = self.v_average(
            lambda inp: (
                0.5
                * (
                    self.source(inp[0], inp[1], -inp[2])
                    + self.source(inp[0], inp[1], -inp[2] + torch.pi)
                )
            ),
            inputs,
        )
        avg_Q2 = self.v_average(
            lambda inp: (
                0.5
                * (
                    self.source(inp[0], inp[1], inp[2])
                    + self.source(inp[0], inp[1], inp[2] + torch.pi)
                )
            ),
            inputs,
        )

        res1 = (
            avg_vdj1
            + self.sigma_a(x) * avg_r1
            + avg_vdj2
            + self.sigma_a(x) * avg_r2
            - avg_Q1
            - avg_Q2
        )
        res2 = (
            self.kn**2 * (vdj1 - avg_vdj1)
            + (self.sigma_s(x) + self.kn**2 * self.sigma_a(x)) * (r1 - avg_r1)
            - self.kn**2 * (Q1_even - avg_Q1)
            + self.kn**2 * (vdj2 - avg_vdj2)
            + (self.sigma_s(x) + self.kn**2 * self.sigma_a(x)) * (r2 - avg_r2)
            - self.kn**2 * (Q2_even - avg_Q2)
        )
        res3 = (
            (self.sigma_s(x) + self.kn**2 * self.sigma_a(x)) * j1
            + vdr1
            - self.kn * Q1_odd
            + (self.sigma_s(x) + self.kn**2 * self.sigma_a(x)) * j2
            + vdr2
            - self.kn * Q2_odd
        )
        res4 = (
            self.kn**2 * vdj1
            + (self.sigma_s(x) + self.kn**2 * self.sigma_a(x)) * r1
            - self.kn**2 * Q1_even
            - self.kn**2 * vdj2
            - (self.sigma_s(x) + self.kn**2 * self.sigma_a(x)) * r2
            + self.kn**2 * Q2_even
        )
        res5 = (
            (self.sigma_s(x) + self.kn**2 * self.sigma_a(x)) * j1
            + vdr1
            - self.kn * Q1_odd
            - (self.sigma_s(x) + self.kn**2 * self.sigma_a(x)) * j2
            - vdr2
            + self.kn * Q2_odd
        )

        return res1, res2, res3, res4, res5

    def bc(self, sol, inputs_left, inputs_right, inputs_bottom, inputs_top):
        net_j1, net_r1, net_j2, net_r2 = sol
        x_left, y_left, theta_left = inputs_left
        x_right, y_right, theta_right = inputs_right
        x_bottom, y_bottom, theta_bottom = inputs_bottom
        x_top, y_top, theta_top = inputs_top

        r1_left = self.model_r1(net_r1, inputs_left)
        j1_left = self.model_j1(net_j1, inputs_left)
        r2_left = self.model_r2(net_r2, inputs_left)
        j2_left = self.model_j2(net_j2, inputs_left)
        r1_right = self.model_r1(net_r1, inputs_right)
        j1_right = self.model_j1(net_j1, inputs_right)
        r2_right = self.model_r2(net_r2, inputs_right)
        j2_right = self.model_j2(net_j2, inputs_right)
        r1_bottom = self.model_r1(net_r1, inputs_bottom)
        j1_bottom = self.model_j1(net_j1, inputs_bottom)
        r2_bottom = self.model_r2(net_r2, inputs_bottom)
        j2_bottom = self.model_j2(net_j2, inputs_bottom)
        r1_top = self.model_r1(net_r1, inputs_top)
        j1_top = self.model_j1(net_j1, inputs_top)
        r2_top = self.model_r2(net_r2, inputs_top)
        j2_top = self.model_j2(net_j2, inputs_top)

        pts_left1 = r1_left + self.kn * j1_left  # [-pi/2,0]
        pts_left2 = r2_left + self.kn * j2_left  # [0,pi/2]
        pts_right1 = r1_right - self.kn * j1_right  # [pi/2,pi]
        pts_right2 = r2_right - self.kn * j2_right  # [-pi, -pi/2]
        pts_bottom1 = r2_bottom + self.kn * j2_bottom  # [0, pi/2]
        pts_bottom2 = r1_bottom - self.kn * j1_bottom  # [pi/2, pi]
        pts_top2 = r2_top - self.kn * j2_top  # [-pi, -pi/2]
        pts_top1 = r1_top + self.kn * j1_top  # [-pi/2, 0]

        bc1_left = pts_left1 - self.bdy_left(y_left)
        bc2_left = pts_left2 - self.bdy_left(y_left)
        bc1_right = pts_right1 - self.bdy_right(y_right)
        bc2_right = pts_right2 - self.bdy_right(y_right)
        bc1_bottom = pts_bottom1 - self.bdy_bottom(x_bottom)
        bc2_bottom = pts_bottom2 - self.bdy_bottom(x_bottom)
        bc1_top = pts_top1 - self.bdy_top(x_top)
        bc2_top = pts_top2 - self.bdy_top(x_top)

        return (
            bc1_left,
            bc2_left,
            bc1_right,
            bc2_right,
            bc1_bottom,
            bc2_bottom,
            bc1_top,
            bc2_top,
        )

    def holebc(self, sol, inputs_left, inputs_right, inputs_bottom, inputs_top):
        net_j1, net_r1, net_j2, net_r2 = sol
        x_left, y_left, theta_left = inputs_left
        x_right, y_right, theta_right = inputs_right
        x_bottom, y_bottom, theta_bottom = inputs_bottom
        x_top, y_top, theta_top = inputs_top

        r1_left = self.model_r1(net_r1, inputs_left)
        j1_left = self.model_j1(net_j1, inputs_left)
        r2_left = self.model_r2(net_r2, inputs_left)
        j2_left = self.model_j2(net_j2, inputs_left)
        r1_right = self.model_r1(net_r1, inputs_right)
        j1_right = self.model_j1(net_j1, inputs_right)
        r2_right = self.model_r2(net_r2, inputs_right)
        j2_right = self.model_j2(net_j2, inputs_right)
        r1_bottom = self.model_r1(net_r1, inputs_bottom)
        j1_bottom = self.model_j1(net_j1, inputs_bottom)
        r2_bottom = self.model_r2(net_r2, inputs_bottom)
        j2_bottom = self.model_j2(net_j2, inputs_bottom)
        r1_top = self.model_r1(net_r1, inputs_top)
        j1_top = self.model_j1(net_j1, inputs_top)
        r2_top = self.model_r2(net_r2, inputs_top)
        j2_top = self.model_j2(net_j2, inputs_top)

        pts_left1 = r1_left + self.kn * j1_left  # [-pi/2,0]
        pts_left2 = r2_left + self.kn * j2_left  # [0,pi/2]
        pts_right1 = r1_right - self.kn * j1_right  # [pi/2,pi]
        pts_right2 = r2_right - self.kn * j2_right  # [-pi, -pi/2]
        pts_bottom1 = r2_bottom + self.kn * j2_bottom  # [0, pi/2]
        pts_bottom2 = r1_bottom - self.kn * j1_bottom  # [pi/2, pi]
        pts_top2 = r2_top - self.kn * j2_top  # [-pi, -pi/2]
        pts_top1 = r1_top + self.kn * j1_top  # [-pi/2, 0]

        bc1_left = pts_left1 - self.bdy_hole_left(y_left)
        bc2_left = pts_left2 - self.bdy_hole_left(y_left)
        bc1_right = pts_right1 - self.bdy_hole_right(y_right)
        bc2_right = pts_right2 - self.bdy_hole_right(y_right)
        bc1_bottom = pts_bottom1 - self.bdy_hole_bottom(x_bottom)
        bc2_bottom = pts_bottom2 - self.bdy_hole_bottom(x_bottom)
        bc1_top = pts_top1 - self.bdy_hole_top(x_top)
        bc2_top = pts_top2 - self.bdy_hole_top(x_top)

        return (
            bc1_left,
            bc2_left,
            bc1_right,
            bc2_right,
            bc1_bottom,
            bc2_bottom,
            bc1_top,
            bc2_top,
        )
