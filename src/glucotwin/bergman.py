"""Bergman Minimal Model as a differentiable PyTorch module."""
import torch
import torch.nn as nn


class BergmanMinimalModel(nn.Module):
    """Bergman's minimal model of glucose-insulin dynamics.

    Implements the three-ODE system with learnable per-patient parameters,
    integrated via 4th-order Runge-Kutta.

    dG/dt = -p1*(G - Gb) - X*G + D(t)
    dX/dt = -p2*X + p3*(I - Ib)
    dI/dt = -n*I + gamma*max(0, G-h) + u(t)
    """

    def __init__(self, dt=5.0):
        super().__init__()
        self.dt = dt  # minutes

        # Population priors (log-space for positivity)
        self.log_p1 = nn.Parameter(torch.tensor(-4.0))     # ~0.018
        self.log_p2 = nn.Parameter(torch.tensor(-3.5))     # ~0.030
        self.log_p3 = nn.Parameter(torch.tensor(-8.0))     # ~3.4e-4
        self.log_n = nn.Parameter(torch.tensor(-2.0))       # ~0.135
        self.log_gamma = nn.Parameter(torch.tensor(-5.5))   # ~0.004
        self.Gb = nn.Parameter(torch.tensor(120.0))
        self.Ib = nn.Parameter(torch.tensor(7.0))
        self.h = nn.Parameter(torch.tensor(90.0))

    @property
    def params(self):
        return {
            'p1': torch.exp(self.log_p1),
            'p2': torch.exp(self.log_p2),
            'p3': torch.exp(self.log_p3),
            'n': torch.exp(self.log_n),
            'gamma': torch.exp(self.log_gamma),
            'Gb': self.Gb,
            'Ib': self.Ib,
            'h': self.h,
        }

    def _ode_rhs(self, state, meal_rate, insulin_rate, p):
        G, X, I = state[..., 0], state[..., 1], state[..., 2]

        dG = -p['p1'] * (G - p['Gb']) - X * G + meal_rate
        dX = -p['p2'] * X + p['p3'] * (I - p['Ib'])
        dI = -p['n'] * I + p['gamma'] * torch.relu(G - p['h']) + insulin_rate

        return torch.stack([dG, dX, dI], dim=-1)

    def _rk4_step(self, state, meal_rate, insulin_rate, p):
        k1 = self._ode_rhs(state, meal_rate, insulin_rate, p)
        k2 = self._ode_rhs(state + 0.5 * self.dt * k1, meal_rate, insulin_rate, p)
        k3 = self._ode_rhs(state + 0.5 * self.dt * k2, meal_rate, insulin_rate, p)
        k4 = self._ode_rhs(state + self.dt * k3, meal_rate, insulin_rate, p)
        return state + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def forward(self, G0, meal_rates, insulin_rates, horizon):
        """Integrate forward from initial glucose G0.

        Args:
            G0: (batch,) initial glucose
            meal_rates: (batch, horizon) meal glucose appearance rate
            insulin_rates: (batch, horizon) exogenous insulin rate
            horizon: number of steps to predict

        Returns:
            G_pred: (batch, horizon) predicted glucose trajectory
        """
        p = self.params
        batch = G0.shape[0]
        device = G0.device

        X0 = torch.zeros(batch, device=device)
        I0 = p['Ib'].expand(batch)
        state = torch.stack([G0, X0, I0], dim=-1)

        predictions = []
        for t in range(horizon):
            m_t = meal_rates[:, t] if meal_rates is not None else torch.zeros(batch, device=device)
            u_t = insulin_rates[:, t] if insulin_rates is not None else torch.zeros(batch, device=device)
            state = self._rk4_step(state, m_t, u_t, p)
            state = torch.clamp(state, min=0.0)
            predictions.append(state[..., 0])

        return torch.stack(predictions, dim=1)

    def ode_residual(self, G_pred, X_pred, I_pred, meal_rates, insulin_rates):
        """Compute physics residual for PINN loss."""
        p = self.params
        dG_dt = (G_pred[:, 1:] - G_pred[:, :-1]) / self.dt

        G = G_pred[:, :-1]
        X = X_pred[:, :-1] if X_pred is not None else torch.zeros_like(G)
        I = I_pred[:, :-1] if I_pred is not None else p['Ib'].expand_as(G)
        m = meal_rates[:, :-1] if meal_rates is not None else torch.zeros_like(G)

        rhs = -p['p1'] * (G - p['Gb']) - X * G + m
        return torch.mean((dG_dt - rhs) ** 2)
