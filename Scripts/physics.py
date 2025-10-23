import torch
import torch.autograd as autograd
from NN_Learner import neural, ParameterLearner, BatteryEmbedding

NUM_BATTERIES = 4  # {B0005,B0006,B0007,B0018}
EMB_DIM = 6

# Global instances
Emb = BatteryEmbedding(num_batteries=NUM_BATTERIES, emb_dim=EMB_DIM)
ANN = neural(emb_dim=EMB_DIM)
Param = ParameterLearner(emb_dim=EMB_DIM)

"""
Corrected Arrhenius equation for battery capacity degradation:
dC/dt = -k * (I ** n) * exp(-Ea / (R * T))

Units clarification:
- C: Capacity (Ah) or normalized capacity (0-1)
- I: Current magnitude (A) 
- T: Temperature (K)
- t: Normalized cycle index (0-1) per battery
- k: Effective per-cycle rate constant (units: [C]/cycle/A^n)
- n: Reaction order (dimensionless, typically 0.5-2.0)
- Ea: Activation energy (J/mol, typically 20k-100k)
- R: Universal gas constant (8.314 J/(mol*K))
"""

R = 8.314  # J/(mol*K)

def ArrheniusRHS(T, I, params):
    """Right-hand side of Arrhenius degradation equation."""
    I_mag = torch.abs(I)  # Handle negative discharge currents
    k, n, Ea = params[:, 0:1], params[:, 1:2], params[:, 2:3]
    # Compute degradation rate
    arrhenius_term = torch.exp(-Ea / (R * T))
    return -k * (I_mag ** n) * arrhenius_term, arrhenius_term, I_mag

def ArrheniusLHS(t, T, I, emb):
    """Left-hand side: compute dC/dt via autograd."""
    C = ANN(t, T, I, emb)
    dCdt = autograd.grad(C, t, grad_outputs=torch.ones_like(C), create_graph=True)[0]
    return dCdt, C

def ArrheniusLoss(t, T_C, T_C_norm, I, bat_idx, stabilize_early=True):
    """
    Physics-informed loss combining ODE residual and data supervision.
    Args:
        stabilize_early: detach C when feeding to parameter learner (disable after epoch 100)
    """
    T_K = T_C + 273.15
    emb = Emb(bat_idx)
    dCdt, C_pred = ArrheniusLHS(t, T_C_norm, I, emb)
    
    # ✅ Only detach if stabilize_early=True
    C_for_params = C_pred.detach() if stabilize_early else C_pred
    
    params = Param(C_for_params, emb)
    rhs, arrhenius_term, I_mag = ArrheniusRHS(T_K, I, params)
    relative_residual = (dCdt - rhs) / (torch.abs(dCdt) + 1e-6)
    physics_loss = (relative_residual ** 2).mean()
    return physics_loss, C_pred, params, rhs, dCdt, arrhenius_term, I_mag

def combined_loss(t, T_C, T_C_norm, I, bat_idx, C_target, lambda_phys=1.0, lambda_data=1.0, epoch=0):
    """
    ✅ Added epoch parameter to control detachment
    """
    stabilize_early = epoch < 100  # Stop detaching after epoch 100
    physics_loss, C_pred, params, rhs, dCdt, arrhenius_term, I_mag = ArrheniusLoss(t, T_C, T_C_norm, I, bat_idx, stabilize_early)
    data_loss = torch.nn.MSELoss()(C_pred, C_target)
    physics_loss_scaled = physics_loss * (data_loss.detach() / (physics_loss.detach() + 1e-8))
    total_loss = lambda_phys * physics_loss_scaled + lambda_data * data_loss
    return total_loss, params, physics_loss, data_loss, rhs, dCdt, arrhenius_term, I_mag
