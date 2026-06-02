import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader
from matplotlib import cm
import os
import time
import csv  # profiling
import pickle

# a manually defined activation function fixed for the output layer for all MLPs
class SoftplusReLU(nn.Module):
    '''
    Modified Softplus activation function where large values 
    are ReLU activated to prevent floating point blowup.
    Args:
        threshold: scalar float for Softplus/ReLU cutoff
    Inputs:
        x: torch float tensor of inputs
    Returns:
        x: torch float tensor of outputs
    '''
    def __init__(self, threshold=20.0):
        super().__init__()
        self.threshold = threshold
        self.softplus = nn.Softplus()
        self.relu = nn.ReLU()
    def forward(self, x):
        # Softplus for small values, ReLU for large
        x = torch.where(x < 20.0, 
                        self.softplus(x), 
                        self.relu(x))
        return x

# MLPs for u, D
class UNet(nn.Module):
    def __init__(self, num_layers, hidden_units, activation_fn, input_dim=2, output_dim=1): 
        super(UNet, self).__init__()

        self.num_layers = num_layers
        self.hidden_units = hidden_units
        self.activation_fn = activation_fn
        
        if hidden_units * num_layers == 1.0:
            # minimal model: one node only NN
            self.hidden_layer = nn.Linear(input_dim, output_dim)
        else:
            # fully connected feedforward MLP
            layers = []
            # Input layer
            layers.append(nn.Linear(input_dim, hidden_units))
            layers.append(activation_fn) 
            # Hidden layers
            for _ in range(num_layers - 1):  # Subtract 1 because we already created the first hidden layer
                layers.append(nn.Linear(hidden_units, hidden_units))
                layers.append(activation_fn)
            # Output layer
            layers.append(nn.Linear(hidden_units, output_dim))
            layers.append(SoftplusReLU()) # specific activation (task-dependent) for output layer 
            # Combine all layers into a single Sequential block
            self.network = nn.Sequential(*layers)

        # Apply weight initialization
        self._initialize_weights()

    def forward(self, x):
        if self.hidden_units * self.num_layers == 1.0:
            # minimal model: one node only NN
            return self.activation_fn(self.hidden_layer(x))
        else:
            # fully connected feedforward MLP
            return self.network(x)
        
    def _initialize_weights(self):
        if hasattr(self, "network"):
            for m in self.network:
                if isinstance(m, nn.Linear):
                    init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        init.zeros_(m.bias)
        else:
            init.xavier_normal_(self.hidden_layer.weight)
            if self.hidden_layer.bias is not None:
                init.zeros_(self.hidden_layer.bias)

class DNet(nn.Module):
    def __init__(self, num_layers, hidden_units, activation_fn, input_dim=1, output_dim=1): 
        super(DNet, self).__init__()

        self.num_layers = num_layers
        self.hidden_units = hidden_units
        self.activation_fn = activation_fn
        
        if hidden_units * num_layers == 1.0:
            # minimal model: one node only NN
            self.hidden_layer = nn.Linear(input_dim, output_dim)
        else:
            # fully connected feedforward MLP
            layers = []
            # Input layer
            layers.append(nn.Linear(input_dim, hidden_units))
            layers.append(activation_fn) 
            # Hidden layers
            for _ in range(num_layers - 1):  # Subtract 1 because we already created the first hidden layer
                layers.append(nn.Linear(hidden_units, hidden_units))
                layers.append(activation_fn)
            # Output layer
            layers.append(nn.Linear(hidden_units, output_dim))
            layers.append(SoftplusReLU()) # specific activation (task-dependent) for output layer 
            # Combine all layers into a single Sequential block
            self.network = nn.Sequential(*layers)

        # Apply weight initialization
        self._initialize_weights()

    def forward(self, x):
        if self.hidden_units * self.num_layers == 1.0:
            # minimal model: one node only NN
            return self.activation_fn(self.hidden_layer(x))
        else:
            # fully connected feedforward MLP
            return self.network(x)
        
    def _initialize_weights(self):
        if hasattr(self, "network"):
            for m in self.network:
                if isinstance(m, nn.Linear):
                    init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        init.zeros_(m.bias)
        else:
            init.xavier_normal_(self.hidden_layer.weight)
            if self.hidden_layer.bias is not None:
                init.zeros_(self.hidden_layer.bias)

# Autograd functions to compute derivatives
def compute_du_dxdt(u, x_input):
    grads = torch.autograd.grad(
        outputs=u, inputs=x_input,
        grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True
    )[0]
    du_dx = grads[:, 0:1] # ∂u/∂x is the derivative w.r.t. the first input
    du_dt = grads[:, 1:2] # ∂u/∂t is the derivative w.r.t. the second input
    return du_dx, du_dt

def compute_J_dx(u, d, x_input):
    du_dx, _ = compute_du_dxdt(u, x_input)
    Jacob = d * du_dx

    grads = torch.autograd.grad(
        outputs=Jacob, inputs=x_input,
        grad_outputs=torch.ones_like(Jacob),
        create_graph=True, retain_graph=True
    )[0]
    J_x = grads[:, 0:1] # ∂J/∂x is the derivative w.r.t. the first input
    return J_x

# 1) Data loss for u (MSE)
def data_loss_Gaussian(u_pred, u_batch):        
    u_loss = torch.mean((u_pred - u_batch) ** 2)
    return u_loss

# 2) PDE loss for for diffusion equation u_t = (D(u)u_x0)_x
def pde_loss_Diffusion(x_max, x_min, t_max, t_min, device, u_net, D_net, N_PDE):
    # N_PDE: number of virtual points in the domain to be sampled to calcualte the PDE loss using minibatch method
    # NOTE: when sampling, we need to exclude the boundaries x_min and x_max, but includes the boundaries t_min=0 and t_max

    t0 = time.perf_counter()
    # sample uniformly in the range [x_min + eps, x_max - eps] to exclude x_min and x_max
    x = torch.rand(N_PDE, 1, requires_grad=True) 
    eps = 1e-10
    x = x * (x_max - x_min - 2*eps) + (x_min + eps) 
    # BUG: we have not included t_max when sampling t -- however results won't be affected much as t_max is just one point.
    t = torch.rand(N_PDE, 1, requires_grad=True)
    t = t * (t_max - t_min) + t_min
    
    # forward pass through the networks u and D
    inputs_rand = torch.cat([x, t], dim=1).float().to(device)
    u_pred_rand = u_net(inputs_rand)           
    D_pred_rand = D_net(u_pred_rand)
    t1 = time.perf_counter()
    t_pde_forward = t1 - t0

    # PDE constraint loss: 
    t0 = time.perf_counter()
    _, du_dt = compute_du_dxdt(u_pred_rand, inputs_rand)
    J_x = compute_J_dx(u_pred_rand, D_pred_rand, inputs_rand)
    pde_loss = torch.mean((du_dt - J_x)**2)
    t1 = time.perf_counter()
    t_pde = t1 - t0
    
    return pde_loss, t_pde_forward, t_pde

# 3) Boundary loss for Neumann BCs: du/dx=0 at x=x_min and x=x_max
def boundary_loss_neumann(x_min, x_max, device, u_net, t_min, t_max, N_BC):
    # N_BC: number of virtual time points to sample at each boundary to calcualte the boundary loss using minibatch method

    t0 = time.perf_counter()
    t_left, t_right = torch.rand(N_BC, 1, requires_grad=True), torch.rand(N_BC, 1, requires_grad=True)
    t_left, t_right = t_left * (t_max - t_min) + t_min, t_right * (t_max - t_min) + t_min
    x_left, x_right = torch.full((N_BC, 1), x_min, dtype=torch.float32), torch.full((N_BC, 1), x_max, dtype=torch.float32)
    bc_left, bc_right = torch.cat([x_left, t_left], dim=1).to(device).requires_grad_(), torch.cat([x_right, t_right], dim=1).to(device).requires_grad_()

    # forward pass through the network u
    u_pred_left= u_net(bc_left)
    u_pred_right = u_net(bc_right)    
    t1 = time.perf_counter()
    t_bc_forward = t1 - t0

    t0 = time.perf_counter()
    du_dx_left, _ = compute_du_dxdt(u_pred_left, bc_left)
    du_dx_right, _ = compute_du_dxdt(u_pred_right, bc_right)
    neumann_bc_loss = torch.mean(du_dx_left**2) + torch.mean(du_dx_right**2)
    t1 = time.perf_counter()
    t_bc = t1 - t0

    return neumann_bc_loss, t_bc_forward, t_bc

# 4) biological constraint loss -- might only used in the Fisher-KPP case
def biology_loss(x_max, x_min, t_min, t_max, device, u_net, D_net, N_bio):
    # N_bio: number of grid points in the domain to be sampled to calcualte the biological constraint loss using minibatch method
    x = torch.rand(N_bio, 1, requires_grad=True) 
    x = x * (x_max - x_min) + x_min
    t = torch.rand(N_bio, 1, requires_grad=True)
    t = t * (t_max - t_min) + t_min

    # forward pass through the networks u, D
    inputs_rand = torch.cat([x, t], dim=1).float().to(device)
    u_pred_rand = u_net(inputs_rand)
    D_pred_rand = D_net(u_pred_rand)

    # biological constraint loss: u in [0, 1], D(u) >= 0
    #u_bio_loss = torch.where(u_pred_rand < 0.0, u_pred_rand**2, torch.zeros_like(u_pred_rand)) + \
    #                torch.where(u_pred_rand > 1.0, (u_pred_rand - 1.0)**2, torch.zeros_like(u_pred_rand))
    u_bio_loss = torch.where(u_pred_rand <= 0.0, u_pred_rand**2, torch.zeros_like(u_pred_rand))
    D_bio_loss = torch.where(D_pred_rand <= 0.0, D_pred_rand**2, torch.zeros_like(D_pred_rand))

    return torch.mean(u_bio_loss), torch.mean(D_bio_loss)

# -------------------------------------- model performance evaluation --------------------------------------
def process_D(D, len):
    """
    ensure D is a NumPy array of length len(U_grid)
    """
    if np.isscalar(D):
        return np.full(len, D)
    elif isinstance(D, np.ndarray):
        if D.shape[0] != len:
            raise ValueError(f"D must be of length len, but got shape {D.shape}")
        return D
    else:
        raise TypeError("D must be a scalar or a NumPy array")   

# 1) plot the learnt u, D
def learnt_u_D_plot(Nt, x_train, u_true, test_u_pred, D, U_grid, test_D_pred_np, u_min, u_max, save_dir, savename):
    # plot the learnt u, D, and V for each repetition's best model
    fig1, axes = plt.subplots(1, 2, figsize=(16, 5))

    # ensure the true value for D is a NumPy array of length len(U_grid)
    D = process_D(D, len(U_grid)) 
    
    # 1). learnt u
    for t in range(Nt):
        axes[0].plot(x_train, u_true[t,:], linestyle='--', label=r'True $u(x, t)$', linewidth=3, color='black') # true u
        axes[0].plot(x_train, test_u_pred[t,:], linewidth=3, alpha=0.5, color='blue') # learnt u
    axes[0].set_title(rf'Learnt $u_{{\theta}}(x,t)$')
    axes[0].set_xlabel(r'$x$')
    axes[0].set_ylabel(r'$u_{{\theta}}$')
    axes[0].set_ylim(-0.1, 1.1)

    # 2). learnt D 
    axes[1].plot(U_grid, D, color='black', linestyle='--', label=r'True $D(u)$', linewidth=3) # true D
    axes[1].plot(U_grid, test_D_pred_np, linewidth=3, alpha=0.5, color='blue') # learnt D
    axes[1].set_title(rf'Learnt $D_{{\phi}}(u)$')
    axes[1].set_xlabel(r'$u$')
    axes[1].set_ylabel(r'$D_{\phi}(u)$')
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].set_xlim(u_min, u_max)
    axes[1].legend()

    plt.tight_layout()
    fig1.savefig(f'{save_dir}{savename}', dpi=300)
    pass

# 2) relative error of the learnt u, D
def u_D_relerr(test_u_pred, u_true, X_grid, T_grid, t_train, x_train, test_D_pred_np, D, U_grid, save_dir, savename):

    # ensure the true value for D is a NumPy array of length len(U_grid)
    D = process_D(D, len(U_grid)) 

    fig2 = plt.figure(figsize=(16, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])  # manually set the relative width of the subplots

    # 1). relative error of u
    u_relerrS = np.where(u_true == 0, np.nan, (test_u_pred - u_true) / u_true)
    ax1 = fig2.add_subplot(gs[0, 0], projection='3d')
    ax1.plot_surface(X_grid, T_grid, u_relerrS.reshape(len(t_train), len(x_train)), cmap=cm.viridis)
    ax1.set_xlabel(r'$x$')
    ax1.set_ylabel(r'$t$')
    ax1.set_zlabel(r'$(u_{\theta} - u)/u$')
    ax1.set_title(rf'Relative error: $(u_{{\theta}} - u)/u$')

    # 2). relative error of D
    D_relerrS = np.where(D == 0, np.nan, (test_D_pred_np - D) / D)
    ax2 = fig2.add_subplot(gs[0, 1])
    ax2.plot(U_grid, D_relerrS, linewidth=3, color='green')
    ax2.set_title(rf'Relative error: $(D_{{\phi}} - D)/D$')
    ax2.set_xlabel(r'$u$')
    ax2.set_ylabel(r'$(D_{\phi} - D)/D$')

    plt.tight_layout()
    fig2.savefig(f'{save_dir}/{savename}', dpi=300)
    pass

# 3) loss history plot for all repetitions 
def loss_plot(BCloss_S, pdeloss_S, val_BCloss_S, val_pdeloss_S, save_dir, savename, store_loss_step, \
              val_u_bio_S, val_D_bio_S, u_bio_S, D_bio_S, uloss_S, val_uloss_S, Num_epochs, bio_constraint_bool):
    # plot the training and validation loss history
    fig4 = plt.figure(figsize=(16, 5))
    epoch_history = np.arange(0, Num_epochs, store_loss_step)\
    
    # training loss
    plt.plot(epoch_history, np.nanmean(uloss_S, axis=0), color='blue', label=r'$\mathcal{L}_{\mathrm{data}}$')
    plt.fill_between(epoch_history, np.nanmean(uloss_S, axis=0) - np.nanstd(uloss_S, axis=0), np.nanmean(uloss_S, axis=0) + np.nanstd(uloss_S, axis=0), alpha=0.2, color='blue')
    plt.plot(epoch_history, np.nanmean(BCloss_S, axis=0), color='orange', label=r'$\mathcal{L}_{\mathrm{BC}}$')
    plt.fill_between(epoch_history, np.nanmean(BCloss_S, axis=0) - np.nanstd(BCloss_S, axis=0), np.nanmean(BCloss_S, axis=0) + np.nanstd(BCloss_S, axis=0), alpha=0.2, color='orange')
    plt.plot(epoch_history, np.nanmean(pdeloss_S, axis=0), color='green', label=r'$\mathcal{L}_{\mathrm{PDE}}$')
    plt.fill_between(epoch_history, np.nanmean(pdeloss_S, axis=0) - np.nanstd(pdeloss_S, axis=0), np.nanmean(pdeloss_S, axis=0) + np.nanstd(pdeloss_S, axis=0), alpha=0.2, color='green')
    if bio_constraint_bool:
        bio_total_loss = []
        for i in range(len(u_bio_S)):
            bio_total_loss.append(u_bio_S[i] + D_bio_S[i])
        bio_total_loss = np.array(bio_total_loss)
        plt.plot(epoch_history, np.nanmean(bio_total_loss, axis=0), color='purple', label=r'$\mathcal{L}_{\mathrm{bio}}$')
        plt.fill_between(epoch_history, np.nanmean(bio_total_loss, axis=0) - np.nanstd(bio_total_loss, axis=0), np.nanmean(bio_total_loss, axis=0) + np.nanstd(bio_total_loss, axis=0), alpha=0.2, color='purple')

    # validation loss
    plt.plot(epoch_history, np.nanmean(val_uloss_S, axis=0), color='blue', linestyle='--')
    plt.fill_between(epoch_history, np.nanmean(val_uloss_S, axis=0) - np.nanstd(val_uloss_S, axis=0), np.nanmean(val_uloss_S, axis=0) + np.nanstd(val_uloss_S, axis=0), alpha=0.2, color='blue', linestyle='--')
    plt.plot(epoch_history, np.nanmean(val_BCloss_S, axis=0), color='orange', linestyle='--')
    plt.fill_between(epoch_history, np.nanmean(val_BCloss_S, axis=0) - np.nanstd(val_BCloss_S, axis=0), np.nanmean(val_BCloss_S, axis=0) + np.nanstd(val_BCloss_S, axis=0), alpha=0.2, color='orange', linestyle='--')
    plt.plot(epoch_history, np.nanmean(val_pdeloss_S, axis=0), color='green', linestyle='--')
    plt.fill_between(epoch_history, np.nanmean(val_pdeloss_S, axis=0) - np.nanstd(val_pdeloss_S, axis=0), np.nanmean(val_pdeloss_S, axis=0) + np.nanstd(val_pdeloss_S, axis=0), alpha=0.2, color='green', linestyle='--')
    if bio_constraint_bool:
        val_bio_total_loss = []
        for i in range(len(val_u_bio_S)):
            val_bio_total_loss.append(val_u_bio_S[i] + val_D_bio_S[i])
        val_bio_total_loss = np.array(val_bio_total_loss)
        plt.plot(epoch_history, np.nanmean(val_bio_total_loss, axis=0), color='purple', linestyle='--')
        plt.fill_between(epoch_history, np.nanmean(val_bio_total_loss, axis=0) - np.nanstd(val_bio_total_loss, axis=0), np.nanmean(val_bio_total_loss, axis=0) + np.nanstd(val_bio_total_loss, axis=0), alpha=0.2, color='purple', linestyle='--')

    plt.yscale('log')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title('Training (solid) and validation (dashed) losses')
    plt.legend(title="Training loss")  # This adds the legend title
    plt.tight_layout()
    fig4.savefig(f'{save_dir}/{savename}', dpi=300)
    pass

# 4) overall model performance evaluation function
def model_performance_eval(save_dir, u_net, D_net, test_loader, device, t_train, x_train, U_grid_tensor, Nt, u_true, D, U_grid, u_min, u_max, X_grid, T_grid,\
                BCloss_S, pdeloss_S, val_BCloss_S, val_pdeloss_S, Numrep, store_loss_step, val_u_bio_S, val_D_bio_S, u_bio_S, D_bio_S, uloss_S, val_uloss_S, Num_epochs, \
                    bio_constraint_bool):
    
     # plot the loss history in the training process
    savename = 'loss_history.png'
    loss_plot(BCloss_S, pdeloss_S, val_BCloss_S, val_pdeloss_S, save_dir, savename, store_loss_step, \
              val_u_bio_S, val_D_bio_S, u_bio_S, D_bio_S, uloss_S, val_uloss_S, Num_epochs, bio_constraint_bool)

    # 4 averaged metrics (l2re, l1re, mse, merr) we use for performance evaluation over repetitions on test data:
    # 1) on full grid points:
    XT_grid = np.stack([X_grid.flatten(), T_grid.flatten()], axis=1)  # shape [N, 2]
    XT_tensor = torch.tensor(XT_grid, dtype=torch.float32).to(device) 
    l2re_uS_full, l2re_DS_full = np.zeros(Numrep), np.zeros(Numrep) # relative L2 error
    l1re_uS_full, l1re_DS_full = np.zeros(Numrep), np.zeros(Numrep) # relative L1 error
    mse_uS_full, mse_DS_full = np.zeros(Numrep), np.zeros(Numrep)    # mean squared error
    merr_uS_full, merr_DS_full = np.zeros(Numrep), np.zeros(Numrep) # maximum absolute error
    # 2) on test data:
    num_batches = len(test_loader) # to average the metrics over batch numbers
    l2re_uS, l2re_DS = np.zeros(Numrep), np.zeros(Numrep) # relative L2 error
    l1re_uS, l1re_DS = np.zeros(Numrep), np.zeros(Numrep) # relative L1 error
    mse_uS, mse_DS = np.zeros(Numrep), np.zeros(Numrep)    # mean squared error
    merr_uS, merr_DS = np.zeros(Numrep), np.zeros(Numrep) # maximum absolute error

    for i in range(Numrep):
        save_dir_i = save_dir + f'Iter{i}/'

        # load the best model of each repetition
        u_net = torch.load(f"{save_dir_i}U_net_best_Iter{i}.pth", map_location=device)
        D_net = torch.load(f"{save_dir_i}D_net_best_Iter{i}.pth", map_location=device)
        # evaluate model
        u_net.eval()
        D_net.eval()

        with torch.no_grad():
            # 1) Performance evaluation on the full grid points:
            u_pred_np_full = u_net(XT_tensor).cpu().numpy().reshape(len(t_train), len(x_train))
            l2re_uS_full[i] = np.sqrt(np.sum((u_pred_np_full - u_true)**2)) / np.sqrt(np.sum(u_true**2))
            l1re_uS_full[i] = np.sum(np.abs(u_pred_np_full - u_true)) / np.sum(np.abs(u_true))
            mse_uS_full[i] = np.mean((u_pred_np_full - u_true)**2)
            merr_uS_full[i] = np.max(np.abs(u_pred_np_full - u_true))

            D_pred_full = D_net(U_grid_tensor.unsqueeze(1)).cpu().numpy().flatten()
            l2re_DS_full[i] = np.sqrt(np.sum((D_pred_full - D)**2)) / np.sqrt(np.sum(D**2))
            l1re_DS_full[i] = np.sum(np.abs(D_pred_full - D)) / np.sum(np.abs(D))
            mse_DS_full[i] = np.mean((D_pred_full - D)**2)
            merr_DS_full[i] = np.max(np.abs(D_pred_full - D))

            np.save(f'{save_dir_i}fullgrid_u_pred.npy', u_pred_np_full)
            np.save(f'{save_dir_i}fullgrid_D_pred.npy', D_pred_full)

            # plot the learnt u and D and V
            savename = 'fullgrid_u_D_learnt.png'
            learnt_u_D_plot(Nt, x_train, u_true, u_pred_np_full, D, U_grid, D_pred_full, u_min, u_max, save_dir_i, savename)
            # plot the relative error of the learnt u and D and V
            savename = 'fullgrid_u_D_learnt_relerr.png'
            u_D_relerr(u_pred_np_full, u_true, X_grid, T_grid, t_train, x_train, D_pred_full, D, U_grid, save_dir_i, savename)
            
            # 2) Performance evaluation on the test dataset: 
            # initialize the metrics for each batch
            l2re_u, l2re_D = 0.0, 0.0
            l1re_u, l1re_D = 0.0, 0.0
            mse_u, mse_D = 0.0, 0.0
            merr_u, merr_D = 0.0, 0.0
            for x_batch, u_batch in test_loader:
                x_batch, u_batch = x_batch.to(device), u_batch.to(device)
                u_pred = u_net(x_batch)
                D_pred = D_net(u_pred)
                l2re_u += np.sqrt(np.sum((u_pred.cpu().detach().numpy() - u_batch.cpu().detach().numpy())**2) / np.sum(u_batch.cpu().detach().numpy()**2))
                l1re_u += np.sum(np.abs(u_pred.cpu().detach().numpy() - u_batch.cpu().detach().numpy())) / np.sum(np.abs(u_batch.cpu().detach().numpy()))
                mse_u += np.mean((u_pred.cpu().detach().numpy() - u_batch.cpu().detach().numpy())**2)
                merr_u += np.max(np.abs(u_pred.cpu().detach().numpy() - u_batch.cpu().detach().numpy()))
                l2re_D += np.sqrt(np.sum((D_pred.cpu().detach().numpy() - D)**2) / np.sum(D**2))
                l1re_D += np.sum(np.abs(D_pred.cpu().detach().numpy() - D)) / np.sum(np.abs(D))
                mse_D += np.mean((D_pred.cpu().detach().numpy() - D)**2)
                merr_D += np.max(np.abs(D_pred.cpu().detach().numpy() - D))
            l2re_uS[i] = l2re_u / num_batches
            l2re_DS[i] = l2re_D / num_batches
            l1re_uS[i] = l1re_u / num_batches
            l1re_DS[i] = l1re_D / num_batches
            mse_uS[i] = mse_u / num_batches
            mse_DS[i] = mse_D / num_batches
            merr_uS[i] = merr_u / num_batches
            merr_DS[i] = merr_D / num_batches

    # save the performance metrics on
    # 1) full grid points:
    np.save(f'{save_dir}fullgrid_performance_u.npy', np.array([l2re_uS_full, l1re_uS_full, mse_uS_full, merr_uS_full]))
    np.save(f'{save_dir}fullgrid_performance_D.npy', np.array([l2re_DS_full, l1re_DS_full, mse_DS_full, merr_DS_full]))
    # 2) test dataset:
    np.save(f'{save_dir}testdataset_performance_u.npy', np.array([l2re_u, l1re_u, mse_u, merr_u]))
    np.save(f'{save_dir}testdataset_performance_D.npy', np.array([l2re_D, l1re_D, mse_D, merr_D])) 
    pass

# -------------------------------------- Train model --------------------------------------

def TrainBINN(save_dir, device, D, t_end, batch_size, Numrep, Num_epochs, u_num_layers, u_hidden_units, u_activation_fn,
              D_num_layers, D_hidden_units, D_activation_fn, init_learning_rate, data_weight, bc_weight, pde_weight,
              N_PDE, N_BC, store_loss_step, grid_points_tensor, U_grid, U_grid_tensor, t_train, x_train, u_true, X_grid, T_grid, Nt, u_min, u_max,
              N_bio, u_bio_weight, D_bio_weight, bio_constraint_bool, val_dataset, test_dataset, train_dataset, x_max, x_min, t_min, FixedTime):
    """
    CPU-only version of TrainBINN.
    Profiling uses time.perf_counter() and records per-epoch mean times for:
      - forward (u_net forward)
      - data (data loss compute)
      - pde_forward  (forward for PDE loss compute)
      - pde   (PDE loss compute)
      - bc_forward  (forward for boundary loss compute)
      - bc   (boundary loss compute)
      - backward (loss.backward)
      - opt  (optimizer.step)
    """

    # Ensure save_dir exists
    os.makedirs(save_dir, exist_ok=True)

    # Load val and test loaders
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # record/append results for each repetition into lists
    u_predS = []
    D_predS = []

    # record/append training and validation losses for each repetition
    uloss_S, BCloss_S, pdeloss_S, u_bio_S, D_bio_S, totalloss_S = [], [], [], [], [], []
    val_uloss_S, val_BCloss_S, val_pdeloss_S, val_u_bio_S, val_D_bio_S, val_totalloss_S = [], [], [], [], [], []
    epoch_history_S = []

    totalloss_bestepoch_alliters = np.ones(Numrep) * 1e12
    min_valloss_epoch = np.zeros(Numrep, dtype=int)
    timeS = []

    # Loop over repetitions (independent training runs)
    for i in range(Numrep):
        save_dir_i = save_dir + f'Iter{i}/'
        os.makedirs(save_dir_i, exist_ok=True)

        # Per-repetition profiling containers (CPU only)
        timings_dict = {
            "mean_forward_s": [],
            "mean_data_s": [],
            "mean_pde_forward_s": [],
            "mean_pde_s": [],
            "mean_bc_forward_s": [],
            "mean_bc_s": [],
            "mean_backward_s": [],
            "mean_opt_s": []
        }

        # per-repetition histories
        epoch_history = []
        uloss_history, BCloss_history, pdeloss_history, totalloss_history = [], [], [], []
        u_bio_history, D_bio_history = [], []
        val_uloss_history, val_BCloss_history, val_pdeloss_history, val_totalloss_history = [], [], [], []
        val_u_bio_history, val_D_bio_history = [], []

        # Prepare data loader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Build models and optimizer (assumes UNet and DNet are in scope)
        u_net = UNet(u_num_layers, u_hidden_units, u_activation_fn).to(device)
        D_net = DNet(D_num_layers, D_hidden_units, D_activation_fn).to(device)
        optimizer = torch.optim.Adam(list(u_net.parameters()) + list(D_net.parameters()), lr=init_learning_rate)

        # Timing the whole repetition
        rep_start = time.perf_counter()
        t0000 = time.perf_counter()

        # Epoch loop
        for epoch in range(Num_epochs):
            u_net.train()
            D_net.train()

            # epoch accumulators for losses
            epoch_total_loss = 0.0
            epoch_u_loss = 0.0
            epoch_bc_loss = 0.0
            epoch_pde_loss = 0.0
            epoch_u_bio_loss = 0.0
            epoch_D_bio_loss = 0.0

            # epoch accumulators for profiling
            sum_time = {"forward": 0.0, "data": 0.0, "pde_forward": 0.0, "pde": 0.0, "bc_forward": 0.0, "bc": 0.0, "backward": 0.0, "opt": 0.0}

            # Batch loop
            for _, (x_batch, u_batch) in enumerate(train_loader):
                x_batch = x_batch.to(device)
                u_batch = u_batch.to(device)

                # 1) forward timing
                t0 = time.perf_counter()
                u_pred_data = u_net(x_batch)
                t1 = time.perf_counter()
                t_forward = t1 - t0
                sum_time["forward"] += t_forward

                # 2) data loss timing
                t0 = time.perf_counter()
                u_loss = data_loss_Gaussian(u_pred_data, u_batch)
                t1 = time.perf_counter()
                t_data = t1 - t0
                sum_time["data"] += t_data
                epoch_u_loss += u_loss.item()

                # 3) PDE loss timing
                pde_loss, t_pde_forward, t_pde = pde_loss_Diffusion(x_max, x_min, t_end, t_min, device, u_net, D_net, N_PDE)
                sum_time["pde_forward"] += t_pde_forward
                sum_time["pde"] += t_pde
                epoch_pde_loss += pde_loss.item()

                # 4) BC loss timing
                neumann_bc_loss, t_bc_forward, t_bc = boundary_loss_neumann(x_min, x_max, device, u_net, t_min, t_end, N_BC)
                sum_time["bc_forward"] += t_bc_forward
                sum_time["bc"] += t_bc
                epoch_bc_loss += neumann_bc_loss.item()

                # 5) biology constraints (optional)
                if bio_constraint_bool:
                    u_bio_loss, D_bio_loss = biology_loss(x_max, x_min, t_min, t_end, device, u_net, D_net, N_bio)
                    # not included in sum_time by default, but we record the loss totals
                    epoch_u_bio_loss += u_bio_loss.item()
                    epoch_D_bio_loss += D_bio_loss.item()

                # 6) assemble total loss
                total_loss = data_weight * u_loss + pde_weight * pde_loss + bc_weight * neumann_bc_loss
                if bio_constraint_bool:
                    total_loss = total_loss + u_bio_weight * u_bio_loss + D_bio_weight * D_bio_loss
                epoch_total_loss += total_loss.item()

                # 7) backward timing
                optimizer.zero_grad()
                t0 = time.perf_counter()
                total_loss.backward()
                t1 = time.perf_counter()
                t_backward = t1 - t0
                sum_time["backward"] += t_backward

                # 8) optimizer step timing
                t0 = time.perf_counter()
                optimizer.step()
                t1 = time.perf_counter()
                t_opt = t1 - t0
                sum_time["opt"] += t_opt

            # End batch loop: normalize epoch losses by number of batches (protect if empty)
            n_batches = len(train_loader) if len(train_loader) > 0 else 1
            epoch_total_loss = epoch_total_loss / n_batches
            epoch_u_loss = epoch_u_loss / n_batches
            epoch_bc_loss = epoch_bc_loss / n_batches
            epoch_pde_loss = epoch_pde_loss / n_batches
            if bio_constraint_bool:
                epoch_u_bio_loss = epoch_u_bio_loss / n_batches
                epoch_D_bio_loss = epoch_D_bio_loss / n_batches

            # compute and append per-epoch mean timings (append every epoch)
            timings_dict["mean_forward_s"].append(sum_time["forward"])
            timings_dict["mean_data_s"].append(sum_time["data"])
            timings_dict["mean_pde_forward_s"].append(sum_time["pde_forward"])
            timings_dict["mean_pde_s"].append(sum_time["pde"])
            timings_dict["mean_bc_forward_s"].append(sum_time["bc_forward"])
            timings_dict["mean_bc_s"].append(sum_time["bc"])
            timings_dict["mean_backward_s"].append(sum_time["backward"])
            timings_dict["mean_opt_s"].append(sum_time["opt"])

            # -------------------------------------------------------- validation mode --------------------------------------------------------
            u_net.eval()
            D_net.eval()
            # accumulate the validation losses over batches
            val_u_loss = 0.0
            val_total_loss, val_u_loss, val_bc_loss, val_pde_loss, val_u_bio_loss, val_D_bio_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            for x_batch_val, u_batch_val in val_loader:
                x_batch_val, u_batch_val = x_batch_val.to(device), u_batch_val.to(device)
                # 1) data loss
                u_pred_val = u_net(x_batch_val)
                u_loss_ = data_loss_Gaussian(u_pred_val, u_batch_val)
                val_u_loss += u_loss_.item()
                # 2) PDE constraint loss (based on sampled 'N_PDE' virtual data):
                pde_loss_, _, _ = pde_loss_Diffusion(x_max, x_min, t_end, t_min, device, u_net, D_net, N_PDE)
                val_pde_loss += pde_loss_.item()              
                # 3) Neumann boundary-condition loss (based on sampled 'N_BC' virtual data):
                bc_loss_, _, _ = boundary_loss_neumann(x_min, x_max, device, u_net, t_min, t_end, N_BC)
                val_bc_loss += bc_loss_.item()
                # 4) Biological constraints: i.e. learnt u, D >= 0
                u_bio_loss_, D_bio_loss_ = biology_loss(x_max, x_min, t_min, t_end, device, u_net, D_net, N_bio)
                val_u_bio_loss += u_bio_loss_.item()
                val_D_bio_loss += D_bio_loss_.item()
                # 5) total loss
                total_loss_ = data_weight * u_loss_ + bc_weight * bc_loss_ + pde_weight * pde_loss_
                if bio_constraint_bool:
                    total_loss_ = total_loss_ + u_bio_weight * u_bio_loss_ + D_bio_weight * D_bio_loss_
                val_total_loss += total_loss_.item()
            val_total_loss /= len(val_loader)
            val_u_loss /= len(val_loader)
            val_bc_loss /= len(val_loader)
            val_pde_loss /= len(val_loader)
            if bio_constraint_bool:
                val_u_bio_loss /= len(val_loader)
                val_D_bio_loss /= len(val_loader)

            # store the best model based on the validation loss
            if val_total_loss < totalloss_bestepoch_alliters[i]: # check if the validation loss has improved
                totalloss_bestepoch_alliters[i] = val_total_loss
                min_valloss_epoch[i] = epoch
                # save the best model for this repetition
                torch.save(u_net, f'{save_dir_i}U_net_best_Iter{i}.pth')
                torch.save(D_net, f'{save_dir_i}D_net_best_Iter{i}.pth')

            # save every 'store_loss_step' epochs
            if epoch % store_loss_step == 0:
                epoch_history.append(epoch)
            
                # 1). training loss
                uloss_history.append(epoch_u_loss)
                BCloss_history.append(epoch_bc_loss)
                pdeloss_history.append(epoch_pde_loss)
                totalloss_history.append(epoch_total_loss)
                if bio_constraint_bool:
                    u_bio_history.append(epoch_u_bio_loss)
                    D_bio_history.append(epoch_D_bio_loss)
                # 2). validation loss
                val_uloss_history.append(val_u_loss) 
                val_BCloss_history.append(val_bc_loss)
                val_pdeloss_history.append(val_pde_loss)
                val_totalloss_history.append(val_total_loss)

                # save epoch_history and loss histories for this repetition to 'save_dir_i'
                np.save(f'{save_dir_i}epoch_history.npy', np.array(epoch_history))
                np.save(f'{save_dir_i}uloss_history.npy', np.array(uloss_history))
                np.save(f'{save_dir_i}BCloss_history.npy', np.array(BCloss_history))
                np.save(f'{save_dir_i}pdeloss_history.npy', np.array(pdeloss_history))
                np.save(f'{save_dir_i}totalloss_history.npy', np.array(totalloss_history))
                np.save(f'{save_dir_i}val_uloss_history.npy', np.array(val_uloss_history))
                np.save(f'{save_dir_i}val_BCloss_history.npy', np.array(val_BCloss_history))
                np.save(f'{save_dir_i}val_pdeloss_history.npy', np.array(val_pdeloss_history))
                np.save(f'{save_dir_i}val_totalloss_history.npy', np.array(val_totalloss_history))


                if bio_constraint_bool:
                    val_u_bio_history.append(val_u_bio_loss)
                    val_D_bio_history.append(val_D_bio_loss)
                    np.save(f'{save_dir_i}val_u_bio_history.npy', np.array(val_u_bio_history))
                    np.save(f'{save_dir_i}val_D_bio_history.npy', np.array(val_D_bio_history))

            # if the time at this stage is bigger than 'FixedTime', break the epoch loop directly to below the repetition loop to save time 
            t1111 = time.perf_counter()
            time_sofar = t1111 - t0000
            if time_sofar >= FixedTime:
                print(f"Time exceeded FixedTime={FixedTime}s at epoch {epoch} for repetition {i}. Breaking epoch loop to save time.")
                break
            # -------------------------------------------------------- end validation mode ------------------------------------------------


        # End epoch loop
        rep_end = time.perf_counter()
        timeS.append(rep_end - rep_start)

        # save per-repetition profiling dictionary: timings_dict
        with open(f'{save_dir_i}profiling_dict{i}.pkl', 'wb') as f:
            pickle.dump(timings_dict, f)


        # Save training/validation histories for this repetition to the global lists
        epoch_history_S.append(np.array(epoch_history))
        uloss_S.append(np.array(uloss_history))
        BCloss_S.append(np.array(BCloss_history))
        pdeloss_S.append(np.array(pdeloss_history))
        totalloss_S.append(np.array(totalloss_history))
        if bio_constraint_bool:
            u_bio_S.append(np.array(u_bio_history))
            D_bio_S.append(np.array(D_bio_history))

        val_uloss_S.append(np.array(val_uloss_history))
        val_BCloss_S.append(np.array(val_BCloss_history))
        val_pdeloss_S.append(np.array(val_pdeloss_history))
        val_totalloss_S.append(np.array(val_totalloss_history))
        if bio_constraint_bool:
            val_u_bio_S.append(np.array(val_u_bio_history))
            val_D_bio_S.append(np.array(val_D_bio_history))

        # Evaluate best models on grid and save predictions
        u_net = torch.load(f"{save_dir_i}U_net_best_Iter{i}.pth", map_location=device)
        D_net = torch.load(f"{save_dir_i}D_net_best_Iter{i}.pth", map_location=device)
        u_net.eval()
        D_net.eval()
        with torch.no_grad():
            u_pred = u_net(grid_points_tensor).cpu().numpy().reshape(len(t_train), len(x_train))
            u_predS.append(u_pred)
            D_pred = D_net(U_grid_tensor.unsqueeze(1)) 
            D_pred_np = D_pred.cpu().numpy().flatten()
            D_predS.append(D_pred_np)

    # save the training and validation results (Ps if apply early stopping, each of Numrep np arrays might have different lengths, 
    # thus use 'allow_pickle=True')
    np.save(f'{save_dir}/u_predS.npy', np.array(u_predS))
    np.save(f'{save_dir}/D_predS.npy', np.array(D_predS))
    np.save(f'{save_dir}/totalloss_bestepoch_alliters.npy', totalloss_bestepoch_alliters)
    np.save(f'{save_dir}/min_valloss_epoch.npy', min_valloss_epoch)
    np.save(f'{save_dir}/timeS.npy', np.array(timeS))
    np.save(f'{save_dir}/uloss_S.npy', np.array(uloss_S, dtype=object), allow_pickle=True)
    np.save(f'{save_dir}/BCloss_S.npy', np.array(BCloss_S, dtype=object), allow_pickle=True)
    np.save(f'{save_dir}/pdeloss_S.npy', np.array(pdeloss_S, dtype=object), allow_pickle=True)
    np.save(f'{save_dir}/totalloss_S.npy', np.array(totalloss_S, dtype=object), allow_pickle=True)
    np.save(f'{save_dir}/val_uloss_S.npy', np.array(val_uloss_S, dtype=object), allow_pickle=True)
    np.save(f'{save_dir}/val_BCloss_S.npy', np.array(val_BCloss_S, dtype=object), allow_pickle=True)
    np.save(f'{save_dir}/val_pdeloss_S.npy', np.array(val_pdeloss_S, dtype=object), allow_pickle=True)
    np.save(f'{save_dir}/val_totalloss_S.npy', np.array(val_totalloss_S, dtype=object), allow_pickle=True)
    np.save(f'{save_dir}/epoch_history_S.npy', np.array(epoch_history_S, dtype=object), allow_pickle=True)
    np.save(f'{save_dir}/val_u_bio_S.npy', np.array(val_u_bio_S, dtype=object), allow_pickle=True)
    np.save(f'{save_dir}/val_D_bio_S.npy', np.array(val_D_bio_S, dtype=object), allow_pickle=True)
    np.save(f'{save_dir}/u_bio_S.npy', np.array(u_bio_S, dtype=object), allow_pickle=True)
    np.save(f'{save_dir}/D_bio_S.npy', np.array(D_bio_S, dtype=object), allow_pickle=True)

    model_performance_eval(save_dir, u_net, D_net, test_loader, device, t_train, x_train, U_grid_tensor, Nt, u_true, \
               D, U_grid, u_min, u_max, X_grid, T_grid, BCloss_S, pdeloss_S, val_BCloss_S, val_pdeloss_S, Numrep, store_loss_step, \
                    val_u_bio_S, val_D_bio_S, u_bio_S, D_bio_S, uloss_S, val_uloss_S, Num_epochs, bio_constraint_bool)

    pass