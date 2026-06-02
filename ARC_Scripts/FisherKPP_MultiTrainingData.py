# import libraries and set the device
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd
import os
import sys
from torch.utils.data import TensorDataset
from scipy import integrate

from subfunction_ReactionDiffusion_MultiTrainingData import *

# --- From Becky ---
# reaction term
def G_true(u):
    return r_true * (1.0 - u/K_true)

# PDE RHS for method-of-lines
def rhs(ti, y):
    u = y
    ux  = np.gradient(u, dx, edge_order=2)
    uxx = np.gradient(ux, dx, edge_order=2)
    return D_true * uxx + u * G_true(u)

# Different initial conditions
def make_ic(kind, params):
    if kind == "step":
        c = params.get("center", 0.0)
        return (x_train_ < c).astype(float)
    if kind == "gauss":
        c = params.get("center", 0.0)
        s = params.get("sigma", 0.7)
        a = params.get("amp", 0.8)
        return a * np.exp(-0.5*((x_train_-c)/s)**2)
    if kind == "bump":
        c1 = params.get("c1", -1.5)
        c2 = params.get("c2", 1.5)
        s  = params.get("sigma", 0.6)
        u0 = np.exp(-0.5*((x_train_-c1)/s)**2) + 0.7*np.exp(-0.5*((x_train_-c2)/s)**2)
        return np.minimum(u0, 1.0)
    raise ValueError("Unknown IC type")
# --- End of Becky ---

if __name__ == "__main__":
    # plot formatting
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.labelweight'] = 'bold'  # Bold axis labels
    plt.rcParams['axes.titleweight'] = 'bold'  # Bold title
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['font.size'] = str(16)
    # set the device 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    # PDE parameters
    D_true = 0.1
    r_true = 1.0
    K_true = 1.0

    # # space resolution
    # Nx, L = 200, 7.0
    # x_min, x_max = -L, L
    # x_train = np.linspace(x_min, x_max, Nx)
    # dx = x_train[1] - x_train[0]

    # # time resolution
    # Nt, t_end, t_min = 6, 4.0, 0.0
    # t_train = np.linspace(t_min, t_end, Nt)

    # space resolution
    L = 7.0
    x_min, x_max = -L, L
    dx = 0.04
    x_train_ = np.arange(x_min, x_max+dx, dx)
    Nx = len(x_train_)

    # time resolution
    Nt, t_end, t_min = 7, 4.5, 0.0
    t_train = np.linspace(t_min, t_end, Nt)

    xx, tt = np.meshgrid(x_train_, t_train)

    train_ratio, val_ratio = 0.8, 0.1 
    u_min_np, u_max_np = 0.0, 1.0
    u_min, u_max = u_min_np, u_max_np
    u_norm = u_max - u_min
    

    # # generate training data given different initial conditions
    # ic_specs = [
    #     ("step",  {"center": -1.0}),
    #     ("gauss", {"center": -0.5, "sigma": 0.6, "amp": 0.8}),
    #     ("gauss", {"center":  0.7, "sigma": 0.7, "amp": 0.7}),
    #     ("bump",  {"c1": -2.0, "c2": 1.0, "sigma": 0.7}),
    # ]
    # generate training data given different initial conditions
    ic_specs = [
            ("step",  {"center": -1.0}),
            ("gauss", {"center": -0.5, "sigma": 0.6, "amp": 0.8}),
            ("gauss", {"center":  0.7, "sigma": 0.7, "amp": 0.7}),
            ("bump",  {"c1": -1.0, "c2": 1.5, "sigma": 0.7}),
    ]

    batch_size = 40
    #noise_level = 0.05
    #noise_level = 0.5
    #variances = 0.0
    #variances = 1e-2
    variances = 1e-3
    noise_level = np.sqrt(variances)
    

    u_allICs, train_dataset_allICs, val_dataset_allICs, test_dataset_allICs = [], [], [], []
    x_batches_train_allICs, u_batches_train_allICs = [], []
    x_batches_val_allICs, u_batches_val_allICs = [], []
    x_batches_test_allICs, u_batches_test_allICs = [], []
    for kind, params in ic_specs:
        u0_i = make_ic(kind, params)
        sol_i = integrate.solve_ivp(rhs, (t_train[0], t_train[-1]), u0_i, t_eval=t_train, method="RK45", rtol=1e-6, atol=1e-8)
        u_train_nonoise_i_ = sol_i.y.T.astype(np.float32)  # shape (Nt, Nx)



        ## YY: subsampling: 1/5th of the datapoints (in space) ----------------------------------
        x_train = x_train_[::5]
        dx = x_train[1] - x_train[0]
        Nx = len(x_train)
        X, T = np.meshgrid(x_train, t_train)
        u_train_nonoise_i = u_train_nonoise_i_[:, ::5]


        shape = (Nt, Nx)
        input_data = np.concatenate([X.reshape(-1)[:, None], T.reshape(-1)[:, None]], axis=1)
        # Resolutions of evaluating the learnt u_net, D_net and G_net after training
        grid_points = np.vstack([X.flatten(), T.flatten()]).T
        grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32).to(device)
        U_grid = np.linspace(u_min_np, u_max_np, Nx) 
        U_grid_tensor = torch.tensor(U_grid, dtype=torch.float32).to(device)
        G_true_np = r_true * (1 - U_grid)
        D_true_np = D_true * np.ones_like(U_grid)




        # Add noise to the training data
        seed = 2 # for reproducibility
        np.random.seed(seed)
        additive_noise = noise_level * np.random.randn(*shape)
        u_train_noise_i = u_train_nonoise_i + additive_noise
        u_allICs.append(u_train_noise_i)

        # prepare training data as tensor dataset
        output_data_i = u_train_noise_i.reshape(-1)[:, None]
        data_i = pd.DataFrame(np.hstack([input_data, output_data_i]), columns=['x', 't', 'u'])
        x_data_i = torch.tensor(data_i[['x', 't']].values, dtype=torch.float32).to(device)
        u_data_i = torch.tensor(data_i[['u']].values, dtype=torch.float32).to(device)
        dataset_i = TensorDataset(x_data_i, u_data_i)

        # split the data into training, validation, and testing sets given different ratios
        train_size_i, val_size_i = int(len(dataset_i) * train_ratio), int(len(dataset_i) * val_ratio)
        test_size_i = len(dataset_i) - train_size_i - val_size_i
        train_dataset_i, val_dataset_i, test_dataset_i = torch.utils.data.random_split(dataset_i, [train_size_i, val_size_i, test_size_i])
        train_dataset_allICs.append(train_dataset_i)
        val_dataset_allICs.append(val_dataset_i)
        test_dataset_allICs.append(test_dataset_i)

        x_batches_train, u_batches_train = [], []
        train_loader_i = DataLoader(train_dataset_i, batch_size=batch_size, shuffle=True)
        for _, (x_batch, u_batch) in enumerate(train_loader_i):
            x_batches_train.append(x_batch)
            u_batches_train.append(u_batch)
        x_batches_train_allICs.append(x_batches_train)
        u_batches_train_allICs.append(u_batches_train)

        x_batches_val, u_batches_val = [], []
        val_loader_i = DataLoader(val_dataset_i, batch_size=batch_size, shuffle=False)
        for _, (x_batch, u_batch) in enumerate(val_loader_i):
            x_batches_val.append(x_batch)
            u_batches_val.append(u_batch)
        x_batches_val_allICs.append(x_batches_val)
        u_batches_val_allICs.append(u_batches_val)

        x_batches_test, u_batches_test = [], []
        test_loader_i = DataLoader(test_dataset_i, batch_size=batch_size, shuffle=False)
        for _, (x_batch, u_batch) in enumerate(test_loader_i):
            x_batches_test.append(x_batch)
            u_batches_test.append(u_batch)
        x_batches_test_allICs.append(x_batches_test)
        u_batches_test_allICs.append(u_batches_test)

    u_true_allICs = u_allICs.copy()

    results_dir = f'Fisher_KPP_MultiICs_var{variances}_subsample/' 
    os.makedirs(results_dir, exist_ok=True)
    Numrep, Num_epochs, store_loss_step = 10, 4001, 1
    bio_constraint_bool = True
    N_bio, u_bio_weight, D_bio_weight, G_bio_weight = 40, 1.0, 1.0, 1.0

    # save u_allICs, train_dataset_allICs, val_dataset_allICs, test_dataset_allICs
    u_allICs_np = np.array(u_allICs)
    np.save(results_dir + f'u_allICs_var{variances}.npy', u_allICs_np)
    torch.save(train_dataset_allICs, results_dir + f'train_dataset_allICs_var{variances}.pt')
    torch.save(val_dataset_allICs, results_dir + f'val_dataset_allICs_var{variances}.pt')
    torch.save(test_dataset_allICs, results_dir + f'test_dataset_allICs_var{variances}.pt')
    torch.save(x_batches_train_allICs, results_dir + f'x_batches_train_allICs_var{variances}.pt')
    torch.save(u_batches_train_allICs, results_dir + f'u_batches_train_allICs_var{variances}.pt')
    torch.save(x_batches_val_allICs, results_dir + f'x_batches_val_allICs_var{variances}.pt')
    torch.save(u_batches_val_allICs, results_dir + f'u_batches_val_allICs_var{variances}.pt')
    torch.save(x_batches_test_allICs, results_dir + f'x_batches_test_allICs_var{variances}.pt')
    torch.save(u_batches_test_allICs, results_dir + f'u_batches_test_allICs_var{variances}.pt')

    # ----------------------------------------------------------------------------------------------------

    benchmark_job = int(sys.argv[1]) 

    if benchmark_job == 0: # YY test
        save_dir = results_dir + f'MultiStart_var{variances}/' 
        os.makedirs(save_dir, exist_ok=True)

        batch_size, N_pde, N_bc = 40, 40, 40
        data_weight, pde_weight, bc_weight= 1.0, 1e-1, 1e-3
        init_learning_rate = 5e-4
        u_num_layers, u_hidden_units = 4, 32
        u_activation_fn = nn.Tanh()
        D_num_layers, D_hidden_units = 4, 32
        D_activation_fn = nn.Tanh()
        G_num_layers, G_hidden_units = 4, 32
        G_activation_fn = nn.Tanh()

        TrainBINN_multiICs(save_dir, device, t_end, Numrep, Num_epochs, u_num_layers, u_hidden_units, u_activation_fn, \
              D_num_layers, D_hidden_units, D_activation_fn, init_learning_rate, data_weight, bc_weight, pde_weight, \
                N_pde, N_bc, store_loss_step, grid_points_tensor, U_grid_tensor, t_train, x_train, u_true_allICs, \
                    N_bio, u_bio_weight, D_bio_weight, G_num_layers, G_hidden_units, G_activation_fn, G_bio_weight, bio_constraint_bool, \
                        x_max, x_min, t_min, x_batches_train_allICs, u_batches_train_allICs, x_batches_val_allICs, u_batches_val_allICs, x_batches_test_allICs)
