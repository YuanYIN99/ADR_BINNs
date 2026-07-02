# import libraries and set the device
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd
import os
import sys
from torch.utils.data import TensorDataset
from scipy.integrate import odeint

from subfunction_AdvectionDiffusion import *

############## EQUATION SOLVING ###############
# Definition of ODE system (PDE ---(FFT)---> ODE system)
def burg_system(u, t, k, mu, nu):
    # Spatial derivative in the Fourier domain
    u_hat = np.fft.fft(u)
    u_hat_x = 1j * k * u_hat
    u_hat_xx = -k**2 * u_hat

    # Switching in the spatial domain
    u_x = np.fft.ifft(u_hat_x)
    u_xx = np.fft.ifft(u_hat_xx)

    # ODE resolution
    u_t = -mu * u * u_x + nu * u_xx
    return u_t.real

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

    # solve the Burgers' equation to generate data: 
    mu = 0.6 # coefficient for the advection term
    nu = 0.01 # diffusion coefficient

    # # space resolution
    # Nx, L = 300, 6
    # x_min, x_max = -L+1, L
    # x_train = np.linspace(x_min, x_max, Nx)
    # dx = x_train[1] - x_train[0]
    # # Wave number discretization
    # k = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    # # time resolution
    # Nt, t_end, t_min = 6, 10, 0.0
    # t_train = np.linspace(t_min, t_end, Nt)

    # space resolution
    L = 7.0
    x_min, x_max = -L, L
    dx = 0.04
    x_train = np.arange(x_min, x_max+dx, dx)
    Nx = len(x_train)

    # time resolution
    Nt, t_end, t_min = 7, 10.0, 0.0
    t_train = np.linspace(t_min, t_end, Nt)

    # Wave number discretization
    k = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)


    # Initial condition
    u0 = np.exp(-(x_train) ** 2 / 2)
    u_train_nonoise_ = odeint(burg_system, u0, t_train, args=(k, mu, nu,), mxstep=5000)


    ## YY: subsampling: 1/5th of the datapoints (in space) ----------------------------------
    x_train = x_train[::5]
    dx = x_train[1] - x_train[0]
    Nx = len(x_train)
    X, T = np.meshgrid(x_train, t_train)
    u_train_nonoise = u_train_nonoise_[:, ::5]

    u_min_np, u_max_np = u_train_nonoise.min(), u_train_nonoise.max()
    u_true = u_train_nonoise.copy() # NOTE: no noise case
    shape = u_train_nonoise.shape

    # Add noise to the training data
    seed = 2 # for reproducibility
    np.random.seed(seed)
    #noise_level = 0.05
    #noise_level = 0.5
    #noise_level = 0.0
    #variance = 0.0
    #variance = 1e-2
    variance = 1e-3
    noise_level = np.sqrt(variance)
    additive_noise = noise_level * np.random.randn(*shape)
    u_train_noise = u_train_nonoise + additive_noise

    # prepare the dataset as tensors for training
    input_data = np.concatenate([X.reshape(-1)[:, None], T.reshape(-1)[:, None]], axis=1)
    #output_data = u_train_nonoise.reshape(-1)[:, None]
    output_data = u_train_noise.reshape(-1)[:, None]

    data = pd.DataFrame(np.hstack([input_data, output_data]), columns=['x', 't', 'u'])
    x_data = torch.tensor(data[['x', 't']].values, dtype=torch.float32).to(device)
    u_data = torch.tensor(data[['u']].values, dtype=torch.float32).to(device)
    # find the min and max of u for normalization
    u_min, u_max = u_data.min(), u_data.max()
    u_norm = u_max - u_min
    dataset = TensorDataset(x_data, u_data)
    # split the data into training, validation, and testing sets given different ratios
    train_ratio, val_ratio = 0.8, 0.1 
    train_size, val_size = int(len(dataset) * train_ratio), int(len(dataset) * val_ratio)
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    # Resolutions of evaluating the learnt u_net and D_net after training
    # 1). evaluate U_net at 'grid_points_tensor'
    grid_points = np.vstack([X.flatten(), T.flatten()]).T
    grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32).to(device)
    # 2). evaluate D_net at 'U_grid_tensor'
    U_grid = np.linspace(u_min_np, u_max_np, Nx) 
    U_grid_tensor = torch.tensor(U_grid, dtype=torch.float32).to(device)

    results_dir = f'Burgers_subsample/'
    os.makedirs(results_dir, exist_ok=True)
    
    Numrep, Num_epochs, store_loss_step = 10, 4001, 1
    bio_constraint_bool = False
    u_bio_weight, D_bio_weight = 1.0, 1.0
    N_bio = 1

    D = nu * np.ones(np.shape(U_grid)) # the true diffusion coefficient
    V_true = (mu / 2) * U_grid

    # save u_train_noise
    np.save(results_dir + f'u_train_noise_var{variance}.npy', u_train_noise)

    # ----------------------------------------------------------------------------------------------------
    benchmark_job = int(sys.argv[1]) 

    if benchmark_job == 0: # YY test
        save_dir = results_dir + 'test/' 
        os.makedirs(save_dir, exist_ok=True)

        batch_size, N_pde, N_bc = 40, 40, 1
        data_weight, pde_weight, bc_weight= 1.0, 1.0, 0.0
        init_learning_rate = 5e-4
        u_num_layers, u_hidden_units = 4, 32
        u_activation_fn = nn.Tanh()
        D_num_layers, D_hidden_units = 4, 32
        D_activation_fn = nn.Tanh()
        V_num_layers, V_hidden_units = 4, 32
        V_activation_fn = nn.Tanh()  
        
        TrainBINN(save_dir, device, D, t_end, batch_size, Numrep, Num_epochs, u_num_layers, u_hidden_units, u_activation_fn, \
              D_num_layers, D_hidden_units, D_activation_fn, init_learning_rate, data_weight, bc_weight, pde_weight, \
                N_pde, N_bc, store_loss_step, grid_points_tensor, U_grid, U_grid_tensor, t_train, x_train, u_true, X, T, Nt, u_min, u_max, \
                    N_bio, u_bio_weight, D_bio_weight, V_num_layers, V_hidden_units, V_activation_fn, V_true, bio_constraint_bool, \
                        val_dataset, test_dataset, train_dataset, x_max, x_min, t_min)
        
    if benchmark_job == 1: # weightings on loss terms
        # data_weight, bc_weight, pde_weight
        weights = np.array([
                           [1.0, 0.0, 1e-3], 
                           [1.0, 0.0, 1e-2], 
                           [1.0, 0.0, 1e-1], 
                           [1.0, 0.0, 1.0], 
                           [1.0, 0.0, 1e1], 
                           [1.0, 0.0, 1e2], 
                           [1.0, 0.0, 1e3], 

                           [1.0, 1e-3, 0.1], 
                           [1.0, 1e-2, 0.1], 
                           [1.0, 1e-1, 0.1], 
                           [1.0, 1.0, 0.1], 
                           [1.0, 1e1, 0.1], 
                           [1.0, 1e2, 0.1], 
                           [1.0, 1e3, 0.1], 
                           ])
    
        
        batch_size, N_pde, N_bc = 40, 40, 1
        init_learning_rate = 5e-4
        u_num_layers, u_hidden_units = 4, 32
        u_activation_fn = nn.Tanh()
        D_num_layers, D_hidden_units = 4, 32
        D_activation_fn = nn.Tanh()
        V_num_layers, V_hidden_units = 4, 32
        V_activation_fn = nn.Tanh() 

        for i in range(len(weights)):
            data_weight, bc_weight, pde_weight = weights[i]
            save_dir = results_dir + f'loss_weightings_noise{noise_level}/data{data_weight}_PDE{pde_weight}_BC{bc_weight}/' 
            os.makedirs(save_dir, exist_ok=True)

            if i == 0: # save the datasets only once
                torch.save(train_dataset, results_dir + f'loss_weightings_noise{noise_level}/train_dataset.pt')
                torch.save(val_dataset, results_dir + f'loss_weightings_noise{noise_level}/val_dataset.pt')
                torch.save(test_dataset, results_dir + f'loss_weightings_noise{noise_level}/test_dataset.pt')
                torch.save(dataset, results_dir + f'loss_weightings_noise{noise_level}/full_dataset.pt')

            TrainBINN(save_dir, device, D, t_end, batch_size, Numrep, Num_epochs, u_num_layers, u_hidden_units, u_activation_fn, \
              D_num_layers, D_hidden_units, D_activation_fn, init_learning_rate, data_weight, bc_weight, pde_weight, \
                N_pde, N_bc, store_loss_step, grid_points_tensor, U_grid, U_grid_tensor, t_train, x_train, u_true, X, T, Nt, u_min, u_max, \
                    N_bio, u_bio_weight, D_bio_weight, V_num_layers, V_hidden_units, V_activation_fn, V_true, bio_constraint_bool, \
                        val_dataset, test_dataset, train_dataset, x_max, x_min, t_min)
            
    if benchmark_job == 2: # learning rates
        init_learning_rates = [
            5e-1, 
            5e-2, 
            5e-3, 
            5e-4, 
            5e-5, 
            5e-6, 
            5e-7
            ]

        batch_size, N_pde, N_bc = 40, 40, 40
        data_weight, pde_weight, bc_weight= 1.0, 1e-1, 1e-3
        u_num_layers, u_hidden_units = 4, 32
        u_activation_fn = nn.Tanh()
        D_num_layers, D_hidden_units = 4, 32
        D_activation_fn = nn.Tanh()
        V_num_layers, V_hidden_units = 4, 32
        V_activation_fn = nn.Tanh() 

        os.makedirs(results_dir + f'Const_lr_var{variance}/', exist_ok=True)
        torch.save(train_dataset, results_dir + f'Const_lr_var{variance}/train_dataset.pt')
        torch.save(val_dataset, results_dir + f'Const_lr_var{variance}/val_dataset.pt')
        torch.save(test_dataset, results_dir + f'Const_lr_var{variance}/test_dataset.pt')
        torch.save(dataset, results_dir + f'Const_lr_var{variance}/full_dataset.pt')  
        
        Num_epochs = 4001

        for init_learning_rate in init_learning_rates:
            save_dir = results_dir + f'Const_lr_var{variance}/{init_learning_rate}/' 
            os.makedirs(save_dir, exist_ok=True)

            TrainBINN(save_dir, device, D, t_end, batch_size, Numrep, Num_epochs, u_num_layers, u_hidden_units, u_activation_fn, \
              D_num_layers, D_hidden_units, D_activation_fn, init_learning_rate, data_weight, bc_weight, pde_weight, \
                N_pde, N_bc, store_loss_step, grid_points_tensor, U_grid, U_grid_tensor, t_train, x_train, u_true, X, T, Nt, u_min, u_max, \
                    N_bio, u_bio_weight, D_bio_weight, V_num_layers, V_hidden_units, V_activation_fn, V_true, bio_constraint_bool, \
                        val_dataset, test_dataset, train_dataset, x_max, x_min, t_min)
            
    if benchmark_job == 3: # DV network structures
        # num_layers, hidden_units
        DV_structures = np.array([
                                [1, 1], 
                                #[2, 1],
                                #[3, 1],
                                #[3, 16], 
                                [4, 1], 
                                [4, 4], 
                                [4, 32], 
                                [4, 64], 
                                #[5, 16], 
                                ])
        
        batch_size, N_pde, N_bc = 40, 40, 1
        data_weight, pde_weight, bc_weight= 1.0, 1e-1, 0.0
        init_learning_rate = 5e-4
        u_num_layers, u_hidden_units = 4, 32 
        u_activation_fn = nn.Tanh()

        for i in range(len(DV_structures)):
            num_layers, hidden_units = DV_structures[i]
            num_layers, hidden_units = int(num_layers), int(hidden_units)
            save_dir = results_dir + f'DV_NN_structure_noise{noise_level}/hlayer{num_layers}_node{hidden_units}/' 
            os.makedirs(save_dir, exist_ok=True)

            if i == 0: # save the datasets only once
                torch.save(train_dataset, results_dir + f'DV_NN_structure_noise{noise_level}/train_dataset.pt')
                torch.save(val_dataset, results_dir + f'DV_NN_structure_noise{noise_level}/val_dataset.pt')
                torch.save(test_dataset, results_dir + f'DV_NN_structure_noise{noise_level}/test_dataset.pt')
                torch.save(dataset, results_dir + f'DV_NN_structure_noise{noise_level}/full_dataset.pt')

            D_num_layers, D_hidden_units = num_layers, hidden_units
            V_num_layers, V_hidden_units = num_layers, hidden_units
            if num_layers * hidden_units < 2: 
                D_activation_fn = nn.Identity()  
                V_activation_fn = nn.Identity()  
            else:
                D_activation_fn = nn.Tanh()
                V_activation_fn = nn.Tanh()

            TrainBINN(save_dir, device, D, t_end, batch_size, Numrep, Num_epochs, u_num_layers, u_hidden_units, u_activation_fn, \
              D_num_layers, D_hidden_units, D_activation_fn, init_learning_rate, data_weight, bc_weight, pde_weight, \
                N_pde, N_bc, store_loss_step, grid_points_tensor, U_grid, U_grid_tensor, t_train, x_train, u_true, X, T, Nt, u_min, u_max, \
                    N_bio, u_bio_weight, D_bio_weight, V_num_layers, V_hidden_units, V_activation_fn, V_true, bio_constraint_bool, \
                        val_dataset, test_dataset, train_dataset, x_max, x_min, t_min)

    if benchmark_job == 4: # DV network structures
        # D_num_layers, D_hidden_units, V_num_layers, V_hidden_units, 
        DV_structures = np.array([
                                [1, 1, 1, 1], 
                                [1, 1, 4, 1],
                                #[3, 1],
                                #[3, 16], 
                                [4, 1], 
                                [4, 4], 
                                #[4, 32], 
                                #[4, 64], 
                                #[5, 16], 
                                ])
        
        batch_size, N_pde, N_bc = 40, 40, 1
        data_weight, pde_weight, bc_weight= 1.0, 1e-1, 0.0
        init_learning_rate = 5e-4
        u_num_layers, u_hidden_units = 4, 32 
        u_activation_fn = nn.Tanh()

        for i in range(len(DV_structures)):
            num_layers, hidden_units = DV_structures[i]
            num_layers, hidden_units = int(num_layers), int(hidden_units)
            save_dir = results_dir + f'DandV_NN_structure/hlayer{num_layers}_node{hidden_units}/' 
            os.makedirs(save_dir, exist_ok=True)

            if i == 0: # save the datasets only once
                torch.save(train_dataset, results_dir + f'DV_NN_structure/train_dataset.pt')
                torch.save(val_dataset, results_dir + f'DV_NN_structure/val_dataset.pt')
                torch.save(test_dataset, results_dir + f'DV_NN_structure/test_dataset.pt')
                torch.save(dataset, results_dir + f'DV_NN_structure/full_dataset.pt')

            D_num_layers, D_hidden_units = num_layers, hidden_units
            V_num_layers, V_hidden_units = num_layers, hidden_units
            if num_layers * hidden_units < 2: 
                D_activation_fn = nn.Identity()  
                V_activation_fn = nn.Identity()  
            else:
                D_activation_fn = nn.Tanh()
                V_activation_fn = nn.Tanh()

            TrainBINN(save_dir, device, D, t_end, batch_size, Numrep, Num_epochs, u_num_layers, u_hidden_units, u_activation_fn, \
              D_num_layers, D_hidden_units, D_activation_fn, init_learning_rate, data_weight, bc_weight, pde_weight, \
                N_pde, N_bc, store_loss_step, grid_points_tensor, U_grid, U_grid_tensor, t_train, x_train, u_true, X, T, Nt, u_min, u_max, \
                    N_bio, u_bio_weight, D_bio_weight, V_num_layers, V_hidden_units, V_activation_fn, V_true, bio_constraint_bool, \
                        val_dataset, test_dataset, train_dataset, x_max, x_min, t_min)


    if benchmark_job == 5: # batch_size but 10 repetitions
        # batch_size, N_pde, N_bc
        num_pts = np.array([
                            #[2, 40, 40], 
                            #[10, 40, 40], 
                            #[40, 40, 40],
                            #[160, 40, 40], 
                            [640, 40, 40], 
                            ])
        
        init_learning_rate = 5e-4
        u_num_layers, u_hidden_units = 4, 32
        u_activation_fn = nn.Tanh()
        D_num_layers, D_hidden_units = 4, 32
        D_activation_fn = nn.Tanh()
        V_num_layers, V_hidden_units = 4, 32
        V_activation_fn = nn.Tanh()  
        data_weight, pde_weight, bc_weight= 1.0, 1e-1, 0.0
        Numrep, Num_epochs = 1, int(1e10)

        for i in range(len(num_pts)):
            batch_size, N_pde, N_bc = num_pts[i]
            batch_size = int(batch_size) 
            N_pde = int(N_pde) 
            N_bc = int(N_bc) 

            os.makedirs(results_dir + f'batching_noise{noise_level}', exist_ok=True)
       
            save_dir = results_dir + f'batching_noise{noise_level}/FIXED5min_1e10epoch_batchsize{batch_size}_Npde{N_pde}_Nbc{N_bc}/'
            os.makedirs(save_dir, exist_ok=True)

            TrainBINN(save_dir, device, D, t_end, batch_size, Numrep, Num_epochs, u_num_layers, u_hidden_units, u_activation_fn, \
              D_num_layers, D_hidden_units, D_activation_fn, init_learning_rate, data_weight, bc_weight, pde_weight, \
                N_pde, N_bc, store_loss_step, grid_points_tensor, U_grid, U_grid_tensor, t_train, x_train, u_true, X, T, Nt, u_min, u_max, \
                    N_bio, u_bio_weight, D_bio_weight, V_num_layers, V_hidden_units, V_activation_fn, V_true, bio_constraint_bool, \
                        val_dataset, test_dataset, train_dataset, x_max, x_min, t_min)