# import libraries and set the device
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd
import os
import sys
from torch.utils.data import TensorDataset

from subfunction_ReactionDiffusion import *

# --- From Becky: Analytic FKPP traveling wave (Ablowitz–Zeppetella) ---
def u_fkpp_az(x, t, D, r, x0=0.0, c=None):
    """
    Analytic AZ solution of u_t = D u_xx + r u (1 - u)
    u(x,t) = 1 / (1 + exp( (x - x0 - c t) * sqrt(r/(6D)) ))^2
    valid only for c = (5/sqrt(6)) * sqrt(D r).
    """
    if c is None:
        c = (5.0/np.sqrt(6.0)) * np.sqrt(D*r)
    z = (x - x0 - c*t) * np.sqrt(r/(6.0*D))
    return 1.0 / (1.0 + np.exp(z))**2, c
# --------------------------------------------------------

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

    # space resolution
    L = 7.0
    x_min, x_max = -L, L
    dx = 0.04
    x_train = np.arange(x_min, x_max+dx, dx)
    Nx = len(x_train)

    # time resolution
    Nt, t_end, t_min = 7, 4.5, 0.0
    t_train = np.linspace(t_min, t_end, Nt)

    # fine solution for training data generation
    xx, tt = np.meshgrid(x_train, t_train)
    D, r = 0.1, 1.0
    #D, r = 0.2, 1.0
    ell = np.sqrt(6.0 * D / r)
    # Place the front near the center at t=0 and keep ~6*ell padding to edges
    x0 = 0.0
    u_train_nonoise, c = u_fkpp_az(xx, tt, D, r, x0=x0)

    ## YY: subsampling: 1/5th of the datapoints (in space) ----------------------------------
    x_train = x_train[::5]
    dx = x_train[1] - x_train[0]
    Nx = len(x_train)
    X, T = np.meshgrid(x_train, t_train)
    u_train_nonoise = u_train_nonoise[:, ::5]
    X, T = np.meshgrid(x_train, t_train)
    

    

    u_min_np, u_max_np = u_train_nonoise.min(), u_train_nonoise.max()
    u_true = u_train_nonoise.copy() # NOTE: no noise case
    shape = u_train_nonoise.shape

    # Add noise to the training data
    seed = 2 # for reproducibility
    np.random.seed(seed)
    #variances = 0.0
    #variances = 1e-2
    variances = 1e-3
    noise_level = np.sqrt(variances)
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

    #results_dir = f'Fisher_KPP_addnoise_var{variances}_D{D}_subsample/' 
    #results_dir = f'Fisher_KPP_addnoise_var{variances}_D{D}/' 
    #results_dir = f'Fisher_KPP_addnoise_var{variances}_D{D}_wbio0/' 
    results_dir = f'Fisher_KPP_addnoise_var{variances}_D{D}_subsample_wbio0/' 
    Numrep, Num_epochs, store_loss_step = 10, 4001, 1
    #bio_constraint_bool = True
    bio_constraint_bool = False
    G_true = r * (1 - U_grid)
    # save u_train_noise as a npy
    os.makedirs(results_dir, exist_ok=True)
    np.save(results_dir + 'u_train_noise.npy', u_train_noise)
    
    # ----------------------------------------------------------------------------------------------------

    benchmark_job = int(sys.argv[1]) 

    if benchmark_job == 0: # YY test
        save_dir = results_dir + 'test/' 
        os.makedirs(save_dir, exist_ok=True)

        batch_size, N_pde, N_bc = 40, 40, 40
        data_weight, pde_weight, bc_weight= 1.0, 1e-1, 1e-3
        init_learning_rate = 5e-4

        #bio_constraint_bool = True
        bio_constraint_bool = False
        u_bio_weight, D_bio_weight, G_bio_weight = 1.0, 1.0, 1.0
        N_bio = 40
        
        u_num_layers, u_hidden_units = 4, 32
        u_activation_fn = nn.Tanh()
        D_num_layers, D_hidden_units = 4, 32
        D_activation_fn = nn.Tanh()
        G_num_layers, G_hidden_units = 4, 32
        G_activation_fn = nn.Tanh()  

        # save training data, validation data, and testing data
        torch.save(train_dataset, results_dir + 'test/train_dataset.pt') 
        torch.save(val_dataset, results_dir + 'test/val_dataset.pt')
        torch.save(test_dataset, results_dir + 'test/test_dataset.pt')
        torch.save(dataset, results_dir + 'test/full_dataset.pt')
        
        TrainBINN(save_dir, device, D, t_end, batch_size, Numrep, Num_epochs, u_num_layers, u_hidden_units, u_activation_fn, \
              D_num_layers, D_hidden_units, D_activation_fn, init_learning_rate, data_weight, bc_weight, pde_weight, \
                N_pde, N_bc, store_loss_step, grid_points_tensor, U_grid, U_grid_tensor, t_train, x_train, u_true, X, T, Nt, u_min, u_max, \
                    N_bio, u_bio_weight, D_bio_weight, G_num_layers, G_hidden_units, G_activation_fn, G_true, G_bio_weight, bio_constraint_bool, \
                        val_dataset, test_dataset, train_dataset, x_max, x_min, t_min)

    if benchmark_job == 1: # network structures for u_net, need 'exclusive' arc sbatch 
        # u_num_layers, u_hidden_units
        u_structures = np.array([#[1, 32], # given 32 nodes, how many hidden layers?
                                #[2, 32], 
                                #[3, 32], 
                                #[4, 32],  
                                #[5, 32],  
                                
                                [4, 64],  # given 4 hidden layers, how many nodes per hidden layer?
                                [4, 128], 
                                [4, 16], 
                                [4, 8],
                                [4, 2]
                                ])

        batch_size, N_pde, N_bc, N_bio = 40, 40, 40, 40
        data_weight, pde_weight, bc_weight= 1e3, 10.0, 100.0
        #u_bio_weight, D_bio_weight, G_bio_weight = 10.0, 10.0, 10.0
        u_bio_weight, D_bio_weight, G_bio_weight = 0.0, 0.0, 0.0
        init_learning_rate = 5e-4
        u_activation_fn = nn.Tanh()
        D_num_layers, D_hidden_units = 1, 1
        D_activation_fn = nn.Identity()
        G_num_layers, G_hidden_units = 1, 1
        G_activation_fn = nn.Identity()  

        for i in range(len(u_structures)):
            u_num_layers, u_hidden_units = u_structures[i]
            u_num_layers = int(u_num_layers)
            u_hidden_units = int(u_hidden_units)
            save_dir = results_dir + f'u_NN_structure_noBioLoss/hlayer{u_num_layers}_node{u_hidden_units}/' 
            os.makedirs(save_dir, exist_ok=True)

            TrainBINN(save_dir, device, D, t_end, batch_size, Numrep, Num_epochs, u_num_layers, u_hidden_units, u_activation_fn, \
              D_num_layers, D_hidden_units, D_activation_fn, init_learning_rate, data_weight, bc_weight, pde_weight, \
                N_pde, N_bc, store_loss_step, grid_points_tensor, U_grid, U_grid_tensor, t_train, x_train, u_true, X, T, Nt, u_min, u_max, \
                    N_bio, u_bio_weight, D_bio_weight, G_num_layers, G_hidden_units, G_activation_fn, G_true, G_bio_weight, bio_constraint_bool, \
                        val_dataset, test_dataset, train_dataset, x_max, x_min, t_min)

    if benchmark_job == 2: # network structures for G_net, need 'exclusive' arc sbatch 
        # G_num_layers, G_hidden_units
        G_structures = np.array([[1, 1], # minimial MLP
                                [3, 1], 
                                [5, 1],  
                                [2, 1], # given 2 hidden layers, how many nodes per hidden layer?
                                [2, 4], 
                                [2, 16], 
                                [2, 32], 
                                [4, 1],  # given 4 hidden layers, how many nodes per hidden layer?
                                [4, 4], 
                                [4, 16],
                                [4, 32]
                                ])
        batch_size, N_pde, N_bc, N_bio = 40, 40, 40, 40
        data_weight, pde_weight, bc_weight= 1e3, 10.0, 100.0
        u_bio_weight, D_bio_weight, G_bio_weight = 10.0, 10.0, 10.0
        init_learning_rate = 5e-4
        u_num_layers, u_hidden_units = 4, 32
        u_activation_fn = nn.Tanh()
        D_num_layers, D_hidden_units = 1, 1
        D_activation_fn = nn.Identity()

        for i in range(len(G_structures)):
            G_num_layers, G_hidden_units = G_structures[i]
            if G_num_layers * G_hidden_units < 2: 
                G_activation_fn = nn.Identity()  
            else:
                G_activation_fn = nn.Tanh()
            G_num_layers = int(G_num_layers)
            G_hidden_units = int(G_hidden_units)
            save_dir = results_dir + f'G_NN_structure/hlayer{G_num_layers}_node{G_hidden_units}/' 
            os.makedirs(save_dir, exist_ok=True)

            TrainBINN(save_dir, device, D, t_end, batch_size, Numrep, Num_epochs, u_num_layers, u_hidden_units, u_activation_fn, \
              D_num_layers, D_hidden_units, D_activation_fn, init_learning_rate, data_weight, bc_weight, pde_weight, \
                N_pde, N_bc, store_loss_step, grid_points_tensor, U_grid, U_grid_tensor, t_train, x_train, u_true, X, T, Nt, u_min, u_max, \
                    N_bio, u_bio_weight, D_bio_weight, G_num_layers, G_hidden_units, G_activation_fn, G_true, G_bio_weight, bio_constraint_bool, \
                        val_dataset, test_dataset, train_dataset, x_max, x_min, t_min)

    if benchmark_job == 3: # number of training and grid data points when calculating loss terms, need 'exclusive' arc sbatch
        # batch_size, N_pde, N_bc, N_bio
        num_pts = np.array([[10, 10, 10, 10], # small batch size (more iterations to go through in minibatch)
                            [40, 40, 40, 40], # medium case
                            [100, 100, 100, 100], # big batch size
                            [100, 40, 40, 40], # more training data to compute data loss (less iterations to go through in minibatch)
                            [40, 100, 40, 40], # more grid data to compute PDE loss
                            [40, 40, 100, 40], # more grid data to compute BC loss
                            [40, 40, 40, 100], # more grid data to compute constraint loss
                            [40, 500, 500, 500], # when we elevate the power of BINN and sample more grid points for PDE and BC loss
                            ])
        
        data_weight, pde_weight, bc_weight= 1e3, 10.0, 100.0
        u_bio_weight, D_bio_weight, G_bio_weight = 10.0, 10.0, 10.0
        init_learning_rate = 5e-4
        u_num_layers, u_hidden_units = 4, 32
        u_activation_fn = nn.Tanh()
        D_num_layers, D_hidden_units = 1, 1
        D_activation_fn = nn.Identity()
        G_num_layers, G_hidden_units = 1, 1
        G_activation_fn = nn.Identity() 

        for i in range(len(num_pts)):
            batch_size, N_pde, N_bc, N_bio = num_pts[i]
            batch_size = int(batch_size) 
            N_pde = int(N_pde) 
            N_bc = int(N_bc) 
            N_bio = int(N_bio)
            
            save_dir = results_dir + f'batching/batchsize{batch_size}_Npde{N_pde}_Nbc{N_bc}_Nbio{N_bio}/'
            os.makedirs(save_dir, exist_ok=True)

            TrainBINN(save_dir, device, D, t_end, batch_size, Numrep, Num_epochs, u_num_layers, u_hidden_units, u_activation_fn, \
              D_num_layers, D_hidden_units, D_activation_fn, init_learning_rate, data_weight, bc_weight, pde_weight, \
                N_pde, N_bc, store_loss_step, grid_points_tensor, U_grid, U_grid_tensor, t_train, x_train, u_true, X, T, Nt, u_min, u_max, \
                    N_bio, u_bio_weight, D_bio_weight, G_num_layers, G_hidden_units, G_activation_fn, G_true, G_bio_weight, bio_constraint_bool, \
                        val_dataset, test_dataset, train_dataset, x_max, x_min, t_min)

    if benchmark_job == 4: # weightings on loss terms
        # data_weight, bc_weight, pde_weight, u_bio_weight, D_bio_weight, G_bio_weight
        weights = np.array([
                            [1.0, 1e-3, 1.0, 1.0, 1.0, 1.0], 
                            [1.0, 1e-2, 1.0, 1.0, 1.0, 1.0], 
                            [1.0, 1e-1, 1.0, 1.0, 1.0, 1.0], 
                            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 
                            [1.0, 1e1, 1.0, 1.0, 1.0, 1.0], 
                            [1.0, 1e2, 1.0, 1.0, 1.0, 1.0], 
                            [1.0, 1e3, 1.0, 1.0, 1.0, 1.0], 


                            [1.0, 1.0, 1e-3, 1.0, 1.0, 1.0], 
                            [1.0, 1.0, 1e-2, 1.0, 1.0, 1.0], 
                            [1.0, 1.0, 1e-1, 1.0, 1.0, 1.0], 
                            [1.0, 1.0, 1e1, 1.0, 1.0, 1.0], 
                            [1.0, 1.0, 1e2, 1.0, 1.0, 1.0], 
                            [1.0, 1.0, 1e3, 1.0, 1.0, 1.0], 

                           ])
        
        batch_size, N_pde, N_bc, N_bio = 40, 40, 40, 40
        init_learning_rate = 5e-4
        u_num_layers, u_hidden_units = 4, 32
        u_activation_fn = nn.Tanh()
        D_num_layers, D_hidden_units = 4, 32
        D_activation_fn = nn.Tanh()
        G_num_layers, G_hidden_units = 4, 32
        G_activation_fn = nn.Tanh()  

        bio_constraint_bool = True
        Numrep = 10

        for i in range(len(weights)):
            data_weight, bc_weight, pde_weight, u_bio_weight, D_bio_weight, G_bio_weight = weights[i]
            save_dir = results_dir + f'loss_weightings/data{data_weight}_PDE{pde_weight}_BC{bc_weight}/' 
            os.makedirs(save_dir, exist_ok=True)

            if i == 0: # save training data, validation data, and testing data
                torch.save(train_dataset, results_dir + 'loss_weightings/train_dataset.pt')
                torch.save(val_dataset, results_dir + 'loss_weightings/val_dataset.pt')
                torch.save(test_dataset, results_dir + 'loss_weightings/test_dataset.pt')
                torch.save(dataset, results_dir + 'loss_weightings/full_dataset.pt')

            TrainBINN(save_dir, device, D, t_end, batch_size, Numrep, Num_epochs, u_num_layers, u_hidden_units, u_activation_fn, \
              D_num_layers, D_hidden_units, D_activation_fn, init_learning_rate, data_weight, bc_weight, pde_weight, \
                N_pde, N_bc, store_loss_step, grid_points_tensor, U_grid, U_grid_tensor, t_train, x_train, u_true, X, T, Nt, u_min, u_max, \
                    N_bio, u_bio_weight, D_bio_weight, G_num_layers, G_hidden_units, G_activation_fn, G_true, G_bio_weight, bio_constraint_bool, \
                        val_dataset, test_dataset, train_dataset, x_max, x_min, t_min)

    if benchmark_job == 5: # learning rates
        init_learning_rates = [5e-1, 
                               5e-2, 
                               5e-3, 
                               5e-4, 
                               5e-5, 
                               5e-6, 
                               5e-7]

        batch_size, N_pde, N_bc = 40, 40, 40
        data_weight, pde_weight, bc_weight= 1.0, 1e-1, 1.0

        #u_bio_weight, D_bio_weight, G_bio_weight = 10.0, 10.0, 10.0
        bio_constraint_bool = False
        u_bio_weight, D_bio_weight, G_bio_weight = 0.0, 0.0, 0.0
        N_bio = 1
        
        u_num_layers, u_hidden_units = 4, 32
        u_activation_fn = nn.Tanh()
        D_num_layers, D_hidden_units = 1, 1
        D_activation_fn = nn.Identity()
        G_num_layers, G_hidden_units = 1, 1
        G_activation_fn = nn.Identity()  

        for init_learning_rate in init_learning_rates:
            save_dir = results_dir + f'Const_lr_noBioLoss_minDGStruct/{init_learning_rate}/' 
            os.makedirs(save_dir, exist_ok=True)

            # save training data, validation data, and testing data
            torch.save(train_dataset, results_dir + 'Const_lr_noBioLoss_minDGStruct/train_dataset.pt')
            torch.save(val_dataset, results_dir + 'Const_lr_noBioLoss_minDGStruct/val_dataset.pt')
            torch.save(test_dataset, results_dir + 'Const_lr_noBioLoss_minDGStruct/test_dataset.pt')
            torch.save(dataset, results_dir + 'Const_lr_noBioLoss_minDGStruct/full_dataset.pt')

            TrainBINN(save_dir, device, D, t_end, batch_size, Numrep, Num_epochs, u_num_layers, u_hidden_units, u_activation_fn, \
              D_num_layers, D_hidden_units, D_activation_fn, init_learning_rate, data_weight, bc_weight, pde_weight, \
                N_pde, N_bc, store_loss_step, grid_points_tensor, U_grid, U_grid_tensor, t_train, x_train, u_true, X, T, Nt, u_min, u_max, \
                    N_bio, u_bio_weight, D_bio_weight, G_num_layers, G_hidden_units, G_activation_fn, G_true, G_bio_weight, bio_constraint_bool, \
                        val_dataset, test_dataset, train_dataset, x_max, x_min, t_min)

    if benchmark_job == 6: # network structures for D_net, need 'exclusive' arc sbatch 
        # D_num_layers, D_hidden_units
        D_structures = np.array([#[1, 1], # minimial MLP
                                #[3, 1], 
                                #[5, 1],  
                                #[2, 1], # given 2 hidden layers, how many nodes per hidden layer?
                                #[2, 4], 

                                [2, 16], 
                                [2, 32], 
                                [4, 1],  # given 4 hidden layers, how many nodes per hidden layer?
                                [4, 4], 
                                [4, 16],
                                [4, 32]
                                ])
        batch_size, N_pde, N_bc, N_bio = 40, 40, 40, 40
        data_weight, pde_weight, bc_weight= 1e3, 10.0, 100.0
        #u_bio_weight, D_bio_weight, G_bio_weight = 10.0, 10.0, 10.0
        u_bio_weight, D_bio_weight, G_bio_weight = 0.0, 0.0, 0.0
        init_learning_rate = 5e-4
        u_num_layers, u_hidden_units = 4, 32
        u_activation_fn = nn.Tanh()
        G_num_layers, G_hidden_units = 1, 1
        G_activation_fn = nn.Identity()  

        for i in range(len(D_structures)):
            D_num_layers, D_hidden_units = D_structures[i]
            if D_num_layers * D_hidden_units < 2: 
                D_activation_fn = nn.Identity()  
            else:
                D_activation_fn = nn.Tanh()
            D_num_layers = int(D_num_layers)
            D_hidden_units = int(D_hidden_units)
            save_dir = results_dir + f'D_NN_structure_noBioLoss/hlayer{D_num_layers}_node{D_hidden_units}/' 
            os.makedirs(save_dir, exist_ok=True)

            TrainBINN(save_dir, device, D, t_end, batch_size, Numrep, Num_epochs, u_num_layers, u_hidden_units, u_activation_fn, \
              D_num_layers, D_hidden_units, D_activation_fn, init_learning_rate, data_weight, bc_weight, pde_weight, \
                N_pde, N_bc, store_loss_step, grid_points_tensor, U_grid, U_grid_tensor, t_train, x_train, u_true, X, T, Nt, u_min, u_max, \
                    N_bio, u_bio_weight, D_bio_weight, G_num_layers, G_hidden_units, G_activation_fn, G_true, G_bio_weight, bio_constraint_bool, \
                        val_dataset, test_dataset, train_dataset, x_max, x_min, t_min)

    if benchmark_job == 7: # complex network structures for D_net and G_net
        # D_num_layers, D_hidden_units, G_num_layers, G_hidden_units
        DG_structures = np.array([
                                [2, 1, 2, 1], 
                                [1, 2, 1, 2], 
                                [4, 4, 4, 4], 
                                [4, 32, 4, 32], 
                                [1, 1, 1, 1]
                                ])
        batch_size, N_pde, N_bc, N_bio = 40, 40, 40, 40
        data_weight, pde_weight, bc_weight= 1.0, 1e-1, 1e-3
        u_bio_weight, D_bio_weight, G_bio_weight = 1.0, 1.0, 1.0
        init_learning_rate = 5e-4
        u_num_layers, u_hidden_units = 4, 32
        u_activation_fn = nn.Tanh()
        
        Numrep = 10

        for i in range(len(DG_structures)):
            D_num_layers, D_hidden_units, G_num_layers, G_hidden_units = DG_structures[i]
            D_num_layers = int(D_num_layers)
            D_hidden_units = int(D_hidden_units)
            G_num_layers = int(G_num_layers)
            G_hidden_units = int(G_hidden_units)

            if D_num_layers * D_hidden_units < 2: 
                D_activation_fn = nn.Identity()  
            else:
                D_activation_fn = nn.Tanh()
            if G_num_layers * G_hidden_units < 2: 
                G_activation_fn = nn.Identity()  
            else:
                G_activation_fn = nn.Tanh()

            save_dir = results_dir + f'DG_NN_structure_noBioLoss/Dhlayer{D_num_layers}_Dnode{D_hidden_units}_Ghlayer{G_num_layers}_Gnode{G_hidden_units}/' 
            os.makedirs(save_dir, exist_ok=True)

            TrainBINN(save_dir, device, D, t_end, batch_size, Numrep, Num_epochs, u_num_layers, u_hidden_units, u_activation_fn, \
              D_num_layers, D_hidden_units, D_activation_fn, init_learning_rate, data_weight, bc_weight, pde_weight, \
                N_pde, N_bc, store_loss_step, grid_points_tensor, U_grid, U_grid_tensor, t_train, x_train, u_true, X, T, Nt, u_min, u_max, \
                    N_bio, u_bio_weight, D_bio_weight, G_num_layers, G_hidden_units, G_activation_fn, G_true, G_bio_weight, bio_constraint_bool, \
                        val_dataset, test_dataset, train_dataset, x_max, x_min, t_min)

    
    if benchmark_job == 8: # learning rates with complex D and G networks
        init_learning_rates = [
                               #5e-1, 
                               #5e-2, 
                               #5e-3, 
                               5e-4, 
                               #5e-5, 
                               #5e-6, 
                               #5e-7
                               ]

        batch_size, N_pde, N_bc = 40, 40, 40
        data_weight, pde_weight, bc_weight= 1.0, 1e-1, 1e-3 

        #bio_constraint_bool = False
        #u_bio_weight, D_bio_weight, G_bio_weight = 0.0, 0.0, 0.0
        #N_bio = 1
        bio_constraint_bool = True
        u_bio_weight, D_bio_weight, G_bio_weight = 1.0, 1.0, 1.0
        N_bio = 40
        
        u_num_layers, u_hidden_units = 4, 32
        u_activation_fn = nn.Tanh()
        D_num_layers, D_hidden_units = 4, 32
        D_activation_fn = nn.Tanh()
        G_num_layers, G_hidden_units = 4, 32
        G_activation_fn = nn.Tanh()  

        Numrep = 10

        for init_learning_rate in init_learning_rates:
            save_dir = results_dir + f'Const_lr_BioLoss_DGStruct_4_32_BC1e-3/{init_learning_rate}/' 
            os.makedirs(save_dir, exist_ok=True)

            # save training data, validation data, and testing data
            torch.save(train_dataset, results_dir + 'Const_lr_BioLoss_DGStruct_4_32_BC1e-3/train_dataset.pt')
            torch.save(val_dataset, results_dir + 'Const_lr_BioLoss_DGStruct_4_32_BC1e-3/val_dataset.pt')
            torch.save(test_dataset, results_dir + 'Const_lr_BioLoss_DGStruct_4_32_BC1e-3/test_dataset.pt')
            torch.save(dataset, results_dir + 'Const_lr_BioLoss_DGStruct_4_32_BC1e-3/full_dataset.pt')

            TrainBINN(save_dir, device, D, t_end, batch_size, Numrep, Num_epochs, u_num_layers, u_hidden_units, u_activation_fn, \
              D_num_layers, D_hidden_units, D_activation_fn, init_learning_rate, data_weight, bc_weight, pde_weight, \
                N_pde, N_bc, store_loss_step, grid_points_tensor, U_grid, U_grid_tensor, t_train, x_train, u_true, X, T, Nt, u_min, u_max, \
                    N_bio, u_bio_weight, D_bio_weight, G_num_layers, G_hidden_units, G_activation_fn, G_true, G_bio_weight, bio_constraint_bool, \
                        val_dataset, test_dataset, train_dataset, x_max, x_min, t_min)