# import libraries and set the device
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd
import os
import sys
from torch.utils.data import TensorDataset

#from subfunction_Diffusion_profiling import *

from subfunction_Diffusion_profiling_FIXEDTIME import * # this is for Fig 7 fixed time L2RE. 

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

    # D = 0.1

    # # space resolution
    # Nx, L = 120, 1
    # x_min, x_max = -L, L
    # x_train = np.linspace(x_min, x_max, Nx)

    # # time resolution
    # Nt, t_end, t_min = 5, 2, 0.0
    # t_train = np.linspace(t_min, t_end, Nt)

    D = 0.5

    # space resolution
    L = 7.0
    x_min, x_max = -L, L
    dx = 0.04
    x_train = np.arange(x_min, x_max+dx, dx)
    Nx = len(x_train)

    # time resolution
    Nt, t_end, t_min = 7, 18, 0.0
    t_train = np.linspace(t_min, t_end, Nt)

    xx, tt = np.meshgrid(x_train, t_train)
    # generate the training data
    u_train_nonoise_ = 0.5 * np.cos(np.pi * xx / L) * np.exp(-tt * D * (np.pi**2) / (L**2)) + 0.5 # so that u in [0, 1]

    ## YY: subsampling: 1/5th of the datapoints (in space) ----------------------------------
    x_train = x_train[::5]
    dx = x_train[1] - x_train[0]
    Nx = len(x_train)
    X, T = np.meshgrid(x_train, t_train)
    u_train_nonoise = u_train_nonoise_[:, ::5]


    u_min_np, u_max_np = u_train_nonoise.min(), u_train_nonoise.max()
    u_true = u_train_nonoise.copy() # NOTE: no noise case
    shape = u_train_nonoise.shape

    # Add noise to the training data
    seed = 8 # for reproducibility
    np.random.seed(seed)
    #noise_level = 0.0 
    #noise_level = 0.05
    #noise_level = 0.5
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

    results_dir = f'Diffusion_linear_subsample/' 
    os.makedirs(results_dir, exist_ok=True)


    Numrep, Num_epochs, store_loss_step = 10, 4001, 1
    bio_constraint_bool = False # no biology constraint for u, D, and G
    N_bio = 1 # not used
    u_bio_weight, D_bio_weight, G_bio_weight = 1.0, 1.0, 0.0

    # save u_train_noise
    np.save(results_dir + f'u_train_var{variance}.npy', u_train_noise)

    
    # ----------------------------------------------------------------------------------------------------

    benchmark_job = int(sys.argv[1]) 

    if benchmark_job == 0: # YY test
        save_dir = results_dir + 'test/' 
        os.makedirs(save_dir, exist_ok=True)

        batch_size, N_pde, N_bc = 40, 40, 40
        data_weight, pde_weight, bc_weight= 1.0, 1e-1, 1e-3
        init_learning_rate = 5e-4
        u_num_layers, u_hidden_units = 4, 32
        u_activation_fn = nn.Tanh()
        D_num_layers, D_hidden_units = 4, 32
        D_activation_fn = nn.Tanh()
        Numrep, Num_epochs, store_loss_step = 1, 4001, 1
        
        TrainBINN(save_dir, device, D, t_end, batch_size, Numrep, Num_epochs, u_num_layers, u_hidden_units, u_activation_fn, \
              D_num_layers, D_hidden_units, D_activation_fn, init_learning_rate, data_weight, bc_weight, pde_weight, \
                N_pde, N_bc, store_loss_step, grid_points_tensor, U_grid, U_grid_tensor, t_train, x_train, u_true, X, T, Nt, u_min, u_max, \
                    N_bio, u_bio_weight, D_bio_weight, bio_constraint_bool, val_dataset, test_dataset, train_dataset, x_max, x_min, t_min)
        

    if benchmark_job == 1: # network structures for u_net, need 'exclusive' arc sbatch 
        # u_num_layers, u_hidden_units
        u_structures = np.array([
                                [1, 4],
                                [1, 16], 
                                #[2, 32], 
                                #[3, 32], 
                                [4, 16],  
                                #[6, 16],  
                                [4, 64],  # given 4 hidden layers, how many nodes per hidden layer?
                                [4, 128], 
                                #[4, 16], 
                                #[4, 8],
                                [4, 4]
                                ])

        batch_size, N_pde, N_bc = 40, 40, 40
        data_weight, pde_weight, bc_weight= 1.0, 1e-1, 1e-3
        init_learning_rate = 5e-4
        u_activation_fn = nn.Tanh()
        D_num_layers, D_hidden_units = 4, 32
        D_activation_fn = nn.Tanh()      

        for i in range(len(u_structures)):
            u_num_layers, u_hidden_units = u_structures[i]
            u_num_layers = int(u_num_layers)
            u_hidden_units = int(u_hidden_units)
            save_dir = results_dir + f'u_NN_structure/hlayer{u_num_layers}_node{u_hidden_units}/' 
            os.makedirs(save_dir, exist_ok=True)

            TrainBINN(save_dir, device, D, t_end, batch_size, Numrep, Num_epochs, u_num_layers, u_hidden_units, u_activation_fn, \
              D_num_layers, D_hidden_units, D_activation_fn, init_learning_rate, data_weight, bc_weight, pde_weight, \
                N_pde, N_bc, store_loss_step, grid_points_tensor, U_grid, U_grid_tensor, t_train, x_train, u_true, X, T, Nt, u_min, u_max, \
                    N_bio, u_bio_weight, D_bio_weight, bio_constraint_bool, val_dataset, test_dataset, train_dataset, x_max, x_min, t_min)

    if benchmark_job == 2: # network structures for D_net, need 'exclusive' arc sbatch 
        # D_num_layers, D_hidden_units
        D_structures = np.array([
                                [1, 1], # minimial MLP
                                #[3, 1], 
                                #[5, 1],  
                                #[2, 1], # given 2 hidden layers, how many nodes per hidden layer?
                                
                                #[2, 4], 
                                #[2, 16], 
                                #[2, 32], 
                                [4, 1],  # given 4 hidden layers, how many nodes per hidden layer?
                                
                                [4, 4], 
                                #[4, 16],
                                [4, 32], 
                                [4, 64],
                                ])
        batch_size, N_pde, N_bc = 40, 40, 40
        data_weight, pde_weight, bc_weight= 1.0, 1e-1, 1e-3
        init_learning_rate = 5e-4
        u_num_layers, u_hidden_units = 4, 32
        u_activation_fn = nn.Tanh()

        for i in range(len(D_structures)):
            D_num_layers, D_hidden_units = D_structures[i]
            if D_num_layers * D_hidden_units < 2: 
                D_activation_fn = nn.Identity()  
            else:
                D_activation_fn = nn.Tanh()
            D_num_layers = int(D_num_layers)
            D_hidden_units = int(D_hidden_units)
            save_dir = results_dir + f'D_NN_structure_noise{noise_level}/hlayer{D_num_layers}_node{D_hidden_units}/' 
            os.makedirs(save_dir, exist_ok=True)

            TrainBINN(save_dir, device, D, t_end, batch_size, Numrep, Num_epochs, u_num_layers, u_hidden_units, u_activation_fn, \
              D_num_layers, D_hidden_units, D_activation_fn, init_learning_rate, data_weight, bc_weight, pde_weight, \
                N_pde, N_bc, store_loss_step, grid_points_tensor, U_grid, U_grid_tensor, t_train, x_train, u_true, X, T, Nt, u_min, u_max, \
                    N_bio, u_bio_weight, D_bio_weight, bio_constraint_bool, val_dataset, test_dataset, train_dataset, x_max, x_min, t_min)
            
    if benchmark_job == 3: # number of training and grid data points when calculating loss terms, need 'exclusive' arc sbatch
        # batch_size, N_pde, N_bc
        num_pts = np.array([[10, 40, 40], 
                            [40, 40, 40],
                            [160, 40, 40], 
                            [640, 40, 40], 
                            ])
        
        data_weight, pde_weight, bc_weight= 1.0, 1e-1, 1e-3
        init_learning_rate = 5e-4
        u_num_layers, u_hidden_units = 4, 32
        u_activation_fn = nn.Tanh()
        D_num_layers, D_hidden_units = 1, 1
        D_activation_fn = nn.Identity()     

        Numrep = 1 

        for i in range(len(num_pts)):
            batch_size, N_pde, N_bc = num_pts[i]
            batch_size = int(batch_size) 
            N_pde = int(N_pde) 
            N_bc = int(N_bc) 
            
            save_dir = results_dir + f'batching_D_minStruct/batchsize{batch_size}_Npde{N_pde}_Nbc{N_bc}/'
            os.makedirs(save_dir, exist_ok=True)

            if i == 0: # save the datasets only once
                torch.save(train_dataset, results_dir + 'batching_D_minStruct/train_dataset.pt')
                torch.save(val_dataset, results_dir + 'batching_D_minStruct/val_dataset.pt')
                torch.save(test_dataset, results_dir + 'batching_D_minStruct/test_dataset.pt')
                torch.save(dataset, results_dir + 'batching_D_minStruct/full_dataset.pt')

            TrainBINN(save_dir, device, D, t_end, batch_size, Numrep, Num_epochs, u_num_layers, u_hidden_units, u_activation_fn, \
              D_num_layers, D_hidden_units, D_activation_fn, init_learning_rate, data_weight, bc_weight, pde_weight, \
                N_pde, N_bc, store_loss_step, grid_points_tensor, U_grid, U_grid_tensor, t_train, x_train, u_true, X, T, Nt, u_min, u_max, \
                    N_bio, u_bio_weight, D_bio_weight, bio_constraint_bool, val_dataset, test_dataset, train_dataset, x_max, x_min, t_min)
            
    if benchmark_job == 33: # N_pde
        # batch_size, N_pde, N_bc
        num_pts = np.array([#[40, 10, 40], 
                            #[40, 40, 40],
                            #[40, 160, 40], 
                            [40, 640, 40], 
                            ])
        
        data_weight, pde_weight, bc_weight= 1.0, 1e-1, 1e-3
        init_learning_rate = 5e-4
        u_num_layers, u_hidden_units = 4, 32
        u_activation_fn = nn.Tanh()
        D_num_layers, D_hidden_units = 4, 32
        D_activation_fn = nn.Tanh()    

        Numrep = 10  

        for i in range(len(num_pts)):
            batch_size, N_pde, N_bc = num_pts[i]
            batch_size = int(batch_size) 
            N_pde = int(N_pde) 
            N_bc = int(N_bc) 
            
            save_dir = results_dir + f'batching_D_4_32/batchsize{batch_size}_Npde{N_pde}_Nbc{N_bc}/'
            os.makedirs(save_dir, exist_ok=True)

            if i == 0: # save the datasets only once
                torch.save(train_dataset, results_dir + 'batching_D_4_32/train_dataset.pt')
                torch.save(val_dataset, results_dir + 'batching_D_4_32/val_dataset.pt')
                torch.save(test_dataset, results_dir + 'batching_D_4_32/test_dataset.pt')
                torch.save(dataset, results_dir + 'batching_D_4_32/full_dataset.pt')

            TrainBINN(save_dir, device, D, t_end, batch_size, Numrep, Num_epochs, u_num_layers, u_hidden_units, u_activation_fn, \
              D_num_layers, D_hidden_units, D_activation_fn, init_learning_rate, data_weight, bc_weight, pde_weight, \
                N_pde, N_bc, store_loss_step, grid_points_tensor, U_grid, U_grid_tensor, t_train, x_train, u_true, X, T, Nt, u_min, u_max, \
                    N_bio, u_bio_weight, D_bio_weight, bio_constraint_bool, val_dataset, test_dataset, train_dataset, x_max, x_min, t_min)
            
    if benchmark_job == 333: # N_bc
        # batch_size, N_pde, N_bc
        num_pts = np.array([
                            [40, 40, 40],
                            [40, 40, 160], 
                            [40, 40, 640], 
                            [40, 40, 10], 
                            ])
        
        data_weight, pde_weight, bc_weight= 1.0, 1e-1, 1e-3
        init_learning_rate = 5e-4
        u_num_layers, u_hidden_units = 4, 32
        u_activation_fn = nn.Tanh()
        D_num_layers, D_hidden_units = 4, 32
        D_activation_fn = nn.Tanh()      

        Numrep = 10

        for i in range(len(num_pts)):
            batch_size, N_pde, N_bc = num_pts[i]
            batch_size = int(batch_size) 
            N_pde = int(N_pde) 
            N_bc = int(N_bc) 
            
            save_dir = results_dir + f'batching_D_4_32/batchsize{batch_size}_Npde{N_pde}_Nbc{N_bc}/'
            os.makedirs(save_dir, exist_ok=True)

            TrainBINN(save_dir, device, D, t_end, batch_size, Numrep, Num_epochs, u_num_layers, u_hidden_units, u_activation_fn, \
              D_num_layers, D_hidden_units, D_activation_fn, init_learning_rate, data_weight, bc_weight, pde_weight, \
                N_pde, N_bc, store_loss_step, grid_points_tensor, U_grid, U_grid_tensor, t_train, x_train, u_true, X, T, Nt, u_min, u_max, \
                    N_bio, u_bio_weight, D_bio_weight, bio_constraint_bool, val_dataset, test_dataset, train_dataset, x_max, x_min, t_min)         

    if benchmark_job == 3333: # batch_size but 10 repetitions
        # batch_size, N_pde, N_bc
        num_pts = np.array([
                            [40, 40, 40],
                            #[160, 40, 40], 
                            #[640, 40, 40], 
                            #[10, 40, 40], 
                            #[2, 40, 40], 
                            ])
        
        data_weight, pde_weight, bc_weight= 1.0, 1e-1, 1e-3
        init_learning_rate = 5e-4
        u_num_layers, u_hidden_units = 4, 32
        u_activation_fn = nn.Tanh()
        D_num_layers, D_hidden_units = 4, 32
        D_activation_fn = nn.Tanh()   

        #Numrep, Num_epochs = 10, 4000
        Numrep, Num_epochs = 1, int(1e10)
        
        for i in range(len(num_pts)):


            batch_size, N_pde, N_bc = num_pts[i]
            batch_size = int(batch_size) 
            N_pde = int(N_pde) 
            N_bc = int(N_bc) 
            
            #save_dir_i = results_dir + f'batching_D_4_32_noise{noise_level}/batchsize{batch_size}_Npde{N_pde}_Nbc{N_bc}/'
            save_dir_i = results_dir + f'batching_D_4_32_var{variance}/FIXED5min_1e10epoch_batchsize{batch_size}_Npde{N_pde}_Nbc{N_bc}/'
            
            os.makedirs(save_dir_i, exist_ok=True)

            TrainBINN(save_dir_i, device, D, t_end, batch_size, Numrep, Num_epochs, u_num_layers, u_hidden_units, u_activation_fn, \
              D_num_layers, D_hidden_units, D_activation_fn, init_learning_rate, data_weight, bc_weight, pde_weight, \
                N_pde, N_bc, store_loss_step, grid_points_tensor, U_grid, U_grid_tensor, t_train, x_train, u_true, X, T, Nt, u_min, u_max, \
                    N_bio, u_bio_weight, D_bio_weight, bio_constraint_bool, val_dataset, test_dataset, train_dataset, x_max, x_min, t_min)

    if benchmark_job == 33333: # batch_size but 10 repetitions
        # batch_size, N_pde, N_bc
        num_pts = np.array([
                            #[40, 40, 40],
                            [160, 40, 40], 
                            #[640, 40, 40], 
                            #[10, 40, 40], 
                            #[2, 40, 40], 
                            ])
        
        data_weight, pde_weight, bc_weight= 1.0, 1e-1, 1e-3
        init_learning_rate = 5e-4
        u_num_layers, u_hidden_units = 4, 32
        u_activation_fn = nn.Tanh()
        D_num_layers, D_hidden_units = 4, 32
        D_activation_fn = nn.Tanh()   

        #Numrep, Num_epochs = 10, 4000
        Numrep, Num_epochs = 1, int(1e10)
        
        for i in range(len(num_pts)):


            batch_size, N_pde, N_bc = num_pts[i]
            batch_size = int(batch_size) 
            N_pde = int(N_pde) 
            N_bc = int(N_bc) 
            
            #save_dir_i = results_dir + f'batching_D_4_32_var{variance}/batchsize{batch_size}_Npde{N_pde}_Nbc{N_bc}/'
            save_dir_i = results_dir + f'batching_D_4_32_var{variance}/FIXED5min_1e10epoch_batchsize{batch_size}_Npde{N_pde}_Nbc{N_bc}/'
            
            os.makedirs(save_dir_i, exist_ok=True)

            TrainBINN(save_dir_i, device, D, t_end, batch_size, Numrep, Num_epochs, u_num_layers, u_hidden_units, u_activation_fn, \
              D_num_layers, D_hidden_units, D_activation_fn, init_learning_rate, data_weight, bc_weight, pde_weight, \
                N_pde, N_bc, store_loss_step, grid_points_tensor, U_grid, U_grid_tensor, t_train, x_train, u_true, X, T, Nt, u_min, u_max, \
                    N_bio, u_bio_weight, D_bio_weight, bio_constraint_bool, val_dataset, test_dataset, train_dataset, x_max, x_min, t_min)

    if benchmark_job == 301: # batch_size but 10 repetitions
        # batch_size, N_pde, N_bc
        num_pts = np.array([
                            #[40, 40, 40],
                            #[160, 40, 40], 
                            [640, 40, 40], 
                            #[10, 40, 40], 
                            #[2, 40, 40], 
                            ])
        
        data_weight, pde_weight, bc_weight= 1.0, 1e-1, 1e-3
        init_learning_rate = 5e-4
        u_num_layers, u_hidden_units = 4, 32
        u_activation_fn = nn.Tanh()
        D_num_layers, D_hidden_units = 4, 32
        D_activation_fn = nn.Tanh()   

        #Numrep, Num_epochs = 10, 4000
        Numrep, Num_epochs = 1, int(1e10)
        
        for i in range(len(num_pts)):


            batch_size, N_pde, N_bc = num_pts[i]
            batch_size = int(batch_size) 
            N_pde = int(N_pde) 
            N_bc = int(N_bc) 
            
            #save_dir_i = results_dir + f'batching_D_4_32_var{variance}/batchsize{batch_size}_Npde{N_pde}_Nbc{N_bc}/'
            save_dir_i = results_dir + f'batching_D_4_32_var{variance}/FIXED5min_1e10epoch_batchsize{batch_size}_Npde{N_pde}_Nbc{N_bc}/'
            
            os.makedirs(save_dir_i, exist_ok=True)

            TrainBINN(save_dir_i, device, D, t_end, batch_size, Numrep, Num_epochs, u_num_layers, u_hidden_units, u_activation_fn, \
              D_num_layers, D_hidden_units, D_activation_fn, init_learning_rate, data_weight, bc_weight, pde_weight, \
                N_pde, N_bc, store_loss_step, grid_points_tensor, U_grid, U_grid_tensor, t_train, x_train, u_true, X, T, Nt, u_min, u_max, \
                    N_bio, u_bio_weight, D_bio_weight, bio_constraint_bool, val_dataset, test_dataset, train_dataset, x_max, x_min, t_min)

    if benchmark_job == 302: # batch_size but 10 repetitions
        # batch_size, N_pde, N_bc
        num_pts = np.array([
                            #[40, 40, 40],
                            #[160, 40, 40], 
                            #[640, 40, 40], 
                            [10, 40, 40], 
                            #[2, 40, 40], 
                            ])
        
        data_weight, pde_weight, bc_weight= 1.0, 1e-1, 1e-3
        init_learning_rate = 5e-4
        u_num_layers, u_hidden_units = 4, 32
        u_activation_fn = nn.Tanh()
        D_num_layers, D_hidden_units = 4, 32
        D_activation_fn = nn.Tanh()   

        #Numrep, Num_epochs = 10, 4000
        Numrep, Num_epochs = 1, int(1e10)
        
        for i in range(len(num_pts)):


            batch_size, N_pde, N_bc = num_pts[i]
            batch_size = int(batch_size) 
            N_pde = int(N_pde) 
            N_bc = int(N_bc) 
            
            #save_dir_i = results_dir + f'batching_D_4_32_var{variance}/batchsize{batch_size}_Npde{N_pde}_Nbc{N_bc}/'
            save_dir_i = results_dir + f'batching_D_4_32_var{variance}/FIXED5min_1e10epoch_batchsize{batch_size}_Npde{N_pde}_Nbc{N_bc}/'
            
            os.makedirs(save_dir_i, exist_ok=True)

            TrainBINN(save_dir_i, device, D, t_end, batch_size, Numrep, Num_epochs, u_num_layers, u_hidden_units, u_activation_fn, \
              D_num_layers, D_hidden_units, D_activation_fn, init_learning_rate, data_weight, bc_weight, pde_weight, \
                N_pde, N_bc, store_loss_step, grid_points_tensor, U_grid, U_grid_tensor, t_train, x_train, u_true, X, T, Nt, u_min, u_max, \
                    N_bio, u_bio_weight, D_bio_weight, bio_constraint_bool, val_dataset, test_dataset, train_dataset, x_max, x_min, t_min)

    if benchmark_job == 303: # batch_size but 10 repetitions
        # batch_size, N_pde, N_bc
        num_pts = np.array([
                            #[40, 40, 40],
                            #[160, 40, 40], 
                            #[640, 40, 40], 
                            #[10, 40, 40], 
                            [2, 40, 40], 
                            ])
        
        data_weight, pde_weight, bc_weight= 1.0, 1e-1, 1e-3
        init_learning_rate = 5e-4
        u_num_layers, u_hidden_units = 4, 32
        u_activation_fn = nn.Tanh()
        D_num_layers, D_hidden_units = 4, 32
        D_activation_fn = nn.Tanh()   

        #Numrep, Num_epochs = 10, 4000
        Numrep, Num_epochs = 1, int(1e10)
        
        for i in range(len(num_pts)):


            batch_size, N_pde, N_bc = num_pts[i]
            batch_size = int(batch_size) 
            N_pde = int(N_pde) 
            N_bc = int(N_bc) 
            
            #save_dir_i = results_dir + f'batching_D_4_32_var{variance}/batchsize{batch_size}_Npde{N_pde}_Nbc{N_bc}/'
            save_dir_i = results_dir + f'batching_D_4_32_var{variance}/FIXED5min_1e10epoch_batchsize{batch_size}_Npde{N_pde}_Nbc{N_bc}/'
            
            os.makedirs(save_dir_i, exist_ok=True)

            TrainBINN(save_dir_i, device, D, t_end, batch_size, Numrep, Num_epochs, u_num_layers, u_hidden_units, u_activation_fn, \
              D_num_layers, D_hidden_units, D_activation_fn, init_learning_rate, data_weight, bc_weight, pde_weight, \
                N_pde, N_bc, store_loss_step, grid_points_tensor, U_grid, U_grid_tensor, t_train, x_train, u_true, X, T, Nt, u_min, u_max, \
                    N_bio, u_bio_weight, D_bio_weight, bio_constraint_bool, val_dataset, test_dataset, train_dataset, x_max, x_min, t_min)
            

            
    if benchmark_job == 4: # weightings on loss terms
        # data_weight, bc_weight, pde_weight
        weights = np.array([
                            [1.0, 1.0, 1e-3],
                            [1.0, 1.0, 1e-2],
                            [1.0, 1.0, 1e-1],
                            [1.0, 1.0, 1.0],
                            [1.0, 1.0, 1e1],
                            [1.0, 1.0, 1e2],
                            [1.0, 1.0, 1e3],

                            [1.0, 1e-3, 1.0],
                            [1.0, 1e-2, 1.0],
                            [1.0, 1e-1, 1.0],
                            [1.0, 1e1, 1.0],
                            [1.0, 1e2, 1.0],
                            [1.0, 1e3, 1.0],
                           ])
        
        batch_size, N_pde, N_bc = 40, 40, 40
        init_learning_rate = 5e-4
        u_num_layers, u_hidden_units = 4, 32
        u_activation_fn = nn.Tanh()
        D_num_layers, D_hidden_units = 4, 32
        D_activation_fn = nn.Tanh()  

        Numrep = 10    

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
                    N_bio, u_bio_weight, D_bio_weight, bio_constraint_bool, val_dataset, test_dataset, train_dataset, x_max, x_min, t_min)

    if benchmark_job == 5: # learning rates
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

        os.makedirs(results_dir + f'Const_lr_noise{noise_level}/', exist_ok=True)
        torch.save(train_dataset, results_dir + f'Const_lr_noise{noise_level}/train_dataset.pt')
        torch.save(val_dataset, results_dir + f'Const_lr_noise{noise_level}/val_dataset.pt')
        torch.save(test_dataset, results_dir + f'Const_lr_noise{noise_level}/test_dataset.pt')
        torch.save(dataset, results_dir + f'Const_lr_noise{noise_level}/full_dataset.pt')
        Numrep = 10

        for init_learning_rate in init_learning_rates:
            save_dir = results_dir + f'Const_lr_noise{noise_level}/{init_learning_rate}/' 
            os.makedirs(save_dir, exist_ok=True)

            TrainBINN(save_dir, device, D, t_end, batch_size, Numrep, Num_epochs, u_num_layers, u_hidden_units, u_activation_fn, \
              D_num_layers, D_hidden_units, D_activation_fn, init_learning_rate, data_weight, bc_weight, pde_weight, \
                N_pde, N_bc, store_loss_step, grid_points_tensor, U_grid, U_grid_tensor, t_train, x_train, u_true, X, T, Nt, u_min, u_max, \
                    N_bio, u_bio_weight, D_bio_weight, bio_constraint_bool, val_dataset, test_dataset, train_dataset, x_max, x_min, t_min) 

    if benchmark_job == 6: # additive noises u (4, 32)
        # noise_levels = [
        #                 #0.0,
        #                 #0.005, 
        #                 #0.01, 0.05, 0.1, 
        #                 #0.2, 
        #                 0.3, 
        #                 #0.4, 0.5
        #                 ]

        # List of target variances
        variances = [0, 1e-3, 5e-3, 1e-2, 5e-2]

        Numrep = 10
        bio_constraint_bool = False # no biology constraint for u, D, and G
        N_bio = 1
        u_bio_weight, D_bio_weight, G_bio_weight = 1.0, 1.0, 0.0

        batch_size, N_pde, N_bc = 40, 40, 40
        data_weight, pde_weight, bc_weight= 1.0, 1e-1, 1e-3
        init_learning_rate = 5e-4
        u_num_layers, u_hidden_units = 4, 32
        u_activation_fn = nn.Tanh()
        D_num_layers, D_hidden_units = 4, 32
        D_activation_fn = nn.Tanh()    

        #for noise_level in noise_levels:
        for var in variances:

            noise_level = np.sqrt(var)

            #seed = 2 # for reproducibility
            seed = 8
            np.random.seed(seed)
            save_dir = results_dir + f'additive_noise_seed{seed}_uStruct{u_num_layers}_{u_hidden_units}/var{var}/' 
            os.makedirs(save_dir, exist_ok=True)

            noise_mul = noise_level * np.random.randn(*shape)
            #u_train_noise = u_train_nonoise * (1 + noise_mul)
            u_train_noise = u_train_nonoise + noise_mul
            output_data = u_train_noise.reshape(-1)[:, None]
            data = pd.DataFrame(np.hstack([input_data, output_data]), columns=['x', 't', 'u'])
            x_data = torch.tensor(data[['x', 't']].values, dtype=torch.float32).to(device)
            u_data = torch.tensor(data[['u']].values, dtype=torch.float32).to(device)
            u_min, u_max = u_data.min(), u_data.max()
            u_norm = u_max - u_min
            dataset = TensorDataset(x_data, u_data)
            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
            # save training data, validation data, and testing data
            torch.save(train_dataset, save_dir + 'train_dataset.pt')
            torch.save(val_dataset, save_dir + 'val_dataset.pt')
            torch.save(test_dataset, save_dir + 'test_dataset.pt')
            torch.save(dataset, save_dir + 'full_dataset.pt')

            TrainBINN(save_dir, device, D, t_end, batch_size, Numrep, Num_epochs, u_num_layers, u_hidden_units, u_activation_fn, \
                  D_num_layers, D_hidden_units, D_activation_fn, init_learning_rate, data_weight, bc_weight, pde_weight, \
                    N_pde, N_bc, store_loss_step, grid_points_tensor, U_grid, U_grid_tensor, t_train, x_train, u_true, X, T, Nt, u_min, u_max, \
                        N_bio, u_bio_weight, D_bio_weight, bio_constraint_bool, val_dataset, test_dataset, train_dataset, x_max, x_min, t_min)

    if benchmark_job == 7: # additive noises u (6, 128)
        # noise_levels = [
        #                 #0.0, 
        #                 #0.005, 
        #                 #0.01, 0.05, 0.1, 
        #                 #0.2, 
        #                 0.3, 
        #                 #0.4, 0.5
        #                 ]

        # List of target variances
        variances = [0, 1e-3, 5e-3, 1e-2, 5e-2]

        Numrep = 10
        bio_constraint_bool = False # no biology constraint for u, D, and G
        N_bio = 1
        u_bio_weight, D_bio_weight, G_bio_weight = 1.0, 1.0, 0.0

        batch_size, N_pde, N_bc = 40, 40, 40
        data_weight, pde_weight, bc_weight= 1.0, 1e-1, 1e-3
        init_learning_rate = 5e-4
        u_num_layers, u_hidden_units = 6, 128
        u_activation_fn = nn.Tanh()
        D_num_layers, D_hidden_units = 4, 32
        D_activation_fn = nn.Tanh()    

        #for noise_level in noise_levels:
        for var in variances:

            noise_level = np.sqrt(var)

            #seed = 2 # for reproducibility
            seed = 8
            np.random.seed(seed)
            save_dir = results_dir + f'additive_noise_seed{seed}_uStruct{u_num_layers}_{u_hidden_units}/var{var}/' 
            os.makedirs(save_dir, exist_ok=True)

            noise_mul = noise_level * np.random.randn(*shape)
            u_train_noise = u_train_nonoise + noise_mul
            output_data = u_train_noise.reshape(-1)[:, None]
            data = pd.DataFrame(np.hstack([input_data, output_data]), columns=['x', 't', 'u'])
            x_data = torch.tensor(data[['x', 't']].values, dtype=torch.float32).to(device)
            u_data = torch.tensor(data[['u']].values, dtype=torch.float32).to(device)
            u_min, u_max = u_data.min(), u_data.max()
            u_norm = u_max - u_min
            dataset = TensorDataset(x_data, u_data)
            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
            # save training data, validation data, and testing data
            torch.save(train_dataset, save_dir + 'train_dataset.pt')
            torch.save(val_dataset, save_dir + 'val_dataset.pt')
            torch.save(test_dataset, save_dir + 'test_dataset.pt')
            torch.save(dataset, save_dir + 'full_dataset.pt')

            TrainBINN(save_dir, device, D, t_end, batch_size, Numrep, Num_epochs, u_num_layers, u_hidden_units, u_activation_fn, \
                  D_num_layers, D_hidden_units, D_activation_fn, init_learning_rate, data_weight, bc_weight, pde_weight, \
                    N_pde, N_bc, store_loss_step, grid_points_tensor, U_grid, U_grid_tensor, t_train, x_train, u_true, X, T, Nt, u_min, u_max, \
                        N_bio, u_bio_weight, D_bio_weight, bio_constraint_bool, val_dataset, test_dataset, train_dataset, x_max, x_min, t_min)