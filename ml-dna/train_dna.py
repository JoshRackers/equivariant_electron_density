import math
import numpy as np
import torch
import torch_geometric
from torch_cluster import radius_graph
from torch_scatter import scatter
from utils import get_iso_permuted_dataset
from utils import get_scalar_density_comparisons
from e3nn.nn.models.gate_points_2101 import Network
from e3nn import o3
import wandb
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print (device)

torch.set_default_dtype(torch.float32)
#torch.set_default_dtype(torch.float64)

# first, get dataset

hhh = "./data/h_s_only_augccpvdz_density.out"
ooo = "./data/o_s_only_augccpvdz_density.out"
ccc = "./data/c_s_only_augccpvdz_density.out"
nnn = "./data/n_s_only_augccpvdz_density.out"
ppp = "./data/p_s_only_augccpvdz_density.out"

train_datasets = ["1at-400.pkl",
    "2ta-400.pkl",
    "3aa-400.pkl",
    "4ca-400.pkl",
    "5gt-400.pkl",
    "6ct-400.pkl",
    "7ga-400.pkl",
    "8cg-400.pkl",
    "9gc-400.pkl",
    "10gg-400.pkl"]

print(train_datasets)

test_datafile = "2mer-test.pkl"
test_dataset = get_iso_permuted_dataset(test_datafile,h_iso=hhh,c_iso=ccc,n_iso=nnn,o_iso=ooo,p_iso=ppp)
random.shuffle(test_dataset)

b = 1
train_split = [100]

test_loader = torch_geometric.data.DataLoader(test_dataset, batch_size=b, shuffle=True)
num_epochs = 251

# second, check if num_ele is correct
# def2 basis set max irreps
Rs = [(14, 0), (5, 1), (5, 2), (2, 3), (1, 4)]

for train_size in train_split:

    model_kwargs = {
            "irreps_in": "5x 0e", #irreps_in 
            "irreps_hidden": [(mul, l, p) for l, mul in enumerate([200,67,40,29]) for p in [-1, 1]], #irreps_hidden
            #"irreps_hidden": "100x0e + 100x0o",
            "irreps_out": "14x0e + 5x1o + 5x2e + 2x3o + 1x4e", #irreps_out
            "irreps_node_attr": None, #irreps_node_attr
            "irreps_edge_attr": o3.Irreps.spherical_harmonics(3), #irreps_edge_attr
            "layers": 5,
            "max_radius": 3.5,
            "num_neighbors": 12.666666,
            "number_of_basis": 10,
            "radial_layers": 1,
            "radial_neurons": 128,
            "num_nodes": 24,
            "reduce_output": False,
    }

    model = Network(**model_kwargs)

    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    optim.zero_grad()

    model.to(device)

    # load previous model
    #checkpoint = torch.load('trainall-100.pkl')
    #model.load_state_dict(checkpoint)

    run = wandb.init(config=model_kwargs, reinit=True)
    wandb.watch(model)

    ele_diff_cum_save = 0.0
    bigIs_cum_save = 0.0
    eps_cum_save = 0.0

    for epoch in range(num_epochs):
        loss_cum = 0.0
        mae_cum = 0.0
        mue_cum = 0.0

        train_num_ele = []
        test_num_ele = []

        for data_file in train_datasets:

            print("Data file: ", data_file)

            train_dataset = get_iso_permuted_dataset(data_file,h_iso=hhh,c_iso=ccc,n_iso=nnn,o_iso=ooo,p_iso=ppp)
            random.shuffle(train_dataset)
            train_loader = torch_geometric.data.DataLoader(train_dataset[:train_size], batch_size=b, shuffle=True)

            for step, data in enumerate(train_loader):
                mask = torch.where(data.y == 0, torch.zeros_like(data.y), torch.ones_like(data.y)).detach()
                y_ml = model(data.to(device))*mask.to(device)
                err = (y_ml - data.y.to(device))

                for mul, l in Rs:
                    if l == 0:
                        num_ele = sum(sum(y_ml[:,:mul])).detach()
                
                train_num_ele.append(num_ele.item())
                
                mue_cum += num_ele
                mae_cum += abs(num_ele)
 
                loss_cum += err.pow(2).mean().detach().abs()
                err.pow(2).mean().backward()
                optim.step()
                optim.zero_grad()
        
        print("Train num ele: ", len(train_num_ele))
        train_tot = len(train_num_ele)
        train_stdev = np.std(train_num_ele)
            
        # now the test loop
        # if epoch % 10 == 0:
        with torch.no_grad():
            metrics = []
            for testset in [test_loader]:
                test_loss_cum = 0.0
                test_mae_cum = 0.0
                test_mue_cum = 0.0
                bigIs_cum = 0.0
                eps_cum = 0.0
                ele_diff_cum = 0.0
                for step, data in enumerate(testset):
                    mask = torch.where(data.y == 0, torch.zeros_like(data.y), torch.ones_like(data.y)).detach()
                    y_ml = model(data.to(device))*mask.to(device)
                    err = (y_ml - data.y.to(device))

                    for mul, l in Rs:
                        if l == 0:
                            num_ele = sum(sum(y_ml[:,:mul])).detach()

                    test_num_ele.append(num_ele.item())

                    test_mue_cum += num_ele
                    test_mae_cum += abs(num_ele)
                    test_loss_cum += err.pow(2).mean().detach().abs()

                    if (epoch != 0 and epoch%10==0):
                        num_ele_target, num_ele_ml, bigI, ep = get_scalar_density_comparisons(data, y_ml, Rs, spacing=0.2, buffer=4.0)
                        n_ele = np.sum(data.z.cpu().detach().numpy())
                        ele_diff_cum += np.abs(n_ele-num_ele_target)
                        bigIs_cum += bigI
                        eps_cum += ep

                        ele_diff_cum_save = ele_diff_cum
                        bigIs_cum_save = bigIs_cum
                        eps_cum_save = eps_cum

                metrics.append([test_loss_cum, test_mae_cum, test_mue_cum])

            test_stdev = np.std(test_num_ele)

            if (epoch ==0 or epoch%10!=0):
                ele_diff_cum = ele_diff_cum_save
                bigIs_cum = bigIs_cum_save
                eps_cum = eps_cum_save
                
        wandb.log({
            "Epoch": epoch,
            "Train_Loss": float(loss_cum)/train_tot,
            "Train_MAE": mae_cum/train_tot,
            "Train_MUE": mue_cum/train_tot,
            "Train_STDEV": train_stdev,
            "Train_tot": train_tot,

            "test_Loss": float(metrics[0][0].item())/len(test_loader),
            "test_MAE": metrics[0][1].item()/len(test_loader),
            "test_MUE": metrics[0][2].item()/len(test_loader),
            "test_STDEV": test_stdev,

            "test electron difference": ele_diff_cum/len(test_loader),
            "test big I": bigIs_cum/len(test_loader),
            "test epsilon": eps_cum/len(test_loader),

        })

        savename = "trainerror-100-"+str(epoch)+".pkl"
        torch.save(model.state_dict(), savename)

    run.finish()


