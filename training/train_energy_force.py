import math
import numpy as np
import torch
import torch_geometric
from torch_cluster import radius_graph
from torch_scatter import scatter
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_iso_dataset
from e3nn.nn.models.gate_points_2101 import Network
from e3nn import o3
from utils import get_scalar_density_comparisons
import wandb
import random
from datetime import date
import argparse

def main():
    parser = argparse.ArgumentParser(description='train energy and force')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--testset', type=str)
    parser.add_argument('--split', type=int)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--gpu', type=str)
    args = parser.parse_args()

    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    print ("What device am I using?", device)
    
    torch.set_default_dtype(torch.float32)

    hhh = "../data/h_s_only_def2-universal-jfit-decontract_density.out"
    ooo = "../data/o_s_only_def2-universal-jfit-decontract_density.out"

    num_epochs = args.epochs

    # def2 basis set max irreps
    Rs = [(12, 0), (5, 1), (4, 2), (2, 3), (1, 4)]

    test_dataset = get_iso_dataset(args.testset,o_iso=ooo,h_iso=hhh)

    split = args.split
    data_file = args.dataset
    lr = 1e-2
    model_kwargs = {
        "irreps_in": "2x 0e", #irreps_in 
        "irreps_hidden": [(mul, l, p) for l, mul in enumerate([125,40,25,15]) for p in [-1, 1]], #irreps_hidden
        "irreps_out": "1x0e", #irreps_out
        "irreps_node_attr": None, #irreps_node_attr
        "irreps_edge_attr": o3.Irreps.spherical_harmonics(3), #irreps_edge_attr
        "layers": 3,
        "max_radius": 3.5,
        "number_of_basis": 10,
        "radial_layers": 1,
        "radial_neurons": 128,
        "num_neighbors": 12.2298,
        "num_nodes": 24,
        "reduce_output": True, 
    }

    dataset = get_iso_dataset(data_file,o_iso=ooo,h_iso=hhh)
    random.shuffle(dataset)
    
    if split > len(dataset):
        # continue # Only run when the split is contained in the dataset
        raise ValueError('Split it too large for the dataset.')
    
    b = 1
    train_loader = torch_geometric.data.DataLoader(dataset[:split], batch_size=b, shuffle=True)
    test_loader = torch_geometric.data.DataLoader(test_dataset, batch_size=b, shuffle=True)

    model = Network(**model_kwargs)

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    optim.zero_grad()

    model.to(device)
    
    energy_coefficient = 1.0
    force_coefficient = 1.0
    
    model_kwargs["energy_coefficient"] = energy_coefficient
    model_kwargs["force_coefficient"] = force_coefficient
    model_kwargs["train_dataset"] = data_file
    model_kwargs["train_dataset_size"] = split
    model_kwargs["lr"] = lr
    wandb.init(config=model_kwargs, reinit=True)
    wandb.run.name = 'DATASET_' + args.dataset + '_SPLIT_' + str(args.split) + '_' + date.today().strftime("%b-%d-%Y")
    wandb.watch(model)

    for epoch in range(num_epochs):
        loss_cum = 0.0
        e_mae = 0.0
        e_mue = 0.0
        f_mae = 0.0
        for step, data in enumerate(train_loader):
            data.pos.requires_grad = True
            y_ml = model(data.to(device))

            # get ml force
            forces = torch.autograd.grad(y_ml, data.pos.to(device), create_graph=True, retain_graph=True)[0]
            
            # subtract energy of water monomers (PBE0)
            monomer_energy = -76.379999960410643
            energy_err = y_ml - (data.energy.to(device) - monomer_energy*data.pos.shape[0]/3 )
            forces_err = forces - data.forces.to(device)

            e_mue += energy_err.detach()
            e_mae += energy_err.detach().abs()
            f_mae += forces_err.detach().flatten().abs().mean()

            energy_loss = energy_err.pow(2).mean()
            force_loss = forces_err.pow(2).mean()
            err = (energy_coefficient * energy_loss) + (force_coefficient * force_loss)
            err.backward()
            loss_cum += err.detach()

            optim.step()
            optim.zero_grad()
        
        # now the test loop
        for testset in [test_loader]:
            test_loss_cum = 0.0
            test_e_mae = 0.0
            test_e_mue = 0.0
            test_f_mae = 0.0
            for step, data in enumerate(testset):
                data.pos.requires_grad = True
                y_ml = model(data.to(device))

                # get ml force
                forces = torch.autograd.grad(y_ml, data.pos.to(device), create_graph=True, retain_graph=True)[0]

                # subtract energy of water monomers
                monomer_energy = -76.379999960410643
                energy_err = y_ml - (data.energy.to(device) - monomer_energy*data.pos.shape[0]/3 )
                forces_err = forces - data.forces.to(device)

                test_e_mue += energy_err.detach()
                test_e_mae += energy_err.detach().abs()
                test_f_mae += forces_err.detach().flatten().abs().mean()

                energy_loss = energy_err.pow(2).mean()
                force_loss = forces_err.pow(2).mean()
                err = (energy_coefficient * energy_loss) + (force_coefficient * force_loss)
                test_loss_cum += err.detach()
                
        wandb.log({
            "Epoch": epoch,
            "Train_Loss": float(loss_cum)/len(train_loader),
            "Train_Energy_MAE": float(e_mae)/len(train_loader),
            "Train_Energy_MUE": float(e_mue)/len(train_loader),
            "Train_Forces_MAE": float(f_mae)/len(train_loader),

            "Test_Loss": float(test_loss_cum)/len(test_loader),
            "Test_Energy_MAE": float(test_e_mae)/len(test_loader),
            "Test_Energy_MUE": float(test_e_mue)/len(test_loader),
            "Test_Forces_MAE": float(test_f_mae)/len(test_loader)
  
        })

        if epoch % 1 == 0:
            print(str(epoch) + " " + f"{float(loss_cum)/len(train_loader):.10f}")

    wandb.finish()

if __name__ == '__main__':
    main()

