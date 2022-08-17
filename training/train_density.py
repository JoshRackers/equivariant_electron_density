import sys
import os
import math
import numpy as np
import torch
import torch_geometric
from torch_cluster import radius_graph
from torch_scatter import scatter
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_iso_permuted_dataset
from e3nn.nn.models.gate_points_2101 import Network
from e3nn import o3
from utils import get_scalar_density_comparisons
import wandb
import random
from datetime import date
import argparse
import os


def lossPerChannel(y_ml, y_target,
    Rs = [(12, 0), (5, 1), (4, 2), (2, 3), (1, 4)]):
 
    err = y_ml - y_target
    pct_dev = torch.div(err.abs(),y_target)
    loss_perChannel_list = np.zeros(len(Rs))
    normalization = err.sum()/err.mean()

    counter = 0
    for mul, l in Rs:  
        if l==0:
            temp_loss = err[:,:mul].pow(2).sum().abs()/normalization
        else:
            temp_loss = err[:,counter:counter+mul*(2*l+1)].pow(2).sum().abs()/normalization
 
 
        loss_perChannel_list[l]+=temp_loss.detach().cpu().numpy()
        pct_deviation_list[l]+=temp_pct_deviation.detach().cpu().numpy()
       
        counter += mul*(2*l+1)
 
    return loss_perChannel_list

def main():
    parser = argparse.ArgumentParser(description='train electron density')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--testset', type=str)
    parser.add_argument('--split', type=int)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--qm', type=str, default="pbe0")
    parser.add_argument('ldep',type=bool, default=False)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print ("What device am I using?", device)

    torch.set_default_dtype(torch.float32)

    if args.qm == 'ccsd':
        hhh = os.path.dirname(os.path.realpath(__file__)) + "/../data/ccsd_h_s_only_def2-universal-jfit-decontract_density.out"
        ooo = os.path.dirname(os.path.realpath(__file__)) + "/../data/ccsd_o_s_only_def2-universal-jfit-decontract_density.out"
    else:
        hhh = os.path.dirname(os.path.realpath(__file__)) + "/../data/h_s_only_def2-universal-jfit-decontract_density.out"
        ooo = os.path.dirname(os.path.realpath(__file__)) + "/../data/o_s_only_def2-universal-jfit-decontract_density.out"

    test_dataset = args.testset
    num_epochs = args.epochs
    ldep_bool = args.ldep

    # def2 basis set max irreps
    # WARNING. this is currently hard-coded for def2_universal
    Rs = [(12, 0), (5, 1), (4, 2), (2, 3), (1, 4)]

    test_dataset = get_iso_permuted_dataset(args.testset,o_iso=ooo,h_iso=hhh)

    split = args.split
    data_file = args.dataset
    lr = 1e-2
    density_spacing = 0.1
    save_interval = 5
    model_kwargs = {
        "irreps_in": "2x 0e", #irreps_in 
        "irreps_hidden": [(mul, (l, p)) for l, mul in enumerate([125,40,25,15]) for p in [-1, 1]], #irreps_hidden
        "irreps_out": "12x0e + 5x1o + 4x2e + 2x3o + 1x4e", #irreps_out
        "irreps_node_attr": None, #irreps_node_attr
        "irreps_edge_attr": o3.Irreps.spherical_harmonics(3), #irreps_edge_attr
        "layers": 3,
        "max_radius": 3.5,
        "number_of_basis": 10,
        "radial_layers": 1,
        "radial_neurons": 128,
        "num_neighbors": 12.2298,
        "num_nodes": 24,
        "reduce_output": False,
    }

    dataset = get_iso_permuted_dataset(data_file,o_iso=ooo,h_iso=hhh)
    random.shuffle(dataset)
    if split > len(dataset):
        raise ValueError('Split is too large for the dataset.')
    
    b = 1
    train_loader = torch_geometric.data.DataLoader(dataset[:split], batch_size=b, shuffle=True)

    test_loader = torch_geometric.data.DataLoader(test_dataset, batch_size=b, shuffle=True)

    model = Network(**model_kwargs)

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    optim.zero_grad()

    model.to(device)

    model_kwargs["train_dataset"] = data_file
    model_kwargs["train_dataset_size"] = split
    model_kwargs["lr"] = lr
    model_kwargs["density_spacing"] = density_spacing
    wandb.init(config=model_kwargs, reinit=True)
    wandb.run.name = 'DATASET_' + args.dataset + '_SPLIT_' + str(args.split) + '_' + date.today().strftime("%b-%d-%Y")
    wandb.watch(model)

    for epoch in range(num_epochs):
        loss_cum = 0.0
        loss_perchannel_cum = np.zeros(len(Rs))
        mae_cum = 0.0
        mue_cum = 0.0
        for step, data in enumerate(train_loader):
            mask = torch.where(data.y == 0, torch.zeros_like(data.y), torch.ones_like(data.y)).detach()
            y_ml = model(data.to(device))*mask.to(device)
            err = (y_ml - data.y.to(device))
            
            for mul, l in Rs:
                if l == 0:
                    num_ele = sum(sum(y_ml[:,:mul])).detach()
            
            mue_cum += num_ele
            mae_cum += abs(num_ele)
            
            #compute loss per channel
            if ldep_bool:
                loss_perchannel_cum += lossPerChannel(y_ml,data.y.to(device), Rs)

            loss_cum += err.pow(2).mean().detach().abs()
            err.pow(2).mean().backward()
            optim.step()
            optim.zero_grad()
        
        # now the test loop
        with torch.no_grad():
            metrics = []
            for testset in [test_loader]:
                test_loss_cum = 0.0
                test_mae_cum = 0.0
                test_mue_cum = 0.0
                bigIs_cum = 0.0
                eps_cum = 0.0
                ep_per_l_cum = np.zeros(len(Rs))

                ele_diff_cum = 0.0
                for step, data in enumerate(testset):
                    mask = torch.where(data.y == 0, torch.zeros_like(data.y), torch.ones_like(data.y)).detach()
                    y_ml = model(data.to(device))*mask.to(device)
                    err = (y_ml - data.y.to(device))

                    for mul, l in Rs:
                        if l == 0:
                            num_ele = torch.mean(y_ml[:,:mul]).detach()
                            # num_ele = sum(sum(y_ml[:,:mul])).detach()

                    test_mue_cum += num_ele
                    test_mae_cum += abs(num_ele)
                    test_loss_cum += err.pow(2).mean().detach().abs()

                    if epoch % save_interval == 0:
                        torch.save(model.state_dict(), os.path.join(wandb.run.dir, "model_weights_epoch_"+str(epoch)+".pt"))
                        wandb.save("model_weights_epoch_"+str(epoch)+".pt")

                    if ldep_bool: 
                        num_ele_target, num_ele_ml, bigI, ep, ep_per_l= get_scalar_density_comparisons(data, y_ml, Rs, spacing=density_spacing, buffer=3.0, ldep=ldep_bool)
                        ep_per_l_cum += ep_per_l
                    else:
                        num_ele_target, num_ele_ml, bigI, ep = get_scalar_density_comparisons(data, y_ml, Rs, spacing=density_spacing, buffer=3.0, ldep=ldeb_bool)

                    n_ele = np.sum(data.z.cpu().detach().numpy())
                    ele_diff_cum += np.abs(n_ele-num_ele_target)
                    bigIs_cum += bigI
                    eps_cum += ep

                metrics.append([test_loss_cum, test_mae_cum, test_mue_cum, ele_diff_cum, bigIs_cum, eps_cum, ep_per_l_cum])

        # eps per l and loss per l hard coded for def2 below
        wandb.log({
            "Epoch": epoch,
            "Train_Loss": float(loss_cum)/len(train_loader),


            "Train_Loss l=0": float(loss_cum_per_l[0])/len(train_loader),
            "Train_Loss l=1": float(loss_cum_per_l[1])/len(train_loader),
            "Train_Loss l=2": float(loss_cum_per_l[2])/len(train_loader),
            "Train_Loss l=3": float(loss_cum_per_l[3])/len(train_loader),
            "Train_Loss l=4": float(loss_cum_per_l[4])/len(train_loader),


            "Train_MAE": mae_cum/len(train_loader),
            "Train_MUE": mue_cum/len(train_loader),



            "Test_Loss": float(metrics[0][0].item())/len(test_loader),
            "Test_MAE": metrics[0][1].item()/len(test_loader),
            "Test_MUE": metrics[0][2].item()/len(test_loader),
            "Test_Electron_Difference": metrics[0][3].item()/len(test_loader),
            "Test_big_I": metrics[0][4].item()/len(test_loader),
            "Test_Epsilon": metrics[0][5].item()/len(test_loader),  
            "Test_Epsilon l=0": metrics[0][-1][0].item()/len(test_loader),  
            "Test_Epsilon l=1": metrics[0][-1][1].item()/len(test_loader),
            "Test_Epsilon l=2": metrics[0][-1][2].item()/len(test_loader),
            "Test_Epsilon l=3": metrics[0][-1][3].item()/len(test_loader),
            "Test_Epsilon l=4": metrics[0][-1][4].item()/len(test_loader),
        })

        if epoch % 1 == 0:
            print(str(epoch) + " " + f"{float(loss_cum)/len(train_loader):.10f}")


            print("Train_Loss l=0", float(loss_cum_per_l[0])/len(train_loader))
            print("Train_Loss l=1",float(loss_cum_per_l[1])/len(train_loader))
            print("Train_Loss l=2", float(loss_cum_per_l[2])/len(train_loader))
            print("Train_Loss l=3",float(loss_cum_per_l[3])/len(train_loader))
            print("Train_Loss l=4", float(loss_cum_per_l[4])/len(train_loader))

            print("    MAE",mae_cum/(len(train_loader)*b))
            print("    MUE",mue_cum/(len(train_loader)*b))
            print("    Test_Loss", float(metrics[0][0].item())/len(test_loader))
            print("    Test_MAE",metrics[0][1].item()/len(test_loader))
            print("    Test_MUE",metrics[0][2].item()/len(test_loader))
            print("    Test_Electron_Difference",metrics[0][3].item()/len(test_loader))
            print("    Test_big_I",metrics[0][4].item()/len(test_loader))
            print("    Test_Epsilon",metrics[0][5].item()/len(test_loader))
            print("    Test_Epsilon l=0",metrics[0][-1][0].item()/len(test_loader)
            print("    Test_Epsilon l=1",metrics[0][-1][1].item()/len(test_loader)
            print("    Test_Epsilon l=2",metrics[0][-1][2].item()/len(test_loader)   
            print("    Test_Epsilon l=3",metrics[0][-1][3].item()/len(test_loader)
            print("    Test_Epsilon l=4",metrics[0][-1][4].item()/len(test_loader)


    wandb.finish()

if __name__ == '__main__':
    main()