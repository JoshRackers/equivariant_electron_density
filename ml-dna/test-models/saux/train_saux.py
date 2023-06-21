import math
import numpy as np
import torch
import torch_geometric
from torch_cluster import radius_graph
from torch_scatter import scatter
from utils import get_iso_permuted_dataset
from utils import get_scalar_density_comparisons
from utils import get_iso_permuted_dataset_lpop_scale
from e3nn.nn.models.gate_points_2101 import Network
from e3nn import o3
import wandb
import random
import psi4
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print (device)

torch.set_default_dtype(torch.float32)
#torch.set_default_dtype(torch.float64)

# first, get dataset
# eights = "../density_data/w8_def2_constrainQ_padded.pkl"
# hhh = "../density_data/h_s_only_def2-universal-jfit-decontract_density.out"
# ooo = "../density_data/o_s_only_def2-universal-jfit-decontract_density.out"
# w8_dataset = get_iso_permuted_dataset(eights,ooo,hhh)

# tens = "../density_data/w10_def2_constrainQ_padded.pkl"
# w10_dataset = get_iso_permuted_dataset(tens,ooo,hhh)

# thirties = "../density_data/w30_def2_constrainQ_padded.pkl"
# w30_dataset = get_iso_permuted_dataset(thirties,ooo,hhh)

hhh = "h_s_only_dzhf_def2_density.out"
ooo = "o_s_only_dzhf_def2_density.out"

#Rs = [(16, 0), (14, 1), (13, 2), (7, 3), (7, 4)]
Rs = [(12, 0), (5, 1), (4, 2), (2, 3), (1, 4)]

train_data_file = "./w3_dzhf_def2_train_energyforce.pkl"
test_data_file = "./w3_dzhf_def2_test_energyforce.pkl"
#w30_datafile = "../h20-ml-tut/data_water/w30_def2_constrainQ_padded_big.pkl"

train_dataset = get_iso_permuted_dataset_lpop_scale(train_data_file, Rs, o_iso=ooo,h_iso=hhh)
test_dataset = get_iso_permuted_dataset_lpop_scale(test_data_file, Rs, o_iso=ooo,h_iso=hhh)
random.shuffle(train_dataset)
random.shuffle(test_dataset)

train_split = [1]
#split = 40
b = 1

# normalize coefficients in training data
# Rs = [(16, 0), (14, 1), (13, 2), (7, 3), (7, 4)]

#train_all = torch_geometric.data.DataLoader(dataset[:split], batch_size=b, shuffle=True)
train_loader = torch_geometric.data.DataLoader(train_dataset, batch_size=b, shuffle=True)
test_loader = torch_geometric.data.DataLoader(test_dataset, batch_size=b, shuffle=True)

num_epochs = 501

# second, check if num_ele is correct
# def2 basis set max irreps

# this assumes that l=0 is first!!!
# for mul, l in Rs:
#     if l == 0:
#         w8_ele =  sum(sum(w8_dataset[0]['y'][:,:mul]))
#         w10_ele = sum(sum(w10_dataset[0]['y'][:,:mul]))
#         w30_ele = sum(sum(w30_dataset[0]['y'][:,:mul]))


for train_size in train_split:

        model_kwargs = {
            "irreps_in": "2x 0e", #irreps_in 
            "irreps_hidden": [(mul, l, p) for l, mul in enumerate([200,67,40,29]) for p in [-1, 1]], #irreps_hidden
            #"irreps_hidden": "100x0e + 100x0o",
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

        model = Network(**model_kwargs)

        optim = torch.optim.Adam(model.parameters(), lr=1e-2)
        optim.zero_grad()

        model.to(device)

        model_kwargs["train_dataset"] = train_data_file
        model_kwargs["training set size"] = train_size
        run = wandb.init(name='w3_dzhf_def2_saux_e3nn2psi4',project='h20-debug', entity='density-lab', config=model_kwargs, reinit=True)
        wandb.watch(model)

        print("train set length:", len(train_loader))
        print("test set length:", len(test_loader))

        ele_diff_cum_save = 0.0
        bigIs_cum_save = 0.0
        eps_cum_save = 0.0

        for epoch in range(num_epochs):

            tic = time.perf_counter()            

            loss_cum = 0.0
            mae_cum = 0.0
            mue_cum = 0.0
            for step, data in enumerate(train_loader):
                mask = torch.where(data.y == 0, torch.zeros_like(data.y), torch.ones_like(data.y)).detach()
                y_ml = model(data.to(device))*mask.to(device)

                # calculate Saux for current sample
                basisname = 'dzhf-dna'
                auxbasis = 'def2-universal-jfit-dna'

                # define molecule
                coords = data.pos.tolist()
                atomic_nums = data.z.tolist()
                string_coords = []
                for item, anum in zip(coords, atomic_nums):
                    string = ' '.join([str(elem) for elem in item])
                    if anum[0] == 1.0:
                        line = ' H  ' + string
                    if anum[0] == 8.0:
                        line = ' O  ' + string
                    string_coords.append(line)
                molstr = """
                {}
                 symmetry c1
                 no_reorient
                 units angstrom
                 no_com
                 0 1
                """.format("\n".join(string_coords))
                #print(molstr)
                mol = psi4.geometry(molstr)

                psi4.core.set_global_option('df_basis_scf', auxbasis)

                orbital_basis = psi4.core.BasisSet.build(mol, key="ORBITAL", target=basisname, fitrole="ORBITAL", quiet=True)
                aux_basis = psi4.core.BasisSet.build(mol, "DF_BASIS_SCF", "", "JFIT", auxbasis, quiet=True)

                mints = psi4.core.MintsHelper(orbital_basis)
                Saux = np.array(mints.ao_overlap(aux_basis, aux_basis), dtype='f')
                Saux = torch.from_numpy(Saux)
                #Saux = torch.eye(Saux.size()[0],Saux.size()[1]) #debug identity matrix for Saux
                Saux = Saux.cuda()

                # err is size Ncoeff x 1

                e3nnorder_err = (y_ml - data.y.to(device))
                psi4order_err = torch.clone(e3nnorder_err)

                # convert e3nn_err to psi4 ordering so you can multiply by psi4's Saux

                e3nn_2_psi4 = [[0],[1,2,0],[2,3,1,4,0],[3,4,2,5,1,6,0],[4,5,3,6,2,7,1,8,0],[5,6,4,7,3,8,2,9,1,10,0],[6,7,5,8,4,9,3,10,2,11,1,12,0]]

                for atom_ind in range(torch.Tensor.size(psi4order_err)[0]):
                    coeff_counter = 0
                    for mul, l in Rs:
                        for j in range(mul):
                            k_counter = 0
                            for k in e3nn_2_psi4[l]:
                                psi4order_err[atom_ind][coeff_counter+k_counter]=e3nnorder_err[atom_ind][coeff_counter+k]
                                k_counter +=1
                            coeff_counter += 2*l+1
                            
                for mul, l in Rs:
                    if l == 0:
                        num_ele = sum(sum(y_ml[:,:mul])).detach()
                
                mue_cum += num_ele
                mae_cum += abs(num_ele)

                errtosaux = torch.zeros(Saux.size()[0]).cuda()

                # eliminate 0 mask components in err so you can multiply by Saux

                k = 0
                for i in range(np.shape(data.norm)[0]):
                    for j in range(np.shape(data.norm)[1]):
                        if(data.norm[i,j]==0.0):
                            pass
                        else:
                            errtosaux[k]=psi4order_err[i,j]
                            k+=1
    
                if (k!=Saux.size()[0]):
                    print("ERROR! Incompatible matrix sizes: k, Saux ", k, Saux.size()[0])
                    exit()

                # Saux is size Ncoeff x Ncoeff; err is Ncoeff x 1
                # New error is Saux*err, which is still Ncoeff x 1; use mean squared error as loss function

                loss = torch.mv(Saux, errtosaux)

                loss_cum += loss.pow(2).mean().detach().abs()
                loss.pow(2).mean().backward()
                
                optim.step()
                optim.zero_grad()

            
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

                        # calculate Saux for current sample
                        basisname = 'dzhf-dna'
                        auxbasis = 'def2-universal-jfit-dna'

                        # define molecule
                        coords = data.pos.tolist()
                        atomic_nums = data.z.tolist()
                        string_coords = []
                        for item, anum in zip(coords, atomic_nums):
                            string = ' '.join([str(elem) for elem in item])
                            if anum[0] == 1.0:
                                line = ' H  ' + string
                            if anum[0] == 8.0:
                                line = ' O  ' + string
                            string_coords.append(line)
                        molstr = """
                        {}
                         symmetry c1
                         no_reorient
                         units angstrom
                         no_com
                         0 1
                        """.format("\n".join(string_coords))
                        mol = psi4.geometry(molstr)

                        psi4.core.set_global_option('df_basis_scf', auxbasis)

                        orbital_basis = psi4.core.BasisSet.build(mol, key="ORBITAL", target=basisname, fitrole="ORBITAL", quiet=True)
                        aux_basis = psi4.core.BasisSet.build(mol, "DF_BASIS_SCF", "", "JFIT", auxbasis, quiet=True)

                        mints = psi4.core.MintsHelper(orbital_basis)
                        Saux = np.array(mints.ao_overlap(aux_basis, aux_basis), dtype='f')
                        Saux = torch.from_numpy(Saux)
                        #Saux = torch.eye(Saux.size()[0],Saux.size()[1]) #debug identity matrix for Saux
                        Saux = Saux.cuda()

                        # err is size Ncoeff x 1

                        e3nnorder_err = (y_ml - data.y.to(device))
                        psi4order_err = torch.clone(e3nnorder_err)

                        for atom_ind in range(torch.Tensor.size(psi4order_err)[0]):
                            coeff_counter = 0
                            for mul, l in Rs:
                                for j in range(mul):
                                    k_counter = 0
                                    for k in e3nn_2_psi4[l]:
                                        psi4order_err[atom_ind][coeff_counter+k_counter]=e3nnorder_err[atom_ind][coeff_counter+k]
                                        k_counter +=1
                                    coeff_counter += 2*l+1
                        
                        for mul, l in Rs:
                            if l == 0:
                                num_ele = sum(sum(y_ml[:,:mul])).detach()
                        
                        test_mue_cum += num_ele
                        test_mae_cum += abs(num_ele)

                        errtosaux = torch.zeros(Saux.size()[0]).cuda()

                        k = 0
                        for i in range(np.shape(data.norm)[0]):
                            for j in range(np.shape(data.norm)[1]):
                                if(data.norm[i,j]==0.0):
                                    pass
                                else:
                                    errtosaux[k]=psi4order_err[i,j]
                                    k+=1
            
                        if (k!=Saux.size()[0]):
                            print("ERROR! Incompatible matrix sizes: k, Saux ", k, Saux.size()[0])
                            exit()

                        # Saux is size Ncoeff x Ncoeff

                        loss = torch.mv(Saux, errtosaux)

                        test_loss_cum += loss.pow(2).mean().detach().abs()
                   
                        if (epoch != 0 and epoch%10==0):
                            num_ele_target, num_ele_ml, bigI, ep = get_scalar_density_comparisons(data, y_ml, Rs, spacing=0.2, buffer=4.0)
                            n_ele = np.sum(data.z.cpu().detach().numpy())
                            ele_diff_cum += np.abs(n_ele-num_ele_target)
                            bigIs_cum += bigI
                            eps_cum += ep

                            ele_diff_cum_save = ele_diff_cum
                            bigIs_cum_save = bigIs_cum
                            eps_cum_save = eps_cum

                            #savename = "./saved_model/def2_lpop/"+str(epoch)+".pkl"
                            #torch.save(model.state_dict(), savename)

                    metrics.append([test_loss_cum, test_mae_cum, test_mue_cum])

                if (epoch ==0 or epoch%10!=0):
                    ele_diff_cum = ele_diff_cum_save
                    bigIs_cum = bigIs_cum_save
                    eps_cum = eps_cum_save 


            toc = time.perf_counter()
            epoch_time = toc-tic

            wandb.log({
                "Epoch": epoch,
                "Epoch time": epoch_time,
                "Train_Loss": float(loss_cum)/len(train_loader),
                "Train_MAE": mae_cum/len(train_loader),
                "Train_MUE": mue_cum/len(train_loader),

                "test_Loss": float(metrics[0][0].item())/len(test_loader),
                "test_MAE": metrics[0][1].item()/len(test_loader),
                "test_MUE": metrics[0][2].item()/len(test_loader),

		        "test electron difference": ele_diff_cum/len(test_loader),
		        "test big I": bigIs_cum/len(test_loader),
		        "test epsilon": eps_cum/len(test_loader),

            })

            if epoch % 1 == 0:
                print(str(epoch) + " " + f"{float(loss_cum)/len(train_loader):.10f}")
                print("    MAE",mae_cum/(len(train_loader)*b))
                print("    MUE",mue_cum/(len(train_loader)*b))
        
        #torch.save(model.state_dict(), 'saved_model.pkl')
        run.finish()


