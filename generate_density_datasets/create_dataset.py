# pylint: disable=invalid-name, no-member, arguments-differ, missing-docstring, line-too-long

import sys
import os
import pickle
import time
import numpy as np
import torch

#sys.path.append("../")
# get the utils.py module in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import flatten_list
from itertools import zip_longest
import periodictable as pt


def get_densities(filepath,dens_file,elements,num_atoms):
    """
    inputs:
    - density file
    - number of atoms
    
    returns:
    - shape [N, X] list of basis function coefficients, where X is the number of basis functions per atom
    """
    ## get density coefficients for each atom
    ## ordered in ascending l order
    ## also get Rs_out for each atom
    dens_file = filepath + "/" + dens_file

    basis_coeffs = []
    basis_exponents = []
    basis_norms = []
    num_basis_func = []
    Rs_outs = []
    for l in range(0,20):
        flag = 0
        atom_index = -1
        counter = 0
        multiplicity = 0
        with open (dens_file,"r") as density_file:
            for line in density_file:
                if (flag == 1):
                    split = line.split()
                    if (int(split[0])==l):
                        basis_coeffs[atom_index].append(float(split[1]))
                        basis_exponents[atom_index].append(float(split[2]))
                        basis_norms[atom_index].append(float(split[3]))
                        multiplicity += 1
                    counter += 1
                    if (counter == num_lines):
                        flag = 0
                        if (multiplicity != 0): 
                            Rs_outs[atom_index].append((multiplicity//(2*l+1),l))
                if ("functions" in line):
                    num_lines = int(line.split()[3])
                    num_basis_func.append(num_lines)
                    counter = 0
                    multiplicity = 0
                    flag = 1
                    atom_index += 1
                    if (l == 0):
                        basis_coeffs.append([])
                        basis_exponents.append([])
                        basis_norms.append([])
                        Rs_outs.append([])


    # break coefficients list up into l-based vectors
    newbasis_coeffs = []
    newbasis_exponents = []
    newbasis_norms = []
    atom_index = -1
    for atom in Rs_outs:
        atom_index += 1
        counter = 0
        newbasis_coeffs.append([])
        newbasis_exponents.append([])
        newbasis_norms.append([])
        for Rs in atom:
            number = Rs[0]
            l = Rs[1]
            for i in range(0,number):
                newbasis_coeffs[atom_index].append(basis_coeffs[atom_index][counter:counter+(2*l+1)])
                newbasis_exponents[atom_index].append(basis_exponents[atom_index][counter:counter+(2*l+1)])
                newbasis_norms[atom_index].append(basis_norms[atom_index][counter:counter+(2*l+1)])
                counter += 2*l+1
    
    Rs_out_list = []
    elementdict = {}
    for i, elem in enumerate(elements):
        if elem not in elementdict:
            elementdict[elem] = Rs_outs[i]
            Rs_out_list.append(Rs_outs[i])
    
    '''
    #psi4
    S: 0	
    P: 0, +1, -1	
    D: 0, +1, -1, +2, -2	
    F: 0, +1, -1, +2, -2, +3, -3	
    G: 0, +1, -1, +2, -2, +3, -3, +4, -4
    H: 0, +1, -1, +2, -2, +3, -3, +4, -4, +5, -5
    I: 0, +1, -1, +2, -2, +3, -3, +4, -4, +5, -5, +6, -6

    #e3nn (wikipedia)
    S: 0	
    P: -1, 0, +1	
    D: -2, -1, 0, +1, +2	
    F: -3, -2, -1, 0, +1, +2, +3	
    G: -4, -3, -2, -1, 0, +1, +2, +3, +4
    H: -5, -4, -3, -2, -1, 0, +1, +2, +3, +4, +5
    I: -6, -5, -4, -3, -2, -1, 0, +1, +2, +3, +4, +5, +6
    '''
    
    ##              s     p         d             f                 g                      h                           i
    psi4_2_e3nn = [[0],[2,0,1],[4,2,0,1,3],[6,4,2,0,1,3,5],[8,6,4,2,0,1,3,5,7],[10,8,6,4,2,0,1,3,5,7,9],[12,10,8,6,4,2,0,1,3,5,7,9,11]]
    
    '''
    test = [[0],[0, +1, -1],[0, +1, -1, +2, -2],[0, +1, -1, +2, -2, +3, -3],	
            [0, +1, -1, +2, -2, +3, -3, +4, -4]]
    for i, item in enumerate(test):
        l = (len(item)-1)//2
        print (l)
        test[i] = [item[i] for i in psi4_2_e3nn[l]]
    '''        


    #change convention from psi4 to e3nn
    for i, atom in enumerate(newbasis_coeffs):
        for j, item in enumerate(atom):
            l = (len(item)-1)//2
            if l > 6:
                raise ValueError('L is too high. Currently only supports L<7')
            newbasis_coeffs[i][j] = [item[k] for k in psi4_2_e3nn[l]]


    return newbasis_coeffs, newbasis_exponents, newbasis_norms, Rs_outs


def get_energy_force(filepath,out_file,num_atoms):
    file = filepath + "/" + out_file

    flag = False
    with open (file,"r") as f:
        for line in f:
            if "Total Energy =" in line:
                energy = float(line.split()[3])
            if "Total Gradient:" in line:
                flag = True
                counter = 0
                forces = []
            if flag == True:
                if 2 < counter < num_atoms+3:
                    forces.append([float(line.split()[1]), float(line.split()[2]), float(line.split()[3])])
                counter += 1

    energy = np.array(energy)
    forces = np.array(forces)

    return energy, forces



def get_coordinates(filepath,inputfile):
    """
    reads in coordinates and atomic number from psi4 input file

    returns:
    -shape [N, 3] numpy array of points
    -shape [N] numpy array of masses
    -shape [N] list of element symbols
    """
    # read in coords and atomic numbers
    inputfile = filepath + "/" + inputfile
    
    if not os.path.exists(inputfile):
        inputfile = inputfile + ".xyz"

    points = np.loadtxt(inputfile, skiprows=2, usecols=range(1,4))
    numatoms = len(points)
    elements = np.genfromtxt(inputfile, skip_header=2, usecols=0,dtype='str')
    atomic_numbers = [getattr(pt,i).number for i in elements]
    unique_elements = len(np.unique(atomic_numbers))
    onehot = np.zeros((numatoms,unique_elements))

    #get one hot vector
    weighted_onehot = onehot
    typedict = {}
    counter = -1
    for i, num in enumerate(atomic_numbers):
        if num not in typedict:
            # dictionary: key = atomic number
            # value = 0,1,2,3 (ascending types)
            counter += 1
            typedict[num] = counter
        weighted_onehot[i, typedict[num]] = num

    #print(weighted_onehot)
            
    return points, numatoms, atomic_numbers, elements, weighted_onehot


def get_dataset(filepath):
    dataset = []

    #coeff_by_type_list = []
    for filename in sorted(os.listdir(filepath)):
        if filename.endswith("density.out"):
            # read in stuff
            densityfile = filename
            split = filename.split("_")
            xyzfile = split[0] + "_" + split[1]
            print(xyzfile)
            doforces = True
            outfile = "output_" + xyzfile + ".dat" 
            if not os.path.exists(filepath + outfile):
                outfile = "output_" + xyzfile + ".xyz.dat"
                if not os.path.exists(filepath + outfile):
                    outfile = xyzfile + "_output.dat"
                    if not os.path.exists(filepath + outfile):
                        outfile = xyzfile + ".xyz_output.dat"
                        if not os.path.exists(filepath + outfile):
                            doforces = False
                            print("No energies or forces because ", outfile, "does not exist")

            # read in xyz file
            # get number of atoms
            # get onehot encoding
            points, num_atoms, atomic_numbers, elements, weighted_onehot = get_coordinates(filepath,xyzfile)
            N = num_atoms

            doPac = False
            if os.path.exists(filepath + xyzfile+"_charge"):
                pac = get_pac(filepath, xyzfile, N)
                doPac = True
                
            # construct one hot encoding
            onehot = weighted_onehot
            # replace all nonzero values with 1
            onehot[onehot > 0.001] = 1

            # read in density file
            coefficients, exponents, norms, Rs_out_list = get_densities(filepath,densityfile,elements,N)
            
            if doforces:
                energy, forces = get_energy_force(filepath,outfile,N)

            # compute Rs_out_max
            # this is necessary because O and H have different Rs_out
            # to deal with this, I set the global Rs_out to be the maximum
            # basically, for each L, 
            # i take whichever entry that has the higher multiplicity

            from itertools import zip_longest

            #print(Rs_out_list)

            a = list(zip_longest(*Rs_out_list))
            # remove Nones
            b = [[v if v is not None else (0,0) for v in nested] for nested in a]

            Rs_out_max = []
            for rss in b:
                Rs_out_max.append(max(rss))

            # HACKY: MANUAL OVERRIDE OF RS_OUT_MAX
            # set manual Rs_out_max (comment out if desired)
            #Rs_out_max=[(14, 0), (5, 1), (5, 2), (2, 3), (1, 4)]
            #print("Using manual Rs_out_max:", Rs_out_max)

            ## now construct coefficient, exponent and norm arrays
            ## from Rs_out_max
            ## pad with zeros

            coeff_dim = 0
            for mul, l in Rs_out_max:
                coeff_dim += mul*((2*l) + 1)
            
            rect_coeffs = torch.zeros(len(Rs_out_list),coeff_dim)
            rect_expos = torch.zeros(len(Rs_out_list),coeff_dim)
            rect_norms = torch.zeros(len(Rs_out_list),coeff_dim)

            for i, (atom, coeff_list, expo_list, norm_list) in enumerate(zip(Rs_out_list, coefficients, exponents, norms)):
                counter = 0
                list_counter = 0
                for (mul, l), (max_mul, max_l) in zip(atom, Rs_out_max):
                    n = mul*((2*l) + 1)
                    rect_coeffs[i,counter:counter+n] = torch.Tensor(list(flatten_list(coeff_list[list_counter:list_counter+mul])))
                    rect_expos[i,counter:counter+n] = torch.Tensor(list(flatten_list(expo_list[list_counter:list_counter+mul])))
                    rect_norms[i,counter:counter+n] = torch.Tensor(list(flatten_list(norm_list[list_counter:list_counter+mul])))
                    list_counter += mul
                    max_n = max_mul*((2*max_l)+1)
                    counter += max_n

            # dataset includes partial atomic charges
            if doPac:

                print("Dataset includes PAC")

                # dataset includes energies and forces
                if doforces:
                    cluster_dict = {
                        'type' : torch.Tensor(atomic_numbers),
                        'pos' : torch.Tensor(points),
                        'onehot' : torch.Tensor(onehot),
                        'coefficients' : rect_coeffs,
                        'exponents' : rect_expos,
                        'norms' : rect_norms,
                        'rs_max' : Rs_out_max,
                        'energy' : torch.Tensor(energy),
                        'forces' : torch.Tensor(forces),
                        'pa_charges' : torch.Tensor(pac)
                    }
                else:
                    cluster_dict = {
                        'type' : torch.Tensor(atomic_numbers),
                        'pos' : torch.Tensor(points),
                        'onehot' : torch.Tensor(onehot),
                        'coefficients' : rect_coeffs,
                        'exponents' : rect_expos,
                        'norms' : rect_norms,
                        'rs_max' : Rs_out_max,
                        'pa_charges' : torch.Tensor(pac)
                    }

            # dataset does NOT include partial atomic charges
            elif doforces:
                cluster_dict = {
                    'type' : torch.Tensor(atomic_numbers),
                    'pos' : torch.Tensor(points),
                    'onehot' : torch.Tensor(onehot),
                    'coefficients' : rect_coeffs,
                    'exponents' : rect_expos,
                    'norms' : rect_norms,
                    'rs_max' : Rs_out_max,
                    'energy' : torch.Tensor(energy),
                    'forces' : torch.Tensor(forces),
                }
            else:
                cluster_dict = {
                    'type' : torch.Tensor(atomic_numbers),
                    'pos' : torch.Tensor(points),
                    'onehot' : torch.Tensor(onehot),
                    'coefficients' : rect_coeffs,
                    'exponents' : rect_expos,
                    'norms' : rect_norms,
                    'rs_max' : Rs_out_max,
                }
            dataset.append(cluster_dict)

    # reset onehot based on whole dataset
    # need list of unique atomic numbers, in ascending order
    # then iterate through atoms
    # onehot[i, index_of_matching_atom_in_unique_elements] = 1
    # look up how to get index- np.where
   
    all_anum = []
    for item in dataset:
        anum = item["type"]
        all_anum.extend(anum)
    
    unique_elements = np.unique(all_anum)
    num_unique_elements = len(unique_elements)

    for item in dataset:
        numatoms = item['pos'].shape[0]
        onehot = np.zeros((numatoms,num_unique_elements))
        for i, num in enumerate(item["type"]):
            n = num.item()
            index = np.where(unique_elements == n)
            onehot[i, index] = 1
        item['onehot'] = torch.Tensor(onehot)


    # now get Rs_out_max for whole dataset
    all_rs = []
    for item in dataset:
        rs = item["rs_max"]
        all_rs.append(rs)
    
    #print(all_rs)

    a = list(zip_longest(*all_rs))
    # remove Nones
    b = [[v if v is not None else (0,0) for v in nested] for nested in a]

    Rs_out_max = []
    for rss in b:
        Rs_out_max.append(max(rss))
    
    print("irreps_out",Rs_out_max)

    return dataset

# get partial atomic charges (from classical force field, for example)
def get_pac(filepath, inputfile, N):
    # partial atomic charges *_charge obtained from Amber parameter file
    # assumes charges are ordered the same as the atoms in the *.xyz file

    inputfile = filepath + "/" + inputfile
    inputfile = inputfile+"_charge"

    charge = []

    with open(inputfile) as f:
        charge_data=f.readlines()

    for lines in charge_data:
        line = lines.split()
        if len(line)==1:
            charge.append(float(line[0]))
        else:
            pass

    if N != len(charge):
        print("ERROR! Number of charges does not correspond to number of atoms!")
        exit()

    return charge

###############################
# Start Program
##############################


datapath = sys.argv[1]
picklename = sys.argv[2]
dataset = get_dataset(datapath)

#print("TACO")
#print(dataset[0])

pickle_file = open(picklename, 'wb')
pickle.dump(dataset,pickle_file)
