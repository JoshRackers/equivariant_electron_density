# +
from copy import deepcopy

def flatten_list(nested_list):
    """Flatten an arbitrarily nested list, without recursion (to avoid
    stack overflows). Returns a new list, the original list is unchanged.
    >> list(flatten_list([1, 2, 3, [4], [], [[[[[[[[[5]]]]]]]]]]))
    [1, 2, 3, 4, 5]
    >> list(flatten_list([[1, 2], 3]))
    [1, 2, 3]
    """
    nested_list = deepcopy(nested_list)
    
    while nested_list:
        sublist = nested_list.pop(0)

        if isinstance(sublist, list):
            nested_list = sublist + nested_list
        else:
            yield sublist
# -

def get_iso_permuted_dataset(picklefile, amberFlag=0, **atm_iso):

    import math
    import pickle
    import torch
    import torch_geometric
    import copy

    dataset = []

    for key, value in atm_iso.items():
        if key=='h_iso':
            h_data = torch.Tensor(np.loadtxt(value,skiprows=2,usecols=1))
        elif key=='c_iso':
            c_data = torch.Tensor(np.loadtxt(value,skiprows=2,usecols=1))
        elif key=='n_iso':
            n_data = torch.Tensor(np.loadtxt(value,skiprows=2,usecols=1))
        elif key=='o_iso':
            o_data = torch.Tensor(np.loadtxt(value,skiprows=2,usecols=1))
        elif key=='p_iso':
            p_data = torch.Tensor(np.loadtxt(value,skiprows=2,usecols=1))
        else:
            raise ValueError("Isolated atom type not found. Use kwargs \"h_iso\", \"c_iso\", etc.")

    for molecule in pickle.load( open (picklefile, "rb")):
        pos = molecule['pos']
        # z is atomic number- may want to make 1,0
        z = molecule['type'].unsqueeze(1)

        x = molecule['onehot']

        c = molecule['coefficients']
        n = molecule['norms']
        exp = molecule['exponents']

        full_c = copy.deepcopy(c)        
        iso_c = torch.zeros_like(c)

        if amberFlag==1:
            amber_chg = molecule['amber_chg']
        
        #now subtract the isolated atoms
        for atom, iso, typ in zip(c,iso_c,z):
            if typ.item() == 1.0:
                atom[:list(h_data.shape)[0]] -= h_data
                iso[:list(h_data.shape)[0]] += h_data
            elif typ.item() == 6.0:
                atom[:list(c_data.shape)[0]] -= c_data
                iso[:list(c_data.shape)[0]] += c_data
            elif typ.item() == 7.0:
                atom[:list(n_data.shape)[0]] -= n_data
                iso[:list(n_data.shape)[0]] += n_data
            elif typ.item() == 8.0:
                atom[:list(o_data.shape)[0]] -= o_data
                iso[:list(o_data.shape)[0]] += o_data
            elif typ.item() == 15.0:
                atom[:list(p_data.shape)[0]] -= p_data
                iso[:list(p_data.shape)[0]] += p_data
            else:
                raise ValueError("Isolated atom type not supported!")

        pop = torch.where(n != 0, c*2*math.sqrt(2)/n, n)

        #now permute, yzx -> xyz
        p_pos = copy.deepcopy(pos)
        p_pos[:,0] = pos[:,1]
        p_pos[:,1] = pos[:,2]
        p_pos[:,2] = pos[:,0]

        if amberFlag==1:

            dataset += [torch_geometric.data.Data(pos=p_pos.to(torch.float32), 
                                              pos_orig=pos.to(torch.float32), 
                                              z=z.to(torch.float32), 
                                              x=x.to(torch.float32), 
                                              y=pop.to(torch.float32), 
                                              c=c.to(torch.float32), 
                                              full_c=full_c.to(torch.float32), 
                                              iso_c=iso_c.to(torch.float32), 
                                              exp=exp.to(torch.float32), 
                                              norm=n.to(torch.float32),
                                              amber_chg=amber_chg.to(torch.float32))]

        else:

            dataset += [torch_geometric.data.Data(pos=p_pos.to(torch.float32), 
                                              pos_orig=pos.to(torch.float32), 
                                              z=z.to(torch.float32), 
                                              x=x.to(torch.float32), 
                                              y=pop.to(torch.float32), 
                                              c=c.to(torch.float32), 
                                              full_c=full_c.to(torch.float32), 
                                              iso_c=iso_c.to(torch.float32), 
                                              exp=exp.to(torch.float32), 
                                              norm=n.to(torch.float32))]

    return dataset


def get_iso_dataset(picklefile,o_iso,h_iso):
    import math
    import pickle
    import torch
    import torch_geometric
    
    dataset = []
    
    o_data = torch.Tensor(np.loadtxt(o_iso,skiprows=2,usecols=1))
    h_data = torch.Tensor(np.loadtxt(h_iso,skiprows=2,usecols=1))
    
    for molecule in pickle.load( open (picklefile, "rb")):
        pos = molecule['pos']
        # z is atomic number- may want to make 1,0
        z = molecule['type']
        x = molecule['onehot']

        c = molecule['coefficients']
        n = molecule['norms']
        exp = molecule['exponents']
        
        #now subtract the isolated atoms
        for atom, typ in zip(c,z):
            if typ.item() == 8.0:
                atom[:list(o_data.shape)[0]] -= o_data
            elif typ.item() == 1.0:
                atom[:list(h_data.shape)[0]] -= h_data
            else:
                raise ValueError("The only supported isolated atoms are O and H")
                
        pop = torch.where(n != 0, c*2*math.sqrt(2)/n, n)
        
        dataset += [torch_geometric.data.Data(pos=pos.to(torch.float32), z=z.to(torch.float32), x=x.to(torch.float32), y=pop.to(torch.float32), exp=exp.to(torch.float32))]
        
    return dataset


def get_dataset(picklefile):
    import math
    import pickle
    import torch
    import torch_geometric
    
    dataset = []
    
    
    for molecule in pickle.load( open (picklefile, "rb")):
        pos = molecule['pos']
        # z is atomic number- may want to make 1,0
        z = molecule['type']
        x = molecule['onehot']

        c = molecule['coefficients']
        n = molecule['norms']
        exp = molecule['exponents']
                
        pop = torch.where(n != 0, c*2*math.sqrt(2)/n, n)
        
        dataset += [torch_geometric.data.Data(pos=pos.to(torch.float32), z=z.to(torch.float32), x=x.to(torch.float32), y=pop.to(torch.float32), exp=exp.to(torch.float32))]
        
    return dataset

# +
import numpy as np


# define a function to return the number of electrons in the target
# and predicted. also return the signed and unsigned error
# only working for batch size 1

def electron_error(dimer_num, true_coeffs, ml_output_coeffs, exponents, norms, Rs_outs):
    ## check number of electrons

    coeffs = true_coeffs
    output_coeffs = ml_output_coeffs
    
    counter = 0
    true_total_ele = 0.0
    ml_total_ele = 0.0
    for i, (exp, norm, rs) in enumerate(zip(exponents, norms, Rs_outs)):
        for mul, l in rs:
            # only s functions contribute
            for j in range(mul):
                for m in range(-l, l+1):
                    if l == 0:
                        c = coeffs[counter].data.detach().numpy()
                        c_ml = output_coeffs[counter].data.detach().numpy()
                        w = exp[j]
                        #normalization = (2*w/(np.pi))**(0.75)
                        normalization = norm[j]
                        #print(coeffs[counter].data.detach().numpy(),w,normalization)
                        integral = c*normalization*(1/(4*w))*np.sqrt(np.pi/w)
                        true_integral = integral
                        space = 4*np.pi
                        true_total_ele += integral*space
                        integral = c_ml*normalization*(1/(4*w))*np.sqrt(np.pi/w)
                        ml_total_ele += integral*space
                        #print("true", true_integral*space,"ml", integral*space)
                    counter += 1

    diff = ml_total_ele - true_total_ele
    signed_error = diff
    unsigned_error = np.absolute(diff)

    return signed_error, unsigned_error, true_total_ele, ml_total_ele


from scipy.special import sph_harm

def real_Y(el, m, phi, theta):
    """Get real spherical harmonic of degree el and order m"""

    # NB In SciPy's sph_harm function the azimuthal coordinate, theta,
    # comes before the polar coordinate, phi.

    # switching the order here, because scipy uses math convention.
    Y = sph_harm(abs(m), el, phi, theta)
        
    if m > 0:
        #Y = (((-1)**m)/np.sqrt(2)) * (Y + np.conj(Y))
        Y = np.sqrt(2)*((-1)**m)*np.real(Y)
    elif m < 0:
        #Y = (((-1)**m)/(np.sqrt(2)*1.j)) * (Y - np.conj(Y))
        Y = np.sqrt(2)*((-1)**m)*np.imag(Y)
    
    real_Y = np.real(Y)
    
    return real_Y


def find_min_max(coords):
    xmin, xmax = coords[0,0], coords[0,0]
    ymin, ymax = coords[0,1], coords[0,1]
    zmin, zmax = coords[0,2], coords[0,2]

    for coord in coords:
        if coord[0] < xmin:
            xmin = coord[0]
        if coord[0] > xmax:
            xmax = coord[0]
        if coord[1] < ymin:
            ymin = coord[1]
        if coord[1] > ymax:
            ymax = coord[1]
        if coord[2] < zmin:
            zmin = coord[2]
        if coord[2] > zmax:
            zmax = coord[2]
    
    return xmin, xmax, ymin, ymax, zmin, zmax


def generate_grid(data, spacing=0.5, buffer=2.0):
    buf = buffer
    xmin, xmax, ymin, ymax, zmin, zmax = find_min_max(data.pos_orig.cpu().detach().numpy())
    #buf = 2.5
    #spacing = 0.2
    # get spacing of 0.05 ang
    # the +1 is for the endpoint
    x_points = int((xmax - xmin + 2*buf)/spacing) + 1
    y_points = int((ymax - ymin + 2*buf)/spacing) + 1
    z_points = int((zmax - zmin + 2*buf)/spacing) + 1
    #print("npoints",x_points,y_points,z_points)
    npoints = int((x_points+y_points+z_points)/3)
    #print("xyz points",x_points,y_points,z_points)

    #vol = ((xmax-xmin+2*buf)/x_points)*((ymax-ymin+2*buf)/y_points)*((zmax-zmin+2*buf)/z_points)
    #print(vol)

    xlin = np.linspace(xmin-buf,xmax+buf,npoints)
    ylin = np.linspace(ymin-buf,ymax+buf,npoints)
    zlin = np.linspace(zmin-buf,zmax+buf,npoints)
    
    x_spacing = xlin[1] - xlin[0]
    y_spacing = ylin[1] - ylin[0]
    z_spacing = zlin[1] - zlin[0]
    vol = x_spacing * y_spacing * z_spacing
    
    # need 'ij' indexing for marching_cubes to work!
    x,y,z = np.meshgrid(xlin,ylin,zlin,indexing='ij')
    
    return x, y, z, vol, x_spacing, y_spacing, z_spacing


def get_old_density(x, y, z, data, ml_y, rs, vol):
    r = np.array(np.sqrt(np.square(x) + np.square(y) + np.square(z)))
    # this uses physics convention. note scipy uses math convention
    #theta = np.array(np.arccos(z/r))
    #phi = np.array(np.arctan(y/x))
    
    ml_density = np.zeros_like(r)
    target_density = np.zeros_like(r)
    
    total = 0
    for coords, coeffs, check, ml_coeffs, alpha, norm in zip(data.pos, data.full_c, data.y, ml_y, data.exp, data.norm):
        center = coords.cpu().numpy()
        #print(center)
        x_center = x - center[0]
        y_center = y - center[1]
        z_center = z - center[2]
        r_center = np.array(np.sqrt(np.square(x_center) + np.square(y_center) + np.square(z_center)))
        theta = np.array(np.arccos(z_center/r_center))
        phi = np.array(np.arctan(y_center/x_center))
        
        counter = 0
        for mul, l in rs:
            for j in range(mul):
                for m in range(-l, l+1):
                    sph = real_Y(l, m, phi, theta)
                    normal = norm[counter].cpu().detach().numpy()
                    a = alpha[counter].cpu().numpy()
                    radial = np.exp(-a * np.square(r_center))
                    #if a < 0.4:
                    #    print(radial)

                    pop_ml = ml_coeffs[counter].cpu().detach().numpy()
                    c_ml = pop_ml * normal / (2 * np.sqrt(2))
                    c_target = coeffs[counter].cpu().detach().numpy()

                    if l == 0:
                        print("only using l=0!!!")
                        ml_density += c_ml * normal * radial 
                        target_density += c_target * normal * radial 
                        #print(sph)
                        #print(c_target,a,normal)
                        #tot = vol * np.sum(c_target * normal * radial)
                        #chk = check[counter].cpu().detach().numpy()
                        #print("taco",tot,chk,tot/chk)
                        
                    counter += 1
                    
    return target_density, ml_density, total



import gau2grid as g2g

def gau2grid_density(x, y, z, data, ml_y, rs):
    # note, this takes x, y and z as flattened arrays
    #r = np.array(np.sqrt(np.square(x) + np.square(y) + np.square(z)))
    xyz = np.vstack([x,y,z])
    
    
    ml_density = np.zeros_like(x)
    target_density = np.zeros_like(x)
    
    for coords, full_coeffs, delta_coeffs, ml_coeffs, alpha, norm in zip(data.pos.cpu().detach().numpy(), data.full_c.cpu().detach().numpy(), data.c.cpu().detach().numpy(), ml_y.cpu().detach().numpy(), data.exp.cpu().detach().numpy(), data.norm.cpu().detach().numpy()):
        center = coords
        counter = 0
        for mul, l in rs:
            for j in range(mul):
                normal = norm[counter]
                exp = [alpha[counter]]
                
                angstrom2bohr = 1.8897259886
                bohr2angstrom = 1/angstrom2bohr
                
                ret_target = g2g.collocation(xyz*angstrom2bohr, l, [1], exp, center*angstrom2bohr) 
                ret_ml = g2g.collocation(xyz*angstrom2bohr, l, [1], exp, center*angstrom2bohr)
                
                target_full_coeffs = full_coeffs[counter:counter+(2*l + 1)]
                scaled_components = (target_full_coeffs * normal * ret_target["PHI"].T).T
                target_tot = np.sum(scaled_components, axis=0)
                
                pop_ml = ml_coeffs[counter:counter+(2*l + 1)]
                c_ml = pop_ml * normal / (2 * np.sqrt(2))
                target_delta_coeffs = delta_coeffs[counter:counter+(2*l + 1)]
                ml_full_coeffs = target_full_coeffs + c_ml - target_delta_coeffs
                ml_scaled_components = (ml_full_coeffs * normal * ret_target["PHI"].T).T
                ml_tot = np.sum(ml_scaled_components, axis=0)
                
                target_density += target_tot
                ml_density += ml_tot
                
                counter += 2*l + 1
                    
    return target_density, ml_density


from scipy import spatial


# NOTE: I the units of x, y, z here are assumed to be angstrom
#       I convert to bohr for gau2grid, but the grid remains in angstroms
def gau2grid_density_kdtree_old(x, y, z, data, ml_y, rs):
    # note, this takes x, y and z as flattened arrays
    #r = np.array(np.sqrt(np.square(x) + np.square(y) + np.square(z)))
    xyz = np.vstack([x,y,z])
    tree = spatial.cKDTree(xyz.T)
    
    ml_density = np.zeros_like(x)
    target_density = np.zeros_like(x)
    
    for coords, full_coeffs, delta_coeffs, ml_coeffs, alpha, norm in zip(data.pos_orig.cpu().detach().numpy(), data.full_c.cpu().detach().numpy(), data.c.cpu().detach().numpy(), ml_y.cpu().detach().numpy(), data.exp.cpu().detach().numpy(), data.norm.cpu().detach().numpy()):
        center = coords
        counter = 0
        for mul, l in rs:
            #print("Rs",mul,l)
            for j in range(mul):
                normal = norm[counter]
                if normal != 0:
                    exp = [alpha[counter]]

                    small = 1e-5
                    angstrom2bohr = 1.8897259886
                    bohr2angstrom = 1/angstrom2bohr
                    
                    target_full_coeffs = full_coeffs[counter:counter+(2*l + 1)]
                    
                    pop_ml = ml_coeffs[counter:counter+(2*l + 1)]
                    c_ml = pop_ml * normal / (2 * np.sqrt(2))
                    target_delta_coeffs = delta_coeffs[counter:counter+(2*l + 1)]
                    ml_full_coeffs = target_full_coeffs + c_ml - target_delta_coeffs
                    
                    target_max = np.amax(np.abs(target_full_coeffs))
                    ml_max = np.amax(np.abs(ml_full_coeffs))
                    max_c = np.amax(np.array([target_max, ml_max]))
                    
                    cutoff = np.sqrt((-1/exp[0])*np.log(small/np.abs(max_c*normal)))*bohr2angstrom

                    close_indices = tree.query_ball_point(center,cutoff)
                    #print("cutoff",cutoff)
                    #print(xyz.shape)
                    #print(l,len(close_indices))
                    points = np.require(xyz[:,close_indices], requirements=['C','A'])

                    ret_target = g2g.collocation(points*angstrom2bohr, l, [1], exp, center*angstrom2bohr) 
                    ret_ml = g2g.collocation(points*angstrom2bohr, l, [1], exp, center*angstrom2bohr)

                    # Now permute back to psi4 ordering
                    ##              s     p         d             f                 g
                    psi4_2_e3nn = [[0],[2,0,1],[4,2,0,1,3],[6,4,2,0,1,3,5],[8,6,4,2,0,1,3,5,7]]
                    e3nn_2_psi4 = [[0],[1,2,0],[2,3,1,4,0],[3,4,2,5,1,6,0],[4,5,3,6,2,7,1,8,0]]
                    target_full_coeffs = np.array([target_full_coeffs[k] for k in e3nn_2_psi4[l]])
                    ml_full_coeffs = np.array([ml_full_coeffs[k] for k in e3nn_2_psi4[l]])
                    
                    #target_full_coeffs = full_coeffs[counter:counter+(2*l + 1)]
                    scaled_components = (target_full_coeffs * normal * ret_target["PHI"].T).T
                    target_tot = np.sum(scaled_components, axis=0)

                    #pop_ml = ml_coeffs[counter:counter+(2*l + 1)]
                    #c_ml = pop_ml * normal / (2 * np.sqrt(2))
                    #target_delta_coeffs = delta_coeffs[counter:counter+(2*l + 1)]
                    #ml_full_coeffs = target_full_coeffs + c_ml - target_delta_coeffs
                    ml_scaled_components = (ml_full_coeffs * normal * ret_target["PHI"].T).T
                    ml_tot = np.sum(ml_scaled_components, axis=0)

                    target_density[close_indices] += target_tot
                    ml_density[close_indices] += ml_tot

                counter += 2*l + 1
                    
    return target_density, ml_density


def gau2grid_density_kdtree(x, y, z, data, ml_y, rs, isoOnlyFlag=0, small=1e-5):

    # note, this takes x, y and z as flattened arrays
    #r = np.array(np.sqrt(np.square(x) + np.square(y) + np.square(z)))
    xyz = np.vstack([x,y,z])
    tree = spatial.cKDTree(xyz.T)
    
    ml_density = np.zeros_like(x)
    target_density = np.zeros_like(x)

    
    for coords, full_coeffs, iso_coeffs, ml_coeffs, alpha, norm in zip(data.pos_orig.cpu().detach().numpy(), data.full_c.cpu().detach().numpy(), data.iso_c.cpu().detach().numpy(), ml_y.cpu().detach().numpy(), data.exp.cpu().detach().numpy(), data.norm.cpu().detach().numpy()):
        center = coords
        counter = 0
        for mul, l in rs:
            #print("Rs",mul,l)
            for j in range(mul):
                normal = norm[counter]
                if normal != 0:
                    exp = [alpha[counter]]

                    angstrom2bohr = 1.8897259886
                    bohr2angstrom = 1/angstrom2bohr

                    target_full_coeffs = full_coeffs[counter:counter+(2*l + 1)]
                    
                    pop_ml = ml_coeffs[counter:counter+(2*l + 1)]
                    c_ml = pop_ml * normal / (2 * np.sqrt(2))

                    if isoOnlyFlag==1:
                        c_ml = 0.0 # ML coeff set to 0.0; only isolated atom used

                    ml_full_coeffs = c_ml + iso_coeffs[counter:counter+(2*l + 1)]
                    
                    target_max = np.amax(np.abs(target_full_coeffs))
                    ml_max = np.amax(np.abs(ml_full_coeffs))
                    max_c = np.amax(np.array([target_max, ml_max]))
                    
                    cutoff = np.sqrt((-1/exp[0])*np.log(small/np.abs(max_c*normal)))*bohr2angstrom

                    close_indices = tree.query_ball_point(center,cutoff)
                    #print("cutoff",cutoff)
                    #print(xyz.shape)
                    #print(l,len(close_indices))
                    points = np.require(xyz[:,close_indices], requirements=['C','A'])

                    ret_target = g2g.collocation(points*angstrom2bohr, l, [1], exp, center*angstrom2bohr) 
                    ret_ml = g2g.collocation(points*angstrom2bohr, l, [1], exp, center*angstrom2bohr)

                    # Now permute back to psi4 ordering
                    ##              s     p         d             f                 g
                    psi4_2_e3nn = [[0],[2,0,1],[4,2,0,1,3],[6,4,2,0,1,3,5],[8,6,4,2,0,1,3,5,7]]
                    e3nn_2_psi4 = [[0],[1,2,0],[2,3,1,4,0],[3,4,2,5,1,6,0],[4,5,3,6,2,7,1,8,0]]
                    target_full_coeffs = np.array([target_full_coeffs[k] for k in e3nn_2_psi4[l]])
                    ml_full_coeffs = np.array([ml_full_coeffs[k] for k in e3nn_2_psi4[l]])
                    
                    #target_full_coeffs = full_coeffs[counter:counter+(2*l + 1)]
                    scaled_components = (target_full_coeffs * normal * ret_target["PHI"].T).T
                    target_tot = np.sum(scaled_components, axis=0)

                    #pop_ml = ml_coeffs[counter:counter+(2*l + 1)]
                    #c_ml = pop_ml * normal / (2 * np.sqrt(2))
                    #target_delta_coeffs = delta_coeffs[counter:counter+(2*l + 1)]
                    #ml_full_coeffs = target_full_coeffs + c_ml - target_delta_coeffs
                    ml_scaled_components = (ml_full_coeffs * normal * ret_target["PHI"].T).T
                    ml_tot = np.sum(ml_scaled_components, axis=0)

                    target_density[close_indices] += target_tot
                    ml_density[close_indices] += ml_tot

                counter += 2*l + 1

    print("small: ", small)
                    
    return target_density, ml_density


def get_scalar_density_comparisons(data, y_ml, Rs, spacing=0.5, buffer=2.0):
    # generate grid in xyz input units (angstroms)
    x,y,z,vol,x_spacing,y_spacing,z_spacing = generate_grid(data, spacing=spacing, buffer=buffer)
    # get density on grid
    target_density, ml_density = gau2grid_density_kdtree(x.flatten(),y.flatten(),z.flatten(),data,y_ml,Rs)
    
    # density is in e-/bohr**3
    angstrom2bohr = 1.8897259886
    bohr2angstrom = 1/angstrom2bohr
    
    num_ele_target = np.sum(target_density)*vol*angstrom2bohr**3
    num_ele_ml = np.sum(ml_density)*vol*angstrom2bohr**3

    numer = np.sum((ml_density - target_density)**2)
    denom = np.sum(ml_density**2) + np.sum(target_density**2)
    bigI = numer/denom
    
    #n_ele = np.sum(data.z.cpu().detach().numpy())
    #ep = 100 * vol * (angstrom2bohr**3) * np.sum(np.abs(target_density - ml_density)) / n_ele
    ep = 100 * np.sum(np.abs(ml_density-target_density)) / np.sum(target_density)
    
    return num_ele_target, num_ele_ml, bigI, ep


from concurrent import futures
import itertools

def gau2grid_density_parallel(x, y, z, data, ml_y, rs):
    # note, this takes x, y and z as flattened arrays
    r = np.array(np.sqrt(np.square(x) + np.square(y) + np.square(z)))
    xyz = np.vstack([x,y,z])
    tree = spatial.cKDTree(xyz.T)
    
    num_atoms = ml_y.shape[0]
    nxyz = np.stack([xyz]*num_atoms)
    nrs = list(itertools.repeat(rs, num_atoms))
    ntree = list(itertools.repeat(tree, num_atoms))
    
#     print(nxyz.shape)
#     print(data.pos.cpu().detach().numpy().shape)
#     print(data.full_c.cpu().detach().numpy().shape)
        
    ml_density = np.zeros_like(x)
    target_density = np.zeros_like(x)
    
    with futures.ProcessPoolExecutor(max_workers=num_atoms) as pool:
        for total in pool.map(get_dens, data.pos.cpu().detach().numpy(), data.full_c.cpu().detach().numpy(), data.c.cpu().detach().numpy(), ml_y.cpu().detach().numpy(), data.exp.cpu().detach().numpy(), data.norm.cpu().detach().numpy(), nxyz, nrs, ntree):
            target_density += total[0]
            ml_density += total[1]
                    
    return target_density, ml_density



def get_dens(coords, full_coeffs, delta_coeffs, ml_coeffs, alpha, norm, xyz, rs, tree):
    atom_ml_density = np.zeros_like(xyz[0])
    atom_target_density = np.zeros_like(xyz[0])
    center = coords
    counter = 0
    #rs = [(12, 0), (5, 1), (4, 2), (2, 3), (1, 4)]
    for mul, l in rs:
        for j in range(mul):
            normal = norm[counter]
            if normal != 0:
                exp = [alpha[counter]]
                
                small = 1e-5
                angstrom2bohr = 1.8897259886
                bohr2angstrom = 1/angstrom2bohr

                target_full_coeffs = full_coeffs[counter:counter+(2*l + 1)]

                pop_ml = ml_coeffs[counter:counter+(2*l + 1)]
                c_ml = pop_ml * normal / (2 * np.sqrt(2))
                target_delta_coeffs = delta_coeffs[counter:counter+(2*l + 1)]
                ml_full_coeffs = target_full_coeffs + c_ml - target_delta_coeffs

                target_max = np.amax(np.abs(target_full_coeffs))
                ml_max = np.amax(np.abs(ml_full_coeffs))
                max_c = np.amax(np.array([target_max, ml_max]))

                cutoff = np.sqrt((-1/exp[0])*np.log(small/np.abs(max_c*normal)))*bohr2angstrom
                close_indices = tree.query_ball_point(center,cutoff)
                points = np.require(xyz[:,close_indices], requirements=['C','A'])
                
                ret_target = g2g.collocation(points*angstrom2bohr, l, [1], exp, center*angstrom2bohr) 
                ret_ml = g2g.collocation(points*angstrom2bohr, l, [1], exp, center*angstrom2bohr)
                
                scaled_components = (target_full_coeffs * normal * ret_target["PHI"].T).T
                target_tot = np.sum(scaled_components, axis=0)

                ml_scaled_components = (ml_full_coeffs * normal * ret_target["PHI"].T).T
                ml_tot = np.sum(ml_scaled_components, axis=0)

                atom_target_density[close_indices] += target_tot
                atom_ml_density[close_indices] += ml_tot
            
            counter += 2*l + 1
    
    return atom_target_density, atom_ml_density





# def gau2grid_density_parallel2(x, y, z, data, ml_y, rs):
#     # note, this takes x, y and z as flattened arrays
#     r = np.array(np.sqrt(np.square(x) + np.square(y) + np.square(z)))
#     xyz = np.vstack([x,y,z])
#     num_atoms = ml_y.shape[0]
#     nxyz = np.stack([xyz]*num_atoms)
#     nrs = np.stack([rs]*num_atoms)
#     #nrs = list(itertools.repeat(rs, num_atoms))
#     #print(nrs)
    
# #     print(nxyz.shape)
# #     print(data.pos.cpu().detach().numpy().shape)
# #     print(data.full_c.cpu().detach().numpy().shape)
        
#     ml_density = np.zeros_like(x)
#     target_density = np.zeros_like(x)
    
#     with futures.ProcessPoolExecutor() as pool:
#         for total in pool.map(get_dens2, data.pos.cpu().detach().numpy(), data.full_c.cpu().detach().numpy(), data.c.cpu().detach().numpy(), ml_y.cpu().detach().numpy(), data.exp.cpu().detach().numpy(), data.norm.cpu().detach().numpy(), nxyz, nrs):
#             target_density += total[0]
#             ml_density += total[1]
                    
#     return target_density, ml_density



# def get_dens2(coords, full_coeffs, delta_coeffs, ml_coeffs, alpha, norm, xyz, rs):
#     atom_ml_density = np.zeros_like(xyz[0])
#     atom_target_density = np.zeros_like(xyz[0])
#     counter = 0
#     num_rs = len(rs)
#     #print(rs)
#     #print(num_rs)
#     nrs_coords = np.stack([coords]*num_rs)
#     nrs_full_coeffs = np.stack([full_coeffs]*num_rs)
#     nrs_delta_coeffs = np.stack([delta_coeffs]*num_rs)
#     nrs_ml_coeffs = np.stack([ml_coeffs]*num_rs)
#     nrs_alpha = np.stack([alpha]*num_rs)
#     nrs_norm = np.stack([norm]*num_rs)
#     nrs_xyz = np.stack([xyz]*num_rs)
    
#     #compute starting points from rs
#     starting_points = []
#     count = 0
#     for mul, l in rs:
#         starting_points.append(count)
#         count += mul*(2*l + 1)
        
    
#     with futures.ProcessPoolExecutor() as pool:
#         for total in pool.map(get_l, nrs_coords, nrs_full_coeffs, nrs_delta_coeffs, nrs_ml_coeffs, nrs_alpha, nrs_norm, nrs_xyz, rs, starting_points):
#             atom_target_density += total[0]
#             atom_ml_density += total[1]
            
#     return atom_target_density, atom_ml_density

            
# def get_l(coords, full_coeffs, delta_coeffs, ml_coeffs, alpha, norm, xyz, rs, starting_point):
#     l_ml_density = np.zeros_like(xyz[0])
#     l_target_density = np.zeros_like(xyz[0])
    
#     mul = rs[0]
#     l = rs[1]
#     center = coords
#     # need to get starting point from rs
#     counter = starting_point
#     for j in range(mul):
#         normal = norm[counter]
#         exp = [alpha[counter]]
#         ret_target = g2g.collocation(xyz, l, [1], exp, center) 
#         ret_ml = g2g.collocation(xyz, l, [1], exp, center)

#         target_full_coeffs = full_coeffs[counter:counter+(2*l + 1)]
#         scaled_components = (target_full_coeffs * normal * ret_target["PHI"].T).T
#         target_tot = np.sum(scaled_components, axis=0)

#         pop_ml = ml_coeffs[counter:counter+(2*l + 1)]
#         c_ml = pop_ml * normal / (2 * np.sqrt(2))
#         target_delta_coeffs = delta_coeffs[counter:counter+(2*l + 1)]
#         ml_full_coeffs = target_full_coeffs + c_ml - target_delta_coeffs
#         ml_scaled_components = (ml_full_coeffs * normal * ret_target["PHI"].T).T
#         ml_tot = np.sum(ml_scaled_components, axis=0)

#         counter += 2*l + 1
        
#         l_target_density += target_tot
#         l_ml_density += ml_tot
    
#     return l_target_density, l_ml_density


import psi4
import time

def compute_potential_field_old(xs,ys,zs,data,y_ml,Rs):
    # xs,ys,zs are the vertices of the isosurface
    
    # define molecule
    coords = data.pos_orig.tolist()
    atomic_nums = data.z.tolist()
    string_coords = []
    for item, anum in zip(coords, atomic_nums):
        string = ' '.join([str(elem) for elem in item])
        if anum[0] == 1.0:
            line = ' H  ' + string
        elif anum[0] == 6.0:
            line = ' C  ' + string
        elif anum[0] == 7.0:
            line = ' N  ' + string
        elif anum[0] == 8.0:
            line = ' O  ' + string
        elif anum[0] == 15.0:
            line = ' P ' + string
        else:
            raise ValueError("Atom type not found!")
        string_coords.append(line)
    molstr = """
    {}
     symmetry c1
     no_reorient
     no_com
     -2 1
    """.format("\n".join(string_coords))
    #print(molstr)
    mol = psi4.geometry(molstr)
    
    # now build the auxiliary basis set
    auxbasis = "def2-universal-jfit-dna"
    psi4.core.set_global_option('df_basis_scf', auxbasis)
    aux_basis = psi4.core.BasisSet.build(mol, "DF_BASIS_SCF", "", "JFIT", auxbasis, quiet=False)
    zero_basis = psi4.core.BasisSet.zero_ao_basis_set()
 
    # now build the integrals
    factory = psi4.core.IntegralFactory(aux_basis, zero_basis, zero_basis, zero_basis)
    nbf = aux_basis.nbf() 
 
    # get nuclear coordinates and charges
    # these coordinates are in bohr!
    natom = mol.natom()
    Zs = []
    coords = []
    for atom in range(natom):
        coords.append(np.array([mol.x(atom), mol.y(atom), mol.z(atom)]))
        Zs.append(mol.Z(atom))

    
    # now loop through the points
    target_potential = []
    ml_potential = []
    target_field = []
    ml_field = []
    
    # this assumes that the points here are given in angstroms
    for x_pt, y_pt, z_pt in zip(xs,ys,zs):
        x_bohr = x_pt/psi4.constants.bohr2angstroms
        y_bohr = y_pt/psi4.constants.bohr2angstroms
        z_bohr = z_pt/psi4.constants.bohr2angstroms
        
        # get nuclear potential
        nuc_potential = 0.0
        nuc_field = np.array([0.0, 0.0, 0.0])
        for center, charge in zip(coords, Zs):
            x_nuc = x_bohr - center[0]
            y_nuc = y_bohr - center[1]
            z_nuc = z_bohr - center[2]
            r_nuc = np.sqrt(x_nuc**2 + y_nuc**2 + z_nuc**2)
            #print("pos",x_pt,y_pt,z_pt)
            #print("center",center)
            #print("rnuc",r_nuc)
            if r_nuc > 0.00001:
                nuc_potential += (charge/r_nuc).squeeze()
                nuc_field[0] += (charge*x_nuc/(r_nuc**3)).squeeze()
                nuc_field[1] += (charge*y_nuc/(r_nuc**3)).squeeze()
                nuc_field[2] += (charge*z_nuc/(r_nuc**3)).squeeze()
                
        position = np.array([x_bohr,y_bohr,z_bohr])
        #print("pos",position)
        #print("nuc",nuc_potential,nuc_field)
        
        # note: will need to use full coefficients to get full potential, both target and ml
        target_ele_potential, target_ele_field, ml_ele_potential, ml_ele_field = get_ele_potential_field(position,factory,nbf,data,y_ml,Rs)
        
        #print("nuc", nuc_potential, nuc_field)
        #print("target",target_ele_potential,target_ele_field)
        #print("    ml",ml_ele_potential,ml_ele_field)
        
        target_potential.append(nuc_potential + np.array(target_ele_potential))
        ml_potential.append(nuc_potential + np.array(ml_ele_potential))
    
        target_field.append(nuc_field + np.array(target_ele_field))
        ml_field.append(nuc_field + np.array(ml_ele_field))

    
    return np.array(target_potential), np.array(target_field), np.array(ml_potential), np.array(ml_field)


def get_ele_potential_field_old(position, factory, nbf, data, ml_delta_pop, Rs): 
    # get the coefficients
    target_coeffs = data.full_c.cpu().detach().numpy()
    
    ml_delta_coeffs = ml_delta_pop.cpu().detach().numpy() * data.norm.cpu().detach().numpy() / (2 * np.sqrt(2))
    #target_delta_coeffs = data.c.cpu().detach().numpy()
    #ml_full_coeffs = target_coeffs + ml_delta_coeffs - target_delta_coeffs
    ml_full_coeffs = ml_delta_coeffs + data.iso_c.cpu().detach().numpy()
    
    # switch spherical harmonic ordering back to psi4 convention
    new_target_coeffs = e3nn_2_psi4_ordering(target_coeffs,Rs)
    new_ml_full_coeffs = e3nn_2_psi4_ordering(ml_full_coeffs,Rs)
    
    # now flatten, remove zeros and expand by empty dimension
    new_target_coeffs = new_target_coeffs.flatten()
    new_target_coeffs = new_target_coeffs[new_target_coeffs != 0]
    new_target_coeffs = np.expand_dims(new_target_coeffs, axis=1)
    
    new_ml_full_coeffs = new_ml_full_coeffs.flatten()
    new_ml_full_coeffs = new_ml_full_coeffs[new_ml_full_coeffs != 0]
    new_ml_full_coeffs = np.expand_dims(new_ml_full_coeffs, axis=1)
        
    # now get integrals! need custom built psi4 code for this
    ints = factory.ao_multipole_potential(1)
    
    results = [psi4.core.Matrix(nbf,1) for i in range(4)]
    posvec = psi4.core.Vector3(*position)
    ints.origin = posvec
    for mat in results:
        mat.zero()
    ints.compute(results)
    
    #print("target_coeffs",target_coeffs.shape)
    #print(target_coeffs[23,27:32])
    
    target_ele_terms = np.array([np.einsum('ab,ab', results[i].np, new_target_coeffs) for i in range(4)])
    sign_change = np.array([-1,1,1,1])
    target_ele_terms = target_ele_terms*sign_change
    target_ele_potential = target_ele_terms[0]
    target_ele_field = target_ele_terms[1:]
    
    ml_ele_terms = np.array([np.einsum('ab,ab', results[i].np, new_ml_full_coeffs) for i in range(4)])
    sign_change = np.array([-1,1,1,1])
    ml_ele_terms = ml_ele_terms*sign_change
    ml_ele_potential = ml_ele_terms[0]
    ml_ele_field = ml_ele_terms[1:]
    
    return target_ele_potential, target_ele_field, ml_ele_potential, ml_ele_field
    
    
    
def e3nn_2_psi4_ordering(coeffs,Rs):
    list_coeffs = []
    for atom in coeffs:
        coeffs_list = []
        counter = 0
        for mul, l in Rs:
            for i in range(mul):
                step = 2*l + 1
                coeffs_list.append(atom[counter:counter+step].tolist())
                counter += step
        list_coeffs.append(coeffs_list)
        
    ##              s     p         d             f                 g
    psi4_2_e3nn = [[0],[2,0,1],[4,2,0,1,3],[6,4,2,0,1,3,5],[8,6,4,2,0,1,3,5,7]]
    e3nn_2_psi4 = [[0],[1,2,0],[2,3,1,4,0],[3,4,2,5,1,6,0],[4,5,3,6,2,7,1,8,0]]
    
    '''
    test = [[0],[-1, 0, +1],[-2, -1, 0, +1, +2],[-3, -2, -1, 0, +1, +2, +3],[-4, -3, -2, -1, 0, +1, +2, +3, +4]]

    e3nn_2_psi4 = [[0],[1,2,0],[2,3,1,4,0],[3,4,2,5,1,6,0],[4,5,3,6,2,7,1,8,0]]
    for j, item in enumerate(test):
        l = (len(item)-1)//2
        if l > 4:
            raise ValueError('L is too high. Currently only supports L<5')
        test[j] = [item[k] for k in e3nn_2_psi4[l]]
    print(test)
    '''
        
    for i, atom in enumerate(list_coeffs):
        for j, item in enumerate(atom):
            #print(item)
            l = (len(item)-1)//2
            if l > 4:
                raise ValueError('L is too high. Currently only supports L<5')
            list_coeffs[i][j] = [item[k] for k in e3nn_2_psi4[l]]
        
    rect_coeffs = []
    for atom in list_coeffs:
        flat_atom = list(flatten_list(atom))
        rect_coeffs.append(flat_atom)
    
    new_coeffs = np.array(rect_coeffs)
    
    return new_coeffs

def compute_amber_potential(xs,ys,zs,data,y_ml,Rs):
    # xs,ys,zs are the vertices of the isosurface

    # need to modify molstr to match target molecule and basis
    # possibly change later on
    
    # define molecule
    coords = data.pos_orig.tolist()
    atomic_nums = data.z.tolist()
    amber_Zs = data.amber_chg.tolist()
    string_coords = []
    for item, anum in zip(coords, atomic_nums):
        string = ' '.join([str(elem) for elem in item])
        if anum[0] == 1.0:
            line = ' H  ' + string
        elif anum[0] == 6.0:
            line = ' C  ' + string
        elif anum[0] == 7.0:
            line = ' N  ' + string
        elif anum[0] == 8.0:
            line = ' O  ' + string
        elif anum[0] == 15.0:
            line = ' P ' + string
        else:
            raise ValueError("Atom type not found!")
        string_coords.append(line)
    molstr = """
    {}
     symmetry c1
     no_reorient 
     no_com
     -2 1
    """.format("\n".join(string_coords))
    #print(molstr)
    mol = psi4.geometry(molstr)
     
    # get nuclear coordinates and AMBER charges
    # these coordinates are in bohr!
    natom = mol.natom()
    coords = []
    for atom in range(natom):
        coords.append(np.array([mol.x(atom), mol.y(atom), mol.z(atom)]))

    if len(amber_Zs) != len(coords):
        print("ERROR! Lengths of amber charges not the same as coords")
        exit()
    
    # now loop through the points
    target_potential = []
    ml_potential = []
    target_field = []
    ml_field = []
 
    # this assumes that the points here are given in angstroms
    for x_pt, y_pt, z_pt in zip(xs,ys,zs):
        x_bohr = x_pt/psi4.constants.bohr2angstroms
        y_bohr = y_pt/psi4.constants.bohr2angstroms
        z_bohr = z_pt/psi4.constants.bohr2angstroms
        
        # get amber potential
        nuc_potential = 0.0
        nuc_field = np.array([0.0, 0.0, 0.0])
        for center, charge in zip(coords, amber_Zs):
            x_nuc = x_bohr - center[0]
            y_nuc = y_bohr - center[1]
            z_nuc = z_bohr - center[2]
            r_nuc = np.sqrt(x_nuc**2 + y_nuc**2 + z_nuc**2)
            #print("pos",x_pt,y_pt,z_pt)
            #print("center",center)
            #print("rnuc",r_nuc)
            if r_nuc > 0.00001:
                nuc_potential += (charge/r_nuc).squeeze()
                
        position = np.array([x_bohr,y_bohr,z_bohr])
         
        target_potential.append(nuc_potential)    
    
    return np.array(target_potential)

def compute_potential_field_ml(xs,ys,zs,data,y_ml,Rs, isoOnlyFlag=0):
    # xs,ys,zs are the vertices of the isosurface
    
    # define molecule
    coords = data.pos_orig.tolist()
    atomic_nums = data.z.tolist()
    string_coords = []
    for item, anum in zip(coords, atomic_nums):
        string = ' '.join([str(elem) for elem in item])
        if anum[0] == 1.0:
            line = ' H  ' + string
        elif anum[0] == 6.0:
            line = ' C  ' + string
        elif anum[0] == 7.0:
            line = ' N  ' + string
        elif anum[0] == 8.0:
            line = ' O  ' + string
        elif anum[0] == 15.0:
            line = ' P ' + string
        else:
            raise ValueError("Atom type not found!")
        string_coords.append(line)
    molstr = """
    {}
     symmetry c1
     no_reorient
     no_com
     -4 1
    """.format("\n".join(string_coords))
    #print(molstr)
    mol = psi4.geometry(molstr)
    
    # now build the auxiliary basis set
    auxbasis = "def2-universal-jfit-dna"
    psi4.core.set_global_option('df_basis_scf', auxbasis)
    aux_basis = psi4.core.BasisSet.build(mol, "DF_BASIS_SCF", "", "JFIT", auxbasis, quiet=False)
    zero_basis = psi4.core.BasisSet.zero_ao_basis_set()
    
    # now build the integrals
    factory = psi4.core.IntegralFactory(aux_basis, zero_basis, zero_basis, zero_basis)
    nbf = aux_basis.nbf() 
 
    # get nuclear coordinates and charges
    # these coordinates are in bohr!
    natom = mol.natom()
    Zs = []
    coords = []
    for atom in range(natom):
        coords.append(np.array([mol.x(atom), mol.y(atom), mol.z(atom)]))
        Zs.append(mol.Z(atom))

    # get the coefficients
    
    ml_delta_coeffs = y_ml.cpu().detach().numpy() * data.norm.cpu().detach().numpy() / (2 * np.sqrt(2))

    # hacky way to get isolated coeffs (cannot set equal to 0)
    if isoOnlyFlag==1:
        ml_delta_coeffs = 1e-6*y_ml.cpu().detach().numpy() * data.norm.cpu().detach().numpy() / (2 * np.sqrt(2))


    ml_full_coeffs = ml_delta_coeffs + data.iso_c.cpu().detach().numpy()
    
    # switch spherical harmonic ordering back to psi4 convention
    new_ml_full_coeffs = e3nn_2_psi4_ordering(ml_full_coeffs,Rs)
    
    # now flatten, remove zeros and expand by empty dimension
    
    new_ml_full_coeffs = new_ml_full_coeffs.flatten()
    new_ml_full_coeffs = new_ml_full_coeffs[new_ml_full_coeffs != 0]
    new_ml_full_coeffs = np.expand_dims(new_ml_full_coeffs, axis=1)
 
    # now get integrals! need custom built psi4 code for this
    ints = factory.ao_multipole_potential(1) 
    
    # now loop through the points
    ml_potential = []
    ml_field = []
    
    # this assumes that the points here are given in angstroms
    for x_pt, y_pt, z_pt in zip(xs,ys,zs):
        x_bohr = x_pt/psi4.constants.bohr2angstroms
        y_bohr = y_pt/psi4.constants.bohr2angstroms
        z_bohr = z_pt/psi4.constants.bohr2angstroms
        
        # get nuclear potential
        nuc_potential = 0.0
        nuc_field = np.array([0.0, 0.0, 0.0])
        for center, charge in zip(coords, Zs):
            x_nuc = x_bohr - center[0]
            y_nuc = y_bohr - center[1]
            z_nuc = z_bohr - center[2]
            r_nuc = np.sqrt(x_nuc**2 + y_nuc**2 + z_nuc**2)
            #print("pos",x_pt,y_pt,z_pt)
            #print("center",center)
            #print("rnuc",r_nuc)
            if r_nuc > 0.00001:
                nuc_potential += (charge/r_nuc).squeeze()
                nuc_field[0] += (charge*x_nuc/(r_nuc**3)).squeeze()
                nuc_field[1] += (charge*y_nuc/(r_nuc**3)).squeeze()
                nuc_field[2] += (charge*z_nuc/(r_nuc**3)).squeeze()
                
        position = np.array([x_bohr,y_bohr,z_bohr])
        #print("pos",position)
        #print("nuc",nuc_potential,nuc_field)
        
        # note: will need to use full coefficients to get full potential, both target and ml
        ml_ele_potential, ml_ele_field = get_ele_potential_field(position,ints,nbf,data,new_ml_full_coeffs,Rs) 
        
        ml_potential.append(nuc_potential + np.array(ml_ele_potential))
    
        ml_field.append(nuc_field + np.array(ml_ele_field))
 
    return np.array(ml_potential), np.array(ml_field)

def get_ele_potential_field(position, ints, nbf, data, new_coeffs, Rs): 

    results = [psi4.core.Matrix(nbf,1) for i in range(4)]
    posvec = psi4.core.Vector3(*position)
    ints.origin = posvec
    for mat in results:
        mat.zero()
    ints.compute(results)
    
    #print("target_coeffs",target_coeffs.shape)
    #print(target_coeffs[23,27:32])
     
    ml_ele_terms = np.array([np.einsum('ab,ab', results[i].np, new_coeffs) for i in range(4)])
    sign_change = np.array([-1,1,1,1])
    ml_ele_terms = ml_ele_terms*sign_change
    ml_ele_potential = ml_ele_terms[0]
    ml_ele_field = ml_ele_terms[1:]
    
    return ml_ele_potential, ml_ele_field
    
def compute_potential_field_target(xs,ys,zs,data,y_ml,Rs):
    # xs,ys,zs are the vertices of the isosurface
    
    # define molecule
    coords = data.pos_orig.tolist()
    atomic_nums = data.z.tolist()
    string_coords = []
    for item, anum in zip(coords, atomic_nums):
        string = ' '.join([str(elem) for elem in item])
        if anum[0] == 1.0:
            line = ' H  ' + string
        elif anum[0] == 6.0:
            line = ' C  ' + string
        elif anum[0] == 7.0:
            line = ' N  ' + string
        elif anum[0] == 8.0:
            line = ' O  ' + string
        elif anum[0] == 15.0:
            line = ' P ' + string
        else:
            raise ValueError("Atom type not found!")
        string_coords.append(line)
    molstr = """
    {}
     symmetry c1
     no_reorient
     no_com
     -2 1
    """.format("\n".join(string_coords))
    #print(molstr)
    mol = psi4.geometry(molstr)
    
    # now build the auxiliary basis set
    auxbasis = "def2-universal-jfit-dna"
    psi4.core.set_global_option('df_basis_scf', auxbasis)
    aux_basis = psi4.core.BasisSet.build(mol, "DF_BASIS_SCF", "", "JFIT", auxbasis, quiet=False)
    zero_basis = psi4.core.BasisSet.zero_ao_basis_set()
 
    # now build the integrals
    factory = psi4.core.IntegralFactory(aux_basis, zero_basis, zero_basis, zero_basis)
    nbf = aux_basis.nbf() 
 
    # get nuclear coordinates and charges
    # these coordinates are in bohr!
    natom = mol.natom()
    Zs = []
    coords = []
    for atom in range(natom):
        coords.append(np.array([mol.x(atom), mol.y(atom), mol.z(atom)]))
        Zs.append(mol.Z(atom))

    # get the coefficients
    target_coeffs = data.full_c.cpu().detach().numpy()

    # switch spherical harmonic ordering back to psi4 convention
    new_target_coeffs = e3nn_2_psi4_ordering(target_coeffs,Rs)
    
    # now flatten, remove zeros and expand by empty dimension
    new_target_coeffs = new_target_coeffs.flatten()
    new_target_coeffs = new_target_coeffs[new_target_coeffs != 0]
    new_target_coeffs = np.expand_dims(new_target_coeffs, axis=1)
     
    # now get integrals! need custom built psi4 code for this
    ints = factory.ao_multipole_potential(1) 
    
    # now loop through the points
    target_potential = []
    target_field = []
    
    # this assumes that the points here are given in angstroms
    for x_pt, y_pt, z_pt in zip(xs,ys,zs):
        x_bohr = x_pt/psi4.constants.bohr2angstroms
        y_bohr = y_pt/psi4.constants.bohr2angstroms
        z_bohr = z_pt/psi4.constants.bohr2angstroms
        
        # get nuclear potential
        nuc_potential = 0.0
        nuc_field = np.array([0.0, 0.0, 0.0])
        for center, charge in zip(coords, Zs):
            x_nuc = x_bohr - center[0]
            y_nuc = y_bohr - center[1]
            z_nuc = z_bohr - center[2]
            r_nuc = np.sqrt(x_nuc**2 + y_nuc**2 + z_nuc**2)
            #print("pos",x_pt,y_pt,z_pt)
            #print("center",center)
            #print("rnuc",r_nuc)
            if r_nuc > 0.00001:
                nuc_potential += (charge/r_nuc).squeeze()
                nuc_field[0] += (charge*x_nuc/(r_nuc**3)).squeeze()
                nuc_field[1] += (charge*y_nuc/(r_nuc**3)).squeeze()
                nuc_field[2] += (charge*z_nuc/(r_nuc**3)).squeeze()
                
        position = np.array([x_bohr,y_bohr,z_bohr])
        #print("pos",position)
        #print("nuc",nuc_potential,nuc_field)
        
        # note: will need to use full coefficients to get full potential, both target and ml
        target_ele_potential, target_ele_field = get_ele_potential_field(position,ints,nbf,data,new_target_coeffs,Rs)
        
        #print("nuc", nuc_potential, nuc_field)
        #print("target",target_ele_potential,target_ele_field)
        #print("    ml",ml_ele_potential,ml_ele_field)
        
        target_potential.append(nuc_potential + np.array(target_ele_potential))
    
        target_field.append(nuc_field + np.array(target_ele_field))
    
    return np.array(target_potential), np.array(target_field)
