
import sys
import numpy as np
import psi4


# read in args

# structure file
xyzfile = sys.argv[1]
# orbital basis
basisname = sys.argv[2]
# auxiliary basis
auxbasis = sys.argv[3]
# level of theory
theory = sys.argv[4]

xyzprefix = xyzfile.split('.')[0]

psi4.set_memory('128 GB')
psi4.set_num_threads(16)
psi4.core.set_output_file('output_' + xyzfile + '.dat', False)

ang2bohr = 1.88973
bohr2ang = 1/ang2bohr

#necessary to skip the first two lines of standard xyz file format
with open(xyzfile) as f:
    temp = f.readlines()[2:]

molstr = ' '.join(temp) 
molstr = molstr + "\n symmetry c1 \n no_reorient \n no_com \n"
mol = psi4.geometry(molstr)

print("Computing " + theory + " gradient...")
grad, wfn = psi4.gradient('{}/{}'.format(theory,basisname), return_wfn=True)
print("finished gradient calculation")
print("")

print("Performing density fit with " + auxbasis + " basis set...")
psi4.core.set_global_option('df_basis_scf', auxbasis)

orbital_basis = wfn.basisset()
aux_basis = psi4.core.BasisSet.build(mol, "DF_BASIS_SCF", "", "JFIT", auxbasis)
#aux_basis.print_detail_out()

numfuncatom = np.zeros(mol.natom())
funcmap = []
shells = []

# note: atoms are 0 indexed
for func in range(0, aux_basis.nbf()):
    current = aux_basis.function_to_center(func)
    shell = aux_basis.function_to_shell(func)
    shells.append(shell)

    funcmap.append(current)
    numfuncatom[current] += 1

shellmap = []
for shell in range(0, aux_basis.nshell()):
    count = shells.count(shell)
    shellmap.append((count-1)//2)

# print(numfuncatom)

zero_basis = psi4.core.BasisSet.zero_ao_basis_set()
mints = psi4.core.MintsHelper(orbital_basis)

#
# Check normalization of the aux basis
#
#Saux = np.array(mints.ao_overlap(aux_basis, aux_basis))
#print(Saux)

#
# Form 3 center integrals (P|mn)
#
J_Pmn = np.squeeze(mints.ao_eri(
    aux_basis, zero_basis, orbital_basis, orbital_basis))

#
# Form metric (P|Q) and invert, filtering out small eigenvalues for stability
#
J_PQ = np.squeeze(mints.ao_eri(aux_basis, zero_basis, aux_basis, zero_basis))
evals, evecs = np.linalg.eigh(J_PQ)
evals = np.where(evals < 1e-10, 0.0, 1.0/evals)
J_PQinv = np.einsum('ik,k,jk->ij', evecs, evals, evecs)

## THIS IS SLOW
#
# Recompute the integrals, as a simple sanity check (mn|rs) = (mn|P) PQinv[P,Q] (Q|rs)
# where PQinv[P,Q] is the P,Qth element of the invert of the matrix (P|Q) (a Coulomb integral)
#
#approx = np.einsum('Pmn,PQ,Qrs->mnrs', J_Pmn,
#                   J_PQinv, J_Pmn, optimize=True)
#exact = mints.ao_eri()
#print("checking how good the fit is")
#print(approx - exact)

#
# Finally, compute and print the fit coefficients.  From the density matrix, D, the
# coefficients of the vector of basis aux basis funcions |P) is given by
#
# D_P = Sum_mnQ D_mn (mn|Q) PQinv[P,Q]
#

# compute q from equations 15-17 in Dunlap paper
# "Variational fitting methods for electronic structure calculations"
q = []
counter = 0
for i in range(0, mol.natom()):
    for j in range(counter, counter + int(numfuncatom[i])):
        # print(D_P[j])
        shell_num = aux_basis.function_to_shell(j)
        shell = aux_basis.shell(shell_num)
        # assumes that each shell only has 1 primitive. true for a2 basis
        normalization = shell.coef(0)
        exponent = shell.exp(0)
        if shellmap[shell_num] == 0:
            integral = (1/(4*exponent))*np.sqrt(np.pi/exponent)
            q.append(4*np.pi*normalization*integral)
        else:
            q.append(0.0)
        counter += 1

q = np.array(q)
bigQ = wfn.nalpha() + wfn.nbeta()

D = np.array(wfn.Da()) + np.array(wfn.Db())

# these are the old coefficients
D_P = np.einsum('mn,Pmn,PQ->Q', D, J_Pmn, J_PQinv, optimize=True)

# compute lambda
numer = bigQ - np.dot(q,D_P)
denom = np.dot(np.dot(q,J_PQinv),q)
lambchop = numer/denom

new_D_P = D_P + np.dot(J_PQinv, lambchop*q)


f = open(xyzprefix + "_" + auxbasis + "_density.out", "w+")
counter = 0
totalq = 0.0
newtotalq = 0.0
for i in range(0, mol.natom()):
    f.write("Atom number: %i \n" % i)
    f.write("number of functions: %i \n" % int(numfuncatom[i]))
    for j in range(counter, counter + int(numfuncatom[i])):
        shell_num = aux_basis.function_to_shell(j)
        shell = aux_basis.shell(shell_num)
        # assumes that each shell only has 1 primitive. true for a2 basis
        normalization = shell.coef(0)
        exponent = shell.exp(0)
        integral = (1/(4*exponent))*np.sqrt(np.pi/exponent)
        
        if shellmap[shell_num] == 0:
            totalq += D_P[j]*4*np.pi*normalization*integral
            newtotalq += new_D_P[j]*4*np.pi*normalization*integral

        f.write(str(shellmap[shell_num]) + " " + np.array2string(new_D_P[j]) + 
                " " + str(exponent) + " " + str(normalization) + "\n")
        counter += 1

f.close()
