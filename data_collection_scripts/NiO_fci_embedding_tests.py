from copy import deepcopy
from pyscf.lib import chkfile
import sys
sys.path.append('/home/connorlenihan/Documents/embedding/Vayesta313/')
import vayesta

import numpy as np
import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.fci
import vayesta
import vayesta.ewf
from vayesta.misc.molecules import ring
from vayesta.core.types.wf.t_to_c import t1_uhf, t2_uhf
from vayesta.solver.ccsd import UCCSD
from vayesta.solver.cisd import UCISD

from src import cluster_solver

import os
from os import environ
from itertools import product as prod
import numpy as np

import pyscf
import pyscf.pbc
import vayesta
import vayesta.ewf
from pyscf.pbc.gto import Cell
from pyscf.pbc import scf as pbcscf
from vayesta.misc import solids
from vayesta.misc.gto_helper import get_atom_shells

from pyscf.pbc import cc as cup_clus
from pyscf import scf

def spin_spin_correlation(rdm1, rdm2, P_A, P_B):
    """
    Calculate the spin-spin correlation between atoms A and B using the corrected equation.
    
    Parameters:
    - rdm1: (np.ndarray) the one-body reduced density matrix
    - rdm2: (np.ndarray) the two-body reduced density matrix
    - P_A, P_B: (np.ndarray) projection operators for atoms A and B
    
    Returns:
    - SzASzB: (float) the spin-spin correlation value.

    Note:
    - This function should give the same result as the following snippet
    >>> from vayesta.core.qemb.corrfunc import get_corrfunc
    >>> rdm1 = np.load("rdm_1.npy")
    >>> rdm2 = np.load("rdm_2.npy")
    >>> corr = get_corrfunc(emb, "sz,sz", dm1=rdm1, dm2=rdm2)
    """
    # Transform spin-free 2-RDM into spinful term
    # print(rdm2.shape)
    # print(rdm2)
    rdm2 = -(rdm2 / 6 + rdm2.transpose(0, 3, 2, 1) / 3)
    # print();print()
    # print(rdm2.shape)
    # print(rdm2)
    # Apply projection operators to the one-body RDMs and sum over indices i and j
    one_body_term = np.einsum('ik,jk,ij->', P_A, P_B, rdm1)

    # Apply projection operators to the two-body RDMs and sum over indices i, j, k, and l
    two_body_term = np.einsum('ij,kl,ijkl->', P_A, P_B, rdm2)

    # Combine terms according to the corrected equation
    SzAszB = 0.25 * one_body_term + 0.5 * two_body_term

    return SzAszB.real

def fci_solver(mf, dm=False):
    h1e = mf.get_hcore()
    n_pada=0
    for m in h1e[0][0,:]:
        if m == 0.:
            n_pada+=1
    n_padb=0
    for m in h1e[1][0,:]:
        if m == 0.:
            n_padb+=1        
    h2e = mf._eri
    # print("h1e",h1e)
    print("npad: ",n_pada,n_padb)
    # print("h2e",h2e)
    norb = (mf.mo_coeff[0].shape[1], mf.mo_coeff[1].shape[1])
    print("norb: ",norb)
    print("occ: ",mf.get_occ())
    # nelec = mf.mol.nelectron
    nocc = (int(sum(mf.get_occ()[0])),int(sum(mf.get_occ()[1])))
    nvir = (norb[0] - nocc[0],norb[1] - nocc[1])
    
    nelec = mf.mol.nelec

    norba, norbb = norb
    nocca, noccb = nocc
    nvira, nvirb = nvir
    assert norb[0]==norb[1]
    norb=norb[0]
    
    energy, civec = pyscf.fci.direct_uhf.kernel(h1e, h2e, norb, nelec, conv_tol=1.e-14)
    print("energy fci: ",energy)
    

    t1addra, t1signa = pyscf.ci.cisd.tn_addrs_signs(norba, nocca, 1)
    t1addrb, t1signb = pyscf.ci.cisd.tn_addrs_signs(norbb, noccb, 1)
    t2addra, t2signa = pyscf.ci.cisd.tn_addrs_signs(norba, nocca, 2)
    t2addrb, t2signb = pyscf.ci.cisd.tn_addrs_signs(norbb, noccb, 2)

    # Change to arrays, in case of empty slice
    t1addra = np.asarray(t1addra, dtype=int)
    t1addrb = np.asarray(t1addrb, dtype=int)

    na = pyscf.fci.cistring.num_strings(norba, nocca)
    nb = pyscf.fci.cistring.num_strings(norbb, noccb)

    ci = civec.reshape(na, nb)
    # print("c1a: ",civec[t1addra, 0])
    # print("c1b: ",civec[0, t1addrb])
    c1a = (civec[t1addra, 0] * t1signa).reshape(nocca, nvira)
    c1b = (civec[0, t1addrb] * t1signb).reshape(noccb, nvirb)
    
    nocca_comb = nocca * (nocca - 1) // 2
    noccb_comb = noccb * (noccb - 1) // 2
    nvira_comb = nvira * (nvira - 1) // 2
    nvirb_comb = nvirb * (nvirb - 1) // 2
    c2aa = (civec[t2addra, 0] * t2signa).reshape(nocca_comb, nvira_comb)
    c2bb = (civec[0, t2addrb] * t2signb).reshape(noccb_comb, nvirb_comb)
    c2aa = pyscf.cc.ccsd._unpack_4fold(c2aa, nocca, nvira)
    c2bb = pyscf.cc.ccsd._unpack_4fold(c2bb, noccb, nvirb)
    c2ab = np.einsum("i,j,ij->ij", t1signa, t1signb, civec[t1addra[:, None], t1addrb])
    c2ab = c2ab.reshape(nocca, nvira, noccb, nvirb).transpose(0, 2, 1, 3)
    # if c0 is None:
    #     c0 = c0
    # else:
    

    c0 = civec[0][0]
    

    if abs(c0)<1e-1:
        print(f"##########################################################################################\nc0 val {c0}\n##########################################################################################")
    assert abs(c0) > 1e-1
    c1a *= 1/c0
    c1b *= 1/c0
    c2aa *= 1/c0
    c2ab *= 1/c0
    c2bb *= 1/c0

    c1 = (c1a[:,:len(c1a[0])-n_pada],c1b[:,:len(c1b[0])-n_padb])#(c1a, c1b)
    c2 = (c2aa[:,:,:len(c1a[0])-n_pada,:len(c1a[0])-n_pada], c2ab[:,:,:len(c1a[0])-n_pada,:len(c1b[0])-n_padb], c2bb[:,:,:len(c1b[0])-n_padb,:len(c1b[0])-n_padb])#(c2aa, c2ab, c2bb)
    c0=1.
    print("c0 fci solver: ",civec[0][0])
    print("c1a: ",c1[0])
    print("c1b: ",c1[1])
    print()
    print("c2ab: \n",c2[0])
    print()
    print("c2ab: \n",c2[0])
    print()
    print("c2ab: \n",c2[2])
    
    # print("c2ab: \n",c2ab[:,:,:len(c1a[0])-n_pada,:])
    # print("c2aa: \n",c2aa)
    # print("c2aa: \n",c2aa[:,:,:len(c1a[0])-n_pada,:len(c1a[0])-n_pada])
    # print("c2bb: \n",c2bb)
    # print("c1a: \n",c1a)
    # print(c1a[:,:len(c1a[0])-n_pada])
    # print("c1b: \n",c1b)
    # print(c1b[:,:len(c1b[0])-n_padb])
    # c0, c1, c2 = #ci.cisdvec_to_amplitudes(civec)

    return dict(c0=c0,c1=c1,c2=c2, converged=True, energy=energy)

def solver_cs(mf):
    #Nshots=1000
    # nelec = mf.mol.nelectron 

    # Organising mean field outputs

    # mo_energy = mf.mo_energy
    # mo_coeff = mf.mo_coeff  # (ao = spatical oribital |MO) due to restricted hartree fock calculation, so no need to consider alpha/beta contributions
    #                                 # to consider spin orbitals as required in UHF and GHF. 
    # mo_occ = mf.mo_occ
    
    
    norb = (mf.mo_coeff[0].shape[1], mf.mo_coeff[1].shape[1])
    print("norb: ",norb)
    # nelec = mf.mol.nelectron
    nocc = (int(sum(mf.get_occ()[0])),int(sum(mf.get_occ()[1])))
    nvir = (norb[0] - nocc[0],norb[1] - nocc[1])

    h1e = mf.get_hcore()
    # Need to convert cderis into standard 4-index tensor when using denisty fitting for the mean-field
    # cderi = mf.with_df._cderi
    # cderi = pyscf.lib.unpack_tril(cderi)
    # h2e = np.einsum('Lpq,Lrs->pqrs', cderi, cderi)

    h1e = mf.get_hcore()

    n_pada=0
    for m in h1e[0][0,:]:
        if m == 0.:
            n_pada+=1
    n_padb=0
    for m in h1e[1][0,:]:
        if m == 0.:
            n_padb+=1  

    h2e = mf._eri
    # print("h2e: ",h2e)
    # norb = mf.mo_coeff.shape[-1]
    nelec = mf.mol.nelec



    solver=cluster_solver.cluster_solver(h1e, h2e, norb, nelec,mf.get_ovlp(),"UHF",solver="FCI")
    solver_options = {"random_initial_params": False,"attempts": 1}
    solver.solve(solver_options)
    energy_cs, ci_vec_cs=solver.measure_fci_civec_coeffs(Nshots,3,ensemble="Clifford")
    c0_cs=ci_vec_cs[0]
    c1_csa,c1_csb=ci_vec_cs[1]/c0_cs.real
    c2_csaa,c2_csab,c2_csbb=ci_vec_cs[2]/c0_cs.real
    c0_cs=1.

    ###slice off padded orbitals before returning
    c1_cs = (c1_csa[:,:len(c1_csa[0])-n_pada].real,c1_csb[:,:len(c1_csb[0])-n_padb].real)#(c1a, c1b)
    c2_cs = (c2_csaa[:,:,:len(c1_csa[0])-n_pada,:len(c1_csa[0])-n_pada].real, c2_csab[:,:,:len(c1_csa[0])-n_pada,:len(c1_csb[0])-n_padb].real, c2_csbb[:,:,:len(c1_csb[0])-n_padb,:len(c1_csb[0])-n_padb].real)#(c2aa, c2ab, c2bb)
    # To use CI amplitudues use return the following line and set energy_functional='wf' to use the projected energy in the EWF arguments below

    print("c1a: ",c1_cs[0])
    print("c1b: ",c1_cs[1])
    print()
    print("c2ab: \n",c2_cs[0])
    print()
    print("c2ab: \n",c2_cs[0])
    print()
    print("c2ab: \n",c2_cs[2])
    return dict(c0=c0_cs, c1=c1_cs, c2=c2_cs, converged=True, energy=energy_cs)



magnetism = sys.argv[1]
atomic_dis = float(sys.argv[2]) #4.1705
Nshots = int(sys.argv[3])
k_grid = int(sys.argv[4])
do_ccsd = bool(int(sys.argv[5]))




kmesh  = [k_grid]*3
# for magnetism in magnetics:
# atom_dis_list = [3.8,3.9,4.0,4.1,4.15,4.2,4.3,4.35][::1]#,4.3,3.8
# for atomic_dis in atom_dis_list: #np.arange(float(sys.argv[2]),float(sys.argv[3]),float(sys.argv[4])):

print("********************************************************************************************")
print(f"dis = {atomic_dis}")
print("********************************************************************************************")

unitcell = "primitive-af1" if "AFM1" in magnetism else "primitive-af2"
kmeshstr = "".join(map(str, kmesh))

basis = 'gth-szv-molopt-sr'#"gth-dzvp-molopt-sr" ##
pseudo = "gth-pade"

cell = Cell()
cell.a, cell.atom = solids.rocksalt(
    atoms=["Ni", "O"], a=atomic_dis, unitcell=unitcell
)  # Oxide
cell.basis = basis
cell.pseudo = pseudo
cell.output = f"NiO_{magnetism}_k{kmeshstr}_d_{atomic_dis}_bas_{basis}_pyscf.txt"
cell.verbose = 5
cell.spin = 4 * np.prod(kmesh) if "HS" in magnetism else 0  # Oxide
cell.exp_to_discard = 0.1
cell.build()
kpts = cell.make_kpts(kmesh, time_reversal_symmetry=True)

kmeshstr = "".join(map(str, kmesh))

mf = pbcscf.KUHF(cell, kpts)
mf = mf.density_fit()
df = mf.with_df

scf_result_dic = chkfile.load("mf_chk_"+magnetism+"_k"+str(kmesh[0])+str(kmesh[1])+str(kmesh[2])+"_basis_"+basis+"_d"+str(atomic_dis)+".chk","scf")#'mf_chk_'+magnetism+'_'+str(atomic_dis)+'.chk', 'scf')


mf.__dict__.update(scf_result_dic)
# print("Hartree-Fock energy          : %s"%mf_load.e_tot)
print("E hf before: ",mf.e_tot)
mf.kernel(mf.make_rdm1())
assert mf.converged
print("E hf after: ",mf.e_tot)        


# CCSD
if do_ccsd:
    
    # cc.diis_space = 10
    # mf = scf.addons.remove_linear_dep_(mf).run()
    # mycc = cc.CCSD(mf).run()
    cc = cup_clus.KUCCSD(mf)
    cc.kernel()

# Vayesta options
use_sym = False
nfrag = 1 
bath_type = "dmet"
bath_opts = dict(bathtype=bath_type)   
# bath_type = "mp2"
# threshold_occ=0
# threshold_vir=1
# bath_opts = dict(bathtype=bath_type,threshold_occ=threshold_occ,threshold_vir=threshold_vir,truncation="number") 
ncells = np.prod(kmesh)

emb_fci_int = vayesta.ewf.EWF(mf, solver="FCI", bath_options=bath_opts, solver_options=dict(conv_tol=1.e-14))
# Set up fragments
with emb_fci_int.iao_fragmentation() as f:
    f.add_atomic_fragment([0], orbital_filter=["Ni 3s","Ni 3p"])#,"Ni 3dx2-y2","Ni 3dz^2"])
    # f.add_atomic_fragment([0], orbital_filter=["Ni 3d"])
    f.add_atomic_fragment([0], orbital_filter=["Ni 3dxz","Ni 3dxy","Ni 3dyz"])
    f.add_atomic_fragment([0], orbital_filter=["Ni 3dx2-y2","Ni 3dz^2"])
    f.add_atomic_fragment([0], orbital_filter=["Ni 4s","Ni 4p"])
    # f.add_atomic_fragment([1], orbital_filter=["O 2s"])
    # f.add_atomic_fragment([1], orbital_filter=["O 2p", "O 3s"])
    f.add_atomic_fragment([1])

    f.add_atomic_fragment([2], orbital_filter=["Ni 3s","Ni 3p"])#,"Ni 3dx2-y2","Ni 3dz^2"])
    # f.add_atomic_fragment([2], orbital_filter=["Ni 3d"])
    # f.add_atomic_fragment([2], orbital_filter=["Ni 3dxz","Ni 3dxy","Ni 3dyz","Ni 3dx2-y2","Ni 3dz^2"])
    f.add_atomic_fragment([2], orbital_filter=["Ni 3dxz","Ni 3dxy","Ni 3dyz"])
    f.add_atomic_fragment([2], orbital_filter=["Ni 3dx2-y2","Ni 3dz^2"])
    f.add_atomic_fragment([2], orbital_filter=["Ni 4s","Ni 4p"])
    # f.add_atomic_fragment([3], orbital_filter=["O 2s"])
    # f.add_atomic_fragment([3], orbital_filter=["O 2p", "O 3s"])
    f.add_atomic_fragment([3])
    # f.add_all_atomic_fragments()
emb_fci_int.kernel()




emb_fci = vayesta.ewf.EWF(mf, solver="CALLBACK", bath_options=bath_opts, solver_options=dict(callback=fci_solver))
# Set up fragments
with emb_fci.iao_fragmentation() as f:
    f.add_atomic_fragment([0], orbital_filter=["Ni 3s","Ni 3p"])#,"Ni 3dx2-y2","Ni 3dz^2"])
    # f.add_atomic_fragment([0], orbital_filter=["Ni 3d"])
    f.add_atomic_fragment([0], orbital_filter=["Ni 3dxz","Ni 3dxy","Ni 3dyz"])
    f.add_atomic_fragment([0], orbital_filter=["Ni 3dx2-y2","Ni 3dz^2"])
    f.add_atomic_fragment([0], orbital_filter=["Ni 4s","Ni 4p"])
    # f.add_atomic_fragment([1], orbital_filter=["O 2s"])
    # f.add_atomic_fragment([1], orbital_filter=["O 2p", "O 3s"])
    f.add_atomic_fragment([1])

    f.add_atomic_fragment([2], orbital_filter=["Ni 3s","Ni 3p"])#,"Ni 3dx2-y2","Ni 3dz^2"])
    # f.add_atomic_fragment([2], orbital_filter=["Ni 3d"])
    # f.add_atomic_fragment([2], orbital_filter=["Ni 3dxz","Ni 3dxy","Ni 3dyz","Ni 3dx2-y2","Ni 3dz^2"])
    f.add_atomic_fragment([2], orbital_filter=["Ni 3dxz","Ni 3dxy","Ni 3dyz"])
    f.add_atomic_fragment([2], orbital_filter=["Ni 3dx2-y2","Ni 3dz^2"])
    f.add_atomic_fragment([2], orbital_filter=["Ni 4s","Ni 4p"])
    # f.add_atomic_fragment([3], orbital_filter=["O 2s"])
    # f.add_atomic_fragment([3], orbital_filter=["O 2p", "O 3s"])
    f.add_atomic_fragment([3])
    # f.add_all_atomic_fragments()
emb_fci.kernel()

print("fci corr energy = ",emb_fci.e_corr)
corr_mf = emb_fci.get_corrfunc_mf("Sz,Sz")
corr_emb_fci = emb_fci.get_corrfunc("Sz,Sz")

emb = vayesta.ewf.EWF(mf, solver="CALLBACK", bath_options=bath_opts, solver_options=dict(callback=solver_cs))
# Set up fragments
with emb.iao_fragmentation() as f:
    f.add_atomic_fragment([0], orbital_filter=["Ni 3s","Ni 3p"])#,"Ni 3dx2-y2","Ni 3dz^2"])
    # f.add_atomic_fragment([0], orbital_filter=["Ni 3d"])
    f.add_atomic_fragment([0], orbital_filter=["Ni 3dxz","Ni 3dxy","Ni 3dyz"])
    f.add_atomic_fragment([0], orbital_filter=["Ni 3dx2-y2","Ni 3dz^2"])
    f.add_atomic_fragment([0], orbital_filter=["Ni 4s","Ni 4p"])
    # f.add_atomic_fragment([1], orbital_filter=["O 2s"])
    # f.add_atomic_fragment([1], orbital_filter=["O 2p", "O 3s"])
    f.add_atomic_fragment([1])

    f.add_atomic_fragment([2], orbital_filter=["Ni 3s","Ni 3p"])#,"Ni 3dx2-y2","Ni 3dz^2"])
    # f.add_atomic_fragment([2], orbital_filter=["Ni 3d"])
    # f.add_atomic_fragment([2], orbital_filter=["Ni 3dxz","Ni 3dxy","Ni 3dyz","Ni 3dx2-y2","Ni 3dz^2"])
    f.add_atomic_fragment([2], orbital_filter=["Ni 3dxz","Ni 3dxy","Ni 3dyz"])
    f.add_atomic_fragment([2], orbital_filter=["Ni 3dx2-y2","Ni 3dz^2"])
    f.add_atomic_fragment([2], orbital_filter=["Ni 4s","Ni 4p"])
    # f.add_atomic_fragment([3], orbital_filter=["O 2s"])
    # f.add_atomic_fragment([3], orbital_filter=["O 2p", "O 3s"])
    f.add_atomic_fragment([3])
    # f.add_all_atomic_fragments()
emb.kernel()


print("Hartree-Fock energy          : %s"%mf.e_tot)
# print("CISD Energy                  : %s"%cisd.e_tot)
if do_ccsd:
    print("CCSD Energy                  : %s"%cc.e_tot)
print("Embedded FCI energy: ",emb_fci.e_tot)



corr_mf = emb_fci.get_corrfunc_mf("Sz,Sz")
corr_emb_fci = emb_fci.get_corrfunc("Sz,Sz")
corr_emb_fci_int = emb_fci_int.get_corrfunc("Sz,Sz")
corr_emb = emb.get_corrfunc("Sz,Sz")


print("corr mf: \n",corr_mf)
print("\ncorr cc: \n",corr_emb_fci)
for a in range(cell.natm):
    for b in range(cell.natm):
        print("A= %d, B= %d:  HF= %+.5f  CC= %+.5f" % (a, b, corr_mf[a, b], corr_emb_fci[a, b]))
        # print("A= %d, B= %d:  HF= %+.5f" % (a, b, corr_mf[a, b]))
# print("Total:       HF= %+.5f  CC= %+.5f" % (corr_mf.sum(), corr_emb_fci.sum()))

projectors = emb._get_atom_projectors()[2]


resultsfolder = "/home/connorlenihan/Documents/Quantum_cluster/NiO_results/"
f_name=resultsfolder+"NiO_lat_con_"+str(atomic_dis)+"_nshots_"+str(Nshots)+"_k_"+str(kmesh[0])+str(kmesh[1])+str(kmesh[2])+"_basis_"+str(cell.basis)+"_"+bath_type+"_bath"+"magentics_"+magnetism+"_ccsd_"+str(do_ccsd)+"_fci_solver_loaded_mf.dat"
if do_ccsd:
    if os.path.isfile(f_name):
        f = open(f_name, "a")
        # f.write(str(mf.e_tot)+"    "+str(cc.e_tot)+"    "+str(fci.e_tot)+"    "+str(emb_atom_fci.e_tot)+"    "+str(emb_fci.e_tot)+"    "+str(emb.e_tot)+"\n")
        f.write(str(mf.e_tot)+"    "+str(cc.e_tot)+"    "+str(emb_fci_int.e_tot)+"    "+str(emb_fci.e_tot)+"    "+str(emb.e_tot)+"    "+str(corr_mf[0,2])+"    "+str(spin_spin_correlation(cc.make_rdm1(),cc.make_rdm2(),projectors[0],projectors[2]))+"    "+str(corr_emb_fci_int[0,2])+"    "+str(corr_emb_fci[0,2])+"    "+str(corr_emb[0,2])+"\n")
        f.close()
    else:
        f = open(f_name, "a")
        # f.write("c0"+"    "+"e_fci"+"    "+"e_hf"+"    "+"vqe_energy"+"    "+"HF+CS"+"    "+"HF+meas_diag"+"    "+"E_pauli_all"+"\n")
        # f.write(str(c0)+"    "+str(e_fci)+"    "+str(myhf.e_tot)+"    "+str(vqe_energy)+"    "+str(e_proj_CS)+"    "+str(e_proj_meas_diag)+"    "+str(result_Pauli_all_comm)+"\n")
        f.write("E_HF"+"    "+"E_CCSD"+"    "+"E_int_FCI_emb"+"    "+"E_FCI_emb"+"    "+"E_CS_emb"+"    "+"Sz_HF"+"    "+"Sz_CCSD"+"    "+"Sz_int_FCI_emb"+"    "+"Sz_FCI_emb"+"    "+"Sz_CS_emb"+"\n")
        f.write(str(mf.e_tot)+"    "+str(cc.e_tot)+"    "+str(emb_fci_int.e_tot)+"    "+str(emb_fci.e_tot)+"    "+str(emb.e_tot)+"    "+str(corr_mf[0,2])+"    "+str(spin_spin_correlation(cc.make_rdm1(),cc.make_rdm2(),projectors[0],projectors[2]))+"    "+str(corr_emb_fci_int[0,2])+"    "+str(corr_emb_fci[0,2])+"    "+str(corr_emb[0,2])+"\n")
        f.close()  

else:
    if os.path.isfile(f_name):
        f = open(f_name, "a")
        # f.write(str(mf.e_tot)+"    "+str(cc.e_tot)+"    "+str(fci.e_tot)+"    "+str(emb_atom_fci.e_tot)+"    "+str(emb_fci.e_tot)+"    "+str(emb.e_tot)+"\n")
        f.write(str(mf.e_tot)+"    "+str(emb_fci_int.e_tot)+"    "+str(emb_fci.e_tot)+"    "+str(emb.e_tot)+"    "+str(corr_mf[0,2])+"    "+str(corr_emb_fci_int[0,2])+"    "+str(corr_emb_fci[0,2])+"    "+str(corr_emb[0,2])+"\n")
        f.close()
    else:
        f = open(f_name, "a")
        # f.write("c0"+"    "+"e_fci"+"    "+"e_hf"+"    "+"vqe_energy"+"    "+"HF+CS"+"    "+"HF+meas_diag"+"    "+"E_pauli_all"+"\n")
        # f.write(str(c0)+"    "+str(e_fci)+"    "+str(myhf.e_tot)+"    "+str(vqe_energy)+"    "+str(e_proj_CS)+"    "+str(e_proj_meas_diag)+"    "+str(result_Pauli_all_comm)+"\n")
        f.write("E_HF"+"    "+"E_int_FCI_emb"+"    "+"E_FCI_emb"+"    "+"E_CS_emb"+"    "+"Sz_HF"+"    "+"Sz_int_FCI_emb"+"    "+"Sz_FCI_emb"+"    "+"Sz_CS_emb"+"\n")
        f.write(str(mf.e_tot)+"    "+str(emb_fci_int.e_tot)+"    "+str(emb_fci.e_tot)+"    "+str(emb.e_tot)+"    "+str(corr_mf[0,2])+"    "+str(corr_emb_fci_int[0,2])+"    "+str(corr_emb_fci[0,2])+"    "+str(corr_emb[0,2])+"\n")
        f.close()        

      

        # print("Hartree-Fock energy          : %s"%mf.e_tot)
        # # print("CISD Energy                  : %s"%cisd.e_tot)
        # if do_ccsd:
        #     print("CCSD Energy                  : %s"%cc.e_tot)
    
