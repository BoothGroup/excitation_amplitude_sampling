###data collection for embedded BN system using a classically optimised VQE solver

import os
import sys
import numpy as np
import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.fci
import vayesta
import vayesta.ewf
from vayesta.misc.molecules import ring
from vayesta.core.types.wf.t_to_c import t1_rhf, t2_rhf
import pyscf.pbc.cc
from src import cluster_solver
from pyscf import ao2mo
import pyscf.pbc.dft

from qiskit_nature.second_q.operators import ElectronicIntegrals, FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy

# User defined FCI solver - takes pyscf mf as input and returns RDMs
# The mf argment contains the hamiltonain in the orthonormal cluster basis
# Pyscf or other solvers may be used to solve the cluster problem and may return RDMs, CISD amplitudes or CCSD amplitudes
def solver(mf):

    # solver=cluster_solver.cluster_solver(cluster.h1e, cluster.h2e, cluster.norb, nelec=(cluster.nocc,cluster.nocc))
    # solver.solve(random_initial_params=False,attempts=4)
    # energy_cs, ci_vec_cs=solver.measure_classical_shadows(Nshots,3,ensemble="Clifford")
    ci = pyscf.ci.CISD(mf)
    cc = pyscf.cc.CCSD(mf)
    cc.kernel()
    #energy, civec = pyscf.fci.direct_spin0.kernel(mf.h1e, mf.h2e, mf.norb, nelec=(mf.nocc, mf.nocc), conv_tol=1.e-16)
    energy, civec = ci.kernel()
    c0, c1, c2 = ci.cisdvec_to_amplitudes(civec)

    # To use CI amplitudues use return the following line and set energy_functional='wf' to use the projected energy in the EWF arguments below
    return dict(c0=c0, c1=c1, c2=c2, converged=True, energy=ci.e_corr)

    # Convert CISD amplitudes to CCSD amplitudes to be able to make use of the patitioned cumulant energy functional
    # t1 = t1_rhf(c1/c0) 
    # t2 = t2_rhf(t1, c2/c0)
    # return dict(t1=t1, t2=t2, l1=t1, l2=t2, converged=True, energy=ci.e_corr)
Nshots=int(float(sys.argv[1]))
do_ccsd = bool(int(sys.argv[5]))
def solver_cs(mf):
    
    nelec = mf.mol.nelectron 
    # mol = mol  
    print("nelec: ",nelec)

    # Organising mean field outputs

    mo_energy = mf.mo_energy
    mo_coeff = mf.mo_coeff  # (ao = spatical oribital |MO) due to restricted hartree fock calculation, so no need to consider alpha/beta contributions
                                    # to consider spin orbitals as required in UHF and GHF. 
    mo_occ = mf.mo_occ
    
    
    num_orbitals = mo_coeff.shape[0]
    num_spin_orbitals = num_orbitals * 2 
    num_particles = nelec 
    nuclear_repulsion_energy = mf.energy_nuc()

    # one body integral (\psi_{mo}| core hamiltonian (in AO basis )| \psi_{mo})
    one_body = mo_coeff.T @ mf.get_hcore() @ mo_coeff
    # print("h core: ",mf.get_hcore())  
    # print("mol: ",ao2mo.kernel(mf.mol, mo_coeff))
    # ci = pyscf.cc.CCSD(mf)
    # myhf = mol.RHF().run()
    h1e = mf.get_hcore()
    # Need to convert cderis into standard 4-index tensor when using denisty fitting for the mean-field
    cderi = mf.with_df._cderi
    cderi = pyscf.lib.unpack_tril(cderi)
    h2e = np.einsum('Lpq,Lrs->pqrs', cderi, cderi)
    norb = mf.mo_coeff.shape[-1]
    nelec = mf.mol.nelec
    # print("_eri: ",ci.ao2mo(mo_coeff))



    # eri = ao2mo.kernel(ci.ao2mo(mo_coeff), (mo_coeff, mo_coeff,mo_coeff,mo_coeff), compact=False)
    # two_body = eri.reshape((mo_coeff.shape[-1],) * 4)
    # print("h1e = ",h1e)
    # print();()
    # print("h2e = ",h2e)


    solver=cluster_solver.cluster_solver(h1e, h2e, norb, nelec,mf.get_ovlp(),"RHF")
    solver.solve(random_initial_params=False,attempts=4)
    energy_cs, ci_vec_cs=solver.measure_classical_shadows_perp(Nshots,3,ensemble="Clifford")
    c0_cs=ci_vec_cs[0]
    c1_cs=ci_vec_cs[1]/c0_cs
    c2_cs=ci_vec_cs[2]/c0_cs
    c0_cs=1

    # To use CI amplitudues use return the following line and set energy_functional='wf' to use the projected energy in the EWF arguments below
    return dict(c0=c0_cs, c1=c1_cs, c2=c2_cs, converged=True, energy=energy_cs)


# a = 3.615
for a in np.arange(float(sys.argv[2]),float(sys.argv[3]),float(sys.argv[4])):
    cell = pyscf.pbc.gto.Cell()
    cell.atom = "B 0 0 0 ; N %f %f %f" % (a / 4, a / 4, a / 4)
    cell.a = np.asarray([[a / 2, a / 2, 0], [0, a / 2, a / 2], [a / 2, 0, a / 2]])
    cell.basis = 'gth-dzvp'#"sto-6g"#
    cell.pseudo = 'gth-pade'
    cell.output = "pyscf_BN_gth.out"
    cell.verbose = 5
    cell.exp_to_discard = 0.1
    cell.build()

    #lda = pyscf.pbc.dft.RKS(cell)
    #lda = lda.density_fit()
    #lda.xc = "svwn"
    #lda.kernel()
    #assert(lda.converged)
    # print(lda.e)

    # Hartree-Fock with k-points
    # kmesh = [3,3,3]
    # kpts = cell.make_kpts(kmesh)
    # mf = pyscf.pbc.scf.KRHF(cell, kpts)
    # mf = mf.density_fit()
    # mf.kernel()
    # assert(mf.converged)
    # print("Hartree-Fock energy          : %s"%mf.e_tot)


    kmesh = [3,3,3]
    kpts = cell.make_kpts(kmesh)
    mf = pyscf.pbc.scf.KRHF(cell, kpts)
    mf = mf.density_fit()
    mf.kernel()
    assert(mf.converged)
    print("Hartree-Fock energy          : %s"%mf.e_tot)

    

    # CISD
    # cisd = pyscf.ci.CISD(mf)
    # cisd.kernel()

    # CCSD
    if do_ccsd:
        cc = pyscf.pbc.cc.KCCSD(mf)
        # cc.diis_space = 10
        cc.kernel()
        assert cc.converged

    # FCI
    # fci = pyscf.fci.FCI(mf)
    # fci.kernel()

    # Vayesta options
    use_sym = False
    nfrag = 1 
    bath_type = "dmet"
    bath_opts = dict(bathtype=bath_type)   
    # bath_type = "mp2"
    # thresh = float(sys.argv[6])
    # bath_opts = dict(bathtype=bath_type,threshold = thresh)   
    ncells = np.product(kmesh)
    # Run vayesta with user defined solver
    emb = vayesta.ewf.EWF(mf, solver="CALLBACK",  energy_functional='wf', bath_options=bath_opts, solver_options=dict(callback=solver_cs))
    # Set up fragments
    with emb.iao_fragmentation() as f:
        # Fragment containing the 1s state of O and 1s and 2s states of Se
        # f.add_atomic_fragment([0], orbital_filter=["B 1s", "B 2s"],sym_factor=ncells)
        # f.add_atomic_fragment([1], orbital_filter=["N 1s", "N 2s"],sym_factor=ncells)
        # f.add_atomic_fragment([0], orbital_filter=["2p"],sym_factor=ncells)
        # f.add_atomic_fragment([1], orbital_filter=["2p"],sym_factor=ncells)

        f.add_atomic_fragment([0], orbital_filter=["B 1s", "B 2s"])
        f.add_atomic_fragment([1], orbital_filter=["N 1s", "N 2s"])
        f.add_atomic_fragment([0], orbital_filter=["2p"])
        f.add_atomic_fragment([1], orbital_filter=["2p"])
        

        # # # Atoms can be specified by labels or indices
        # # # Fragment containing the 2s state at O and 3s and 4s states of Se
        
        # f.add_all_atomic_fragments()
    emb.kernel()

    emb_fci = vayesta.ewf.EWF(mf, solver="FCI", bath_options=bath_opts, solver_options=dict(conv_tol=1.e-14))
    # Set up fragments
    with emb_fci.iao_fragmentation() as f:
        f.add_atomic_fragment([0], orbital_filter=["B 1s", "B 2s"])
        f.add_atomic_fragment([1], orbital_filter=["N 1s", "N 2s"])
        f.add_atomic_fragment([0], orbital_filter=["2p"])
        f.add_atomic_fragment([1], orbital_filter=["2p"])
    emb_fci.kernel()
    fci_dmet_energy = emb_fci.get_dmet_energy(part_cumulant=False, approx_cumulant=False)
    fci_cumul_energy = emb_fci.get_dmet_energy(part_cumulant=True, approx_cumulant=False)

    emb_atom_fci = vayesta.ewf.EWF(mf, solver="FCI", bath_options=bath_opts, solver_options=dict(conv_tol=1.e-14))
    # Set up fragments
    with emb_atom_fci.iao_fragmentation() as f:
        f.add_all_atomic_fragments()
    emb_atom_fci.kernel()

    print("Hartree-Fock energy          : %s"%mf.e_tot)
    # print("CISD Energy                  : %s"%cisd.e_tot)
    if do_ccsd:
        print("CCSD Energy                  : %s"%cc.e_tot)
    # print("FCI  Energy                  : %s"%fci.e_tot)
    print("Emb. Partitioned Cumulant    : %s"%emb.e_tot)
    if bath_type == "dmet":
        if do_ccsd:
            f_name="BN_w_dem_part_lat_con_"+str(a)+"_nshots_"+str(Nshots)+"_k_"+str(kmesh[0])+str(kmesh[1])+str(kmesh[2])+"_basis_"+str(cell.basis)+"_"+bath_type+"_bath.dat"
            if os.path.isfile(f_name):
                f = open(f_name, "a")
                # f.write(str(mf.e_tot)+"    "+str(cc.e_tot)+"    "+str(fci.e_tot)+"    "+str(emb_atom_fci.e_tot)+"    "+str(emb_fci.e_tot)+"    "+str(emb.e_tot)+"\n")
                f.write(str(mf.e_tot)+"    "+str(cc.e_tot)+"    "+str(emb_atom_fci.e_tot)+"    "+str(emb_fci.e_tot)+"    "+str(emb.e_tot)+"\n")
                f.close()
            else:
                f = open(f_name, "a")
                # f.write("c0"+"    "+"e_fci"+"    "+"e_hf"+"    "+"vqe_energy"+"    "+"HF+CS"+"    "+"HF+meas_diag"+"    "+"E_pauli_all"+"\n")
                # f.write(str(c0)+"    "+str(e_fci)+"    "+str(myhf.e_tot)+"    "+str(vqe_energy)+"    "+str(e_proj_CS)+"    "+str(e_proj_meas_diag)+"    "+str(result_Pauli_all_comm)+"\n")
                f.write("HF"+"    "+"CCSD"+"    "+"FCI_atom"+"    "+"FCI_atom_dem_part"+"    "+"FCI_atom_cumul"+"    "+"FCI_emb"+"    "+"FCI_dem_part"+"    "+"FCI_cumul"+"    "+"VQE_emb"+"\n")
                f.write(str(mf.e_tot)+"    "+str(cc.e_tot)+"    "+str(emb_atom_fci.e_tot)+"    "+str(emb_atom_fci.get_dmet_energy(part_cumulant=False, approx_cumulant=False))+"    "+str(emb_atom_fci.get_dmet_energy(part_cumulant=True, approx_cumulant=False))+"    "+str(emb_fci.e_tot)+"    "+str(fci_dmet_energy)+"    "+str(fci_cumul_energy)+"    "+str(emb.e_tot)+"\n")
                f.close()
        else:
            f_name="BN_w_dem_part_lat_con_"+str(a)+"_nshots_"+str(Nshots)+"_k_"+str(kmesh[0])+str(kmesh[1])+str(kmesh[2])+"_basis_"+str(cell.basis)+"_"+bath_type+"_bath_no_ccsd.dat"
            if os.path.isfile(f_name):
                f = open(f_name, "a")
                # f.write(str(mf.e_tot)+"    "+str(cc.e_tot)+"    "+str(fci.e_tot)+"    "+str(emb_atom_fci.e_tot)+"    "+str(emb_fci.e_tot)+"    "+str(emb.e_tot)+"\n")
                f.write(str(mf.e_tot)+"    "+str(emb_atom_fci.e_tot)+"    "+str(emb_fci.e_tot)+"    "+str(emb.e_tot)+"\n")
                f.close()
            else:
                f = open(f_name, "a")
                # f.write("c0"+"    "+"e_fci"+"    "+"e_hf"+"    "+"vqe_energy"+"    "+"HF+CS"+"    "+"HF+meas_diag"+"    "+"E_pauli_all"+"\n")
                # f.write(str(c0)+"    "+str(e_fci)+"    "+str(myhf.e_tot)+"    "+str(vqe_energy)+"    "+str(e_proj_CS)+"    "+str(e_proj_meas_diag)+"    "+str(result_Pauli_all_comm)+"\n")
                f.write("HF"+"    "+"FCI_atom"+"    "+"FCI_atom_dem_part"+"    "+"FCI_atom_cumul"+"    "+"FCI_emb"+"    "+"FCI_dem_part"+"    "+"FCI_cumul"+"    "+"VQE_emb"+"\n")
                f.write(str(mf.e_tot)+"    "+str(emb_atom_fci.e_tot)+"    "+str(emb_atom_fci.get_dmet_energy(part_cumulant=False, approx_cumulant=False))+"    "+str(emb_atom_fci.get_dmet_energy(part_cumulant=True, approx_cumulant=False))+"    "+str(emb_fci.e_tot)+"    "+str(fci_dmet_energy)+"    "+str(fci_cumul_energy)+"    "+str(emb.e_tot)+"\n")
                f.close()
    elif bath_type == "mp2":
        if do_ccsd:
            f_name="BN_w_dem_part_lat_con_"+str(a)+"_nshots_"+str(Nshots)+"_k_"+str(kmesh[0])+str(kmesh[1])+str(kmesh[2])+"_basis_"+str(cell.basis)+"_"+bath_type+"_bath_"+str(thresh)+".dat"
            if os.path.isfile(f_name):
                f = open(f_name, "a")
                # f.write(str(mf.e_tot)+"    "+str(cc.e_tot)+"    "+str(fci.e_tot)+"    "+str(emb_atom_fci.e_tot)+"    "+str(emb_fci.e_tot)+"    "+str(emb.e_tot)+"\n")
                f.write(str(mf.e_tot)+"    "+str(cc.e_tot)+"    "+str(emb_atom_fci.e_tot)+"    "+str(emb_fci.e_tot)+"    "+str(emb.e_tot)+"\n")
                f.close()
            else:
                f = open(f_name, "a")
                # f.write("c0"+"    "+"e_fci"+"    "+"e_hf"+"    "+"vqe_energy"+"    "+"HF+CS"+"    "+"HF+meas_diag"+"    "+"E_pauli_all"+"\n")
                # f.write(str(c0)+"    "+str(e_fci)+"    "+str(myhf.e_tot)+"    "+str(vqe_energy)+"    "+str(e_proj_CS)+"    "+str(e_proj_meas_diag)+"    "+str(result_Pauli_all_comm)+"\n")
                f.write("HF"+"    "+"CCSD"+"    "+"FCI_atom"+"    "+"FCI_atom_dem_part"+"    "+"FCI_atom_cumul"+"    "+"FCI_emb"+"    "+"FCI_dem_part"+"    "+"FCI_cumul"+"    "+"VQE_emb"+"\n")
                f.write(str(mf.e_tot)+"    "+str(cc.e_tot)+"    "+str(emb_atom_fci.e_tot)+"    "+str(emb_atom_fci.get_dmet_energy(part_cumulant=False, approx_cumulant=False))+"    "+str(emb_atom_fci.get_dmet_energy(part_cumulant=True, approx_cumulant=False))+"    "+str(emb_fci.e_tot)+"    "+str(fci_dmet_energy)+"    "+str(fci_cumul_energy)+"    "+str(emb.e_tot)+"\n")
                f.close()
        else:
            f_name="BN_w_dem_part_lat_con_"+str(a)+"_nshots_"+str(Nshots)+"_k_"+str(kmesh[0])+str(kmesh[1])+str(kmesh[2])+"_basis_"+str(cell.basis)+"_"+bath_type+"_bath_"+str(thresh)+"_no_ccsd.dat"
            if os.path.isfile(f_name):
                f = open(f_name, "a")
                # f.write(str(mf.e_tot)+"    "+str(cc.e_tot)+"    "+str(fci.e_tot)+"    "+str(emb_atom_fci.e_tot)+"    "+str(emb_fci.e_tot)+"    "+str(emb.e_tot)+"\n")
                f.write(str(mf.e_tot)+"    "+str(emb_atom_fci.e_tot)+"    "+str(emb_fci.e_tot)+"    "+str(emb.e_tot)+"\n")
                f.close()
            else:
                f = open(f_name, "a")
                # f.write("c0"+"    "+"e_fci"+"    "+"e_hf"+"    "+"vqe_energy"+"    "+"HF+CS"+"    "+"HF+meas_diag"+"    "+"E_pauli_all"+"\n")
                # f.write(str(c0)+"    "+str(e_fci)+"    "+str(myhf.e_tot)+"    "+str(vqe_energy)+"    "+str(e_proj_CS)+"    "+str(e_proj_meas_diag)+"    "+str(result_Pauli_all_comm)+"\n")
                f.write("HF"+"    "+"FCI_atom"+"    "+"FCI_atom_dem_part"+"    "+"FCI_atom_cumul"+"    "+"FCI_emb"+"    "+"FCI_dem_part"+"    "+"FCI_cumul"+"    "+"VQE_emb"+"\n")
                f.write(str(mf.e_tot)+"    "+str(emb_atom_fci.e_tot)+"    "+str(emb_atom_fci.get_dmet_energy(part_cumulant=False, approx_cumulant=False))+"    "+str(emb_atom_fci.get_dmet_energy(part_cumulant=True, approx_cumulant=False))+"    "+str(emb_fci.e_tot)+"    "+str(fci_dmet_energy)+"    "+str(fci_cumul_energy)+"    "+str(emb.e_tot)+"\n")
                f.close()
