### data collection for embedded NiO UHF calculation

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

import sys, os
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

Nshots=int(float(sys.argv[1]))
do_ccsd = bool(int(sys.argv[5]))

# User defined FCI solver - takes pyscf mf as input and returns RDMs
# The mf argment contains the hamiltonain in the orthonormal cluster basis
# Pyscf or other solvers may be used to solve the cluster problem and may return RDMs, CISD amplitudes or CCSD amplitudes
def cisd_solver(mf):
    ci = UCISD(mf)
    energy, civec = ci.kernel()
    c0, c1, c2 = ci.cisdvec_to_amplitudes(civec)

    # To use CI amplitudues use return the following line and set energy_functional='wf' to use the projected energy in the EWF arguments below
    # return dict(c0=c0, c1=c1, c2=c2, converged=True, energy=ci.e_corr)

    # Convert CISD amplitudes to CCSD amplitudes to be able to make use of the patitioned cumulant energy functional
    t1 = t1_uhf(c1/c0) 
    t2 = t2_uhf(t1, c2/c0)
    return dict(t1=t1, t2=t2, l1=t1, l2=t2, converged=True, energy=ci.e_corr)

# def ccsd_solver(mf, dm=False):
#     cc = UCCSD(mf)
#     cc.kernel()
#     t1, t2 = cc.t1, cc.t2
#     l1, l2 = cc.solve_lambda()
#     return dict(t1=t1, t2=t2, l1=l1, l2=l2, converged=True, energy=cc.e_corr)

def ccsd_solver(mf, dm=False):
    cc = UCCSD(mf)
    cc.kernel()
    t1, t2 = cc.t1, cc.t2
    l1, l2 = cc.solve_lambda()
    

    c0_cc=1.0
    c1_cc = c0_cc * t1
    c2_cc = c0_cc * (t2 + np.einsum("ia,jb->ijab", t1, t1))
    return dict(c0=c0_cc, c1=c1_cc, c2=c2_cc, converged=cc.converged, energy=cc.e_corr)

def fci_solver(mf, dm=False):
    h1e = mf.get_hcore()
    h2e = mf._eri
    # print("h1e",h1e)
    # print("h2e",h2e)
    norb = h1e[0].shape[-1]
    print("norb:",norb)
    print("mf get_occ: ",mf.get_occ())
    print("mf get_occ: ",mf.mo_occ)
    print("mf energies: ",mf.mo_energy)
    nelec = mf.mol.nelec
    energy, civec = pyscf.fci.direct_uhf.kernel(h1e, h2e, norb, nelec, conv_tol=1.e-14)
    print("c0: ",civec[0][0])
    return dict(civec=civec, converged=True, energy=energy)


def solver_cs(mf):
    # Nshots=1000
    # nelec = mf.mol.nelectron 

    # Organising mean field outputs

    # mo_energy = mf.mo_energy
    # mo_coeff = mf.mo_coeff  # (ao = spatical oribital |MO) due to restricted hartree fock calculation, so no need to consider alpha/beta contributions
    #                                 # to consider spin orbitals as required in UHF and GHF. 
    # mo_occ = mf.mo_occ
    
    
    norb = (mf.mo_coeff[0].shape[1], mf.mo_coeff[1].shape[1])
    print("norb: ",norb)
    # nelec = mf.mol.nelectron
    print("mf get_occ: ",mf.get_occ())
    print("mf get_occ: ",mf.mo_occ)
    print("mf energies: ",mf.mo_energy)
    nocc = (int(sum(mf.get_occ()[0])),int(sum(mf.get_occ()[1])))
    nvir = (norb[0] - nocc[0],norb[1] - nocc[1])

    h1e = mf.get_hcore()
    # Need to convert cderis into standard 4-index tensor when using denisty fitting for the mean-field
    # cderi = mf.with_df._cderi
    # cderi = pyscf.lib.unpack_tril(cderi)
    # h2e = np.einsum('Lpq,Lrs->pqrs', cderi, cderi)

    h1e = mf.get_hcore()
    h2e = mf._eri
    # print("h2e: ",h2e)
    # norb = mf.mo_coeff.shape[-1]
    nelec = mf.mol.nelec



    solver=cluster_solver.cluster_solver(h1e, h2e, norb, nelec,mf.get_ovlp(),"UHF",solver="FCI")
    solver.solve(random_initial_params=False,attempts=4)
    energy_cs, ci_vec_cs=solver.measure_classical_shadows_perp(Nshots,3,ensemble="Clifford")
    c0_cs=ci_vec_cs[0]
    c1_cs=ci_vec_cs[1]/c0_cs.real
    c2_cs=ci_vec_cs[2]/c0_cs.real
    c0_cs=1

    # To use CI amplitudues use return the following line and set energy_functional='wf' to use the projected energy in the EWF arguments below
    return dict(c0=c0_cs, c1=c1_cs, c2=c2_cs, converged=True, energy=energy_cs)







def get_mf_init_guess(mf, kpts, magnetism):
    dm0 = mf.get_init_guess()
    if magnetism == "PM":
        return dm0
    if magnetism == "FM":
        return dm0
    elif magnetism[:2] == "AF":

        def init_afm(dm0):
            dm0 = dm0.copy()
            # 3d-orbitals
            atoms_a = [0]
            atoms_b = [2]
            atoms = mf.mol.elements[0]  # metal atoms
            da = mf.mol.search_ao_label(["^%d %s 3d" % (i, atoms) for i in atoms_a])
            db = mf.mol.search_ao_label(["^%d %s 3d" % (i, atoms) for i in atoms_b])
            a_sites = mf.mol.search_ao_label(["^%d %s" % (i, atoms) for i in atoms_a])
            b_sites = mf.mol.search_ao_label(["^%d %s" % (i, atoms) for i in atoms_b])
            a_sites_2d = np.ix_(a_sites, a_sites)
            b_sites_2d = np.ix_(b_sites, b_sites)

            if fm_to_afm:
                for k in range(len(kpts)):
                    # Flip spins on B sites:
                    dma = dm0[0][k][a_sites_2d].copy()
                    dmb = dm0[1][k][a_sites_2d].copy()
                    dm0[0][k][b_sites_2d] = dmb
                    dm0[1][k][b_sites_2d] = dma

            elif local_spin is None:
                delta = 0.1

                if kpts is None:
                    da = np.ix_(da, da)
                    db = np.ix_(db, db)
                else:
                    da = np.ix_(list(range(len(kpts))), da, da)
                    db = np.ix_(list(range(len(kpts))), db, db)

                ddm = delta * dm0[0][da]
                # Sites A -> Majority spin=a
                dm0[0][da] += ddm
                dm0[1][da] -= ddm
                # Sites B -> Majority spin=b
                ddm = delta * dm0[1][db]
                dm0[0][db] -= ddm
                dm0[1][db] += ddm
            else:
                ovlp = mf.get_ovlp()
                fprint = lambda *args: print(*args, flush=True)
                fprint("Preparing AF initial guess:")
                for k in range(len(kpts)):
                    fprint("k-point %d" % k)
                    da2 = np.ix_(da, da)
                    db2 = np.ix_(db, db)
                    nAa = np.sum(dm0[0][k][da2] * ovlp[k][da2])
                    nAb = np.sum(dm0[1][k][da2] * ovlp[k][da2])
                    sA = np.real(nAa - nAb)
                    nA = np.real(nAa + nAb)
                    factor = (local_spin - sA) / nA
                    fprint(
                        "nAa= %f %f nAb= %f %f sA= %f nA= %f factor= %f"
                        % (nAa.real, nAa.imag, nAb.real, nAb.imag, sA, nA, factor)
                    )
                    # A sites:
                    dm0[0][k][da2] *= 1 + factor
                    dm0[1][k][da2] *= 1 - factor
                    # B sites:
                    dm0[0][k][db2] *= 1 - factor
                    dm0[1][k][db2] *= 1 + factor

                    # Check
                    na = np.sum(dm0[0][k][da2] * ovlp[k][da2])
                    nb = np.sum(dm0[1][k][da2] * ovlp[k][da2])
                    s = na - nb
                    fprint(
                        "Result(A) real: na= %f nb= %f s= %f", na.real, nb.real, s.real
                    )
                    fprint(
                        "Result(A) imag: na= %f nb= %f s= %f", na.imag, nb.imag, s.imag
                    )
                    na = np.sum(dm0[0][k][db2] * ovlp[k][db2])
                    nb = np.sum(dm0[1][k][db2] * ovlp[k][db2])
                    s = na - nb
                    fprint(
                        "Result(B) real: na= %f nb= %f s= %f", na.real, nb.real, s.real
                    )
                    fprint(
                        "Result(B) imag: na= %f nb= %f s= %f", na.imag, nb.imag, s.imag
                    )

            return dm0

        if np.allclose(dm0[0], dm0[1], rtol=0, atol=1e-2):
            print("Seeding AFM")
            dm0 = init_afm(dm0)
        else:
            print("DM already spin-broken")
    else:
        raise NotImplementedError
    return dm0


kmeshes = [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]]
magnetics = ["AFM1","AFM2"]






# ############################################################################################
# def pop_analysis(emb, dm1, name):
#     emb.pop_analysis(dm1=dm1, filename="%s-lowdin.pop" % name, orbital_resolved=True)
#     emb.pop_analysis(
#         dm1=dm1,
#         filename="%s-mulliken.pop" % name,
#         local_orbitals="mulliken",
#         orbital_resolved=True,
#     )
#     emb.pop_analysis(
#         dm1=dm1,
#         filename="%s-iaopao.pop" % name,
#         local_orbitals="iao+pao",
#         orbital_resolved=True,
#     )




# corrected_ccs = list(prod(modes, projs, bnoetas, fragmentations))

resultsfolder = "/home/connorlenihan/Documents/Quantum_cluster/NiO_results"




from datetime import datetime

now = datetime.now() # current date and time

#4.1705
for atomic_dis in np.arange(float(sys.argv[2]),float(sys.argv[3]),float(sys.argv[4])):
    kmesh  = [1,1,1]
    for magnetism in magnetics:
        # magnetism = "AFM2"
        basis, pseudo = "gth-dzvp-molopt-sr", "gth-pade"


        kmeshstr = "".join(map(str, kmesh))



        unitcell = "primitive-af1" if "AFM1" in magnetism else "primitive-af2"

        cell = Cell()
        cell.a, cell.atom = solids.rocksalt(
            atoms=["Ni", "O"], a=atomic_dis, unitcell=unitcell
        )  # Oxide
        cell.basis = basis
        cell.pseudo = pseudo
        cell.output = f"{kmeshstr}_"+now.strftime("%d")+"_"+now.strftime("%m")+"_"+now.strftime("%Y")+"_"+now.strftime("%H")+"_"+now.strftime("%M")+"_"+now.strftime("%S")+"_pyscf.txt"
        cell.verbose = 5
        cell.spin = 4 * np.product(kmesh) if "HS" in magnetism else 0  # Oxide
        cell.exp_to_discard = 0.1
        cell.build()
        kpts = cell.make_kpts(kmesh, time_reversal_symmetry=False)

        local_spin = None
        fm_to_afm = False

        mf = pbcscf.KUHF(cell, kpts)
        mf = mf.density_fit()
        df = mf.with_df
        # mortyscratch = "/home/connorlenihan/Documents/Quantum_cluster/cderifiles"
        # cderifile = f"{mortyscratch}/NiO/CDERI_{basis}_{kmeshstr}_{magnetism}.h5"
        # if os.path.exists(cderifile):
        #     df._cderi = cderifile
        #     df.build()
        #     print(f"Reading CDERIs from file {cderifile}")
        # else:
        #    df._cderi_to_save = cderifile

        dm = get_mf_init_guess(mf, kpts, magnetism)
        mf.kernel(dm0=dm)
        mf.max_cycle = 250
        if not mf.converged:
            dm0 = mf.make_rdm1()
            mf = mf.newton()
            mf.kernel(dm0)

        counti = 0
        while True:
            mo1 = mf.stability()[0]
            stable = mo1 is mf.mo_coeff
            dm1 = mf.make_rdm1(mo_coeff=mo1)
            if stable:
                break
            counti += 1
            if counti > 0:
                print("Persistent instability after 0 attempts")
                print("\ncontinued nevertheless to coupled-cluster...")
                break
            mf.kernel(dm1)
            if not mf.converged:
                mf = mf.newton()
                mf.kernel(mf.make_rdm1())


        # assert mf.converged

        # with open("KUHF", "w") as fhf:
        #     fhf.write(f"{magnetism} ,  {kmeshstr} , {mf.e_tot}\n")
        # mf.analyze()
        assert(mf.converged)
        print("Hartree-Fock energy          : %s"%mf.e_tot)

        

        # CISD
        # cisd = pyscf.ci.CISD(mf)
        # cisd.kernel()

        # CCSD
        if do_ccsd:
            
            # cc.diis_space = 10
            mf2 = scf.addons.remove_linear_dep_(mf).run()
            # mycc = cc.CCSD(mf).run()
            cc = cup_clus.KUCCSD(mf2)
            cc.kernel()

        # FCI
        # fci = pyscf.fci.FCI(mf)
        # fci.kernel()

        # Vayesta options
        use_sym = False
        nfrag = 1 
        # bath_type = "dmet"
        # bath_opts = dict(bathtype=bath_type)   
        bath_type = "mp2"
        bath_opts = dict(bathtype=bath_type,threshold_occ=1,threshold_vir=1,truncation_occ="number",truncation_vir="number") 
        ncells = np.product(kmesh)
        # Run vayesta with user defined solver
        emb = vayesta.ewf.EWF(mf, solver="CALLBACK",  energy_functional='wf', bath_options=bath_opts, solver_options=dict(callback=fci_solver))
        # Set up fragments
        # # emb = vayesta.ewf.EWF(mf,energy_functional='wf',bath_options=dict(bathtype="dmet"))
        with emb.iao_fragmentation() as f:
            # f.add_atomic_fragment([0], orbital_filter=["Ni 3dxy","Ni 3dyz","Ni 3dxz"])
            # f.add_atomic_fragment([0], orbital_filter=["Ni 3dx2-y2","Ni 3dz^2"])
            f.add_atomic_fragment([0], orbital_filter=["Ni 3d"])
            f.add_atomic_fragment([0], orbital_filter=["Ni 3s","Ni 3p"])#,"Ni 3dx2-y2","Ni 3dz^2"])
            
            f.add_atomic_fragment([0], orbital_filter=["Ni 4s","Ni 4p"])

            # f.add_atomic_fragment([0,1], orbital_filter=["Ni 4s","Ni 4p","O 2p"])
            # f.add_atomic_fragment([1], orbital_filter=["O 2s", "O 3s"])
            
            # f.add_atomic_fragment([1], orbital_filter=["O 2s", "O 3s"])
            # f.add_atomic_fragment([1], orbital_filter=["O 2p"])
            f.add_atomic_fragment([1], orbital_filter=["O 2s","O 3s"])
            f.add_atomic_fragment([1], orbital_filter=["O 2p"])

            f.add_atomic_fragment([2], orbital_filter=["Ni 3s","Ni 3p"])#,"Ni 3dx2-y2","Ni 3dz^2"])
            # f.add_atomic_fragment([2], orbital_filter=["Ni 3dxy","Ni 3dyz","Ni 3dxz","Ni 3dx2-y2","Ni 3dz^2"])
            # f.add_atomic_fragment([2], orbital_filter=["Ni 3dxy","Ni 3dyz","Ni 3dxz"])
            # f.add_atomic_fragment([2], orbital_filter=["Ni 3dx2-y2","Ni 3dz^2"])
            f.add_atomic_fragment([2], orbital_filter=["Ni 3d"])
            # f.add_atomic_fragment([2], orbital_filter=["Ni 3d"])
            # f.add_atomic_fragment([2], orbital_filter=["Ni 3dxz","Ni 3dxy","Ni 3dyz","Ni 3dx2-y2","Ni 3dz\^2"])
            # f.add_atomic_fragment([2], orbital_filter=["Ni 3dxy","Ni 3dyz"])
            # f.add_atomic_fragment([2], orbital_filter=["Ni 3dxz","Ni 3dx2-y2","Ni 3dz\^2"])
            # f.add_atomic_fragment([2], orbital_filter=["Ni 3dx2-y2","Ni 3dz\^2"])
            f.add_atomic_fragment([2], orbital_filter=["Ni 4s","Ni 4p"])
            # f.add_atomic_fragment([2], orbital_filter=["Ni 4s","Ni 4p"])
            # f.add_atomic_fragment([2,3], orbital_filter=["Ni 4s","Ni 4p","O 2p"])
            # f.add_atomic_fragment([3], orbital_filter=["O 2s", "O 3s"])
            
            f.add_atomic_fragment([3], orbital_filter=["O 2s","O 3s"])
            f.add_atomic_fragment([3], orbital_filter=["O 2p"])
        emb.kernel()

        emb_fci = vayesta.ewf.EWF(mf, solver="FCI", bath_options=bath_opts, solver_options=dict(conv_tol=1.e-14))
        # Set up fragments
        with emb_fci.iao_fragmentation() as f:
            f.add_atomic_fragment([0], orbital_filter=["Ni 3d"])
            # f.add_atomic_fragment([0], orbital_filter=["Ni 3dxy","Ni 3dyz"])
            # f.add_atomic_fragment([0], orbital_filter=["Ni 3dxy","Ni 3dyz","Ni 3dxz"])
            # f.add_atomic_fragment([0], orbital_filter=["Ni 3dx2-y2","Ni 3dz^2"])
            f.add_atomic_fragment([0], orbital_filter=["Ni 3s","Ni 3p"])#,"Ni 3dx2-y2","Ni 3dz^2"])
            
            f.add_atomic_fragment([0], orbital_filter=["Ni 4s","Ni 4p"])

            # f.add_atomic_fragment([0,1], orbital_filter=["Ni 4s","Ni 4p","O 2p"])
            # f.add_atomic_fragment([1], orbital_filter=["O 2s", "O 3s"])
            
            # f.add_atomic_fragment([1], orbital_filter=["O 2s", "O 3s"])
            f.add_atomic_fragment([1], orbital_filter=["O 2p"])
            f.add_atomic_fragment([1], orbital_filter=["O 2s","O 3s"])
            # f.add_atomic_fragment([1], orbital_filter=["O 2p"])

            f.add_atomic_fragment([2], orbital_filter=["Ni 3s","Ni 3p"])#,"Ni 3dx2-y2","Ni 3dz^2"])
            # f.add_atomic_fragment([2], orbital_filter=["Ni 3dxy","Ni 3dyz","Ni 3dxz","Ni 3dx2-y2","Ni 3dz^2"])
            # f.add_atomic_fragment([2], orbital_filter=["Ni 3dxy","Ni 3dyz","Ni 3dxz"])
            # f.add_atomic_fragment([2], orbital_filter=["Ni 3dx2-y2","Ni 3dz^2"])
            f.add_atomic_fragment([2], orbital_filter=["Ni 3d"])
            # f.add_atomic_fragment([2], orbital_filter=["Ni 3dxz","Ni 3dxy","Ni 3dyz","Ni 3dx2-y2","Ni 3dz\^2"])
            # f.add_atomic_fragment([2], orbital_filter=["Ni 3dxy","Ni 3dyz"])
            # f.add_atomic_fragment([2], orbital_filter=["Ni 3dxz","Ni 3dx2-y2","Ni 3dz\^2"])
            # f.add_atomic_fragment([2], orbital_filter=["Ni 3dx2-y2","Ni 3dz\^2"])
            f.add_atomic_fragment([2], orbital_filter=["Ni 4s","Ni 4p"])
            # f.add_atomic_fragment([2], orbital_filter=["Ni 4s","Ni 4p"])
            # f.add_atomic_fragment([2,3], orbital_filter=["Ni 4s","Ni 4p","O 2p"])
            # f.add_atomic_fragment([3], orbital_filter=["O 2s", "O 3s"])
            
            f.add_atomic_fragment([3], orbital_filter=["O 2s","O 3s"])
            f.add_atomic_fragment([3], orbital_filter=["O 2p"])
            # f.add_atomic_fragment([3], orbital_filter=["O 2p"])
            # f.add_all_atomic_fragments()
        emb_fci.kernel()

        # emb_atom_fci = vayesta.ewf.EWF(mf, solver="CALLBACK", bath_options=bath_opts, solver_options=dict(callback=ccsd_solver))
        # # Set up fragments
        # with emb_atom_fci.iao_fragmentation() as f:
        #     f.add_all_atomic_fragments()
        # emb_atom_fci.kernel()

        # emb_atom_fci = vayesta.ewf.EWF(mf, solver="FCI", bath_options=bath_opts, solver_options=dict(conv_tol=1.e-14))
        # # Set up fragments
        # with emb_atom_fci.iao_fragmentation() as f:
        #     f.add_all_atomic_fragments()
        # emb_atom_fci.kernel()

        print("Hartree-Fock energy          : %s"%mf.e_tot)
        # print("CISD Energy                  : %s"%cisd.e_tot)
        if do_ccsd:
            print("CCSD Energy                  : %s"%cc.e_tot)
        # print("FCI  Energy                  : %s"%fci.e_tot)
        print("Emb. Partitioned Cumulant    : %s"%emb.e_tot)
        if do_ccsd:
            f_name="NiO_HS_only_fci_solver_lat_con_"+str(atomic_dis)+"_nshots_"+str(Nshots)+"_k_"+str(kmesh[0])+str(kmesh[1])+str(kmesh[2])+"_basis_"+str(cell.basis)+"_"+bath_type+"_bath"+"magentics_"+magnetism+".dat"
            if os.path.isfile(f_name):
                f = open(f_name, "a")
                # f.write(str(mf.e_tot)+"    "+str(cc.e_tot)+"    "+str(fci.e_tot)+"    "+str(emb_atom_fci.e_tot)+"    "+str(emb_fci.e_tot)+"    "+str(emb.e_tot)+"\n")
                # f.write(str(mf.e_tot)+"    "+str(cc.e_tot)+"    "+str(emb_atom_fci.e_tot)+"    "+str(emb_fci.e_tot)+"    "+str(emb.e_tot)+"\n")
                f.write(str(mf.e_tot)+"    "+str(cc.e_tot)+"    "+str(emb_fci.e_tot)+"    "+str(emb.e_tot)+"\n")
                f.close()
            else:
                f = open(f_name, "a")
                # f.write("c0"+"    "+"e_fci"+"    "+"e_hf"+"    "+"vqe_energy"+"    "+"HF+CS"+"    "+"HF+meas_diag"+"    "+"E_pauli_all"+"\n")
                # f.write(str(c0)+"    "+str(e_fci)+"    "+str(myhf.e_tot)+"    "+str(vqe_energy)+"    "+str(e_proj_CS)+"    "+str(e_proj_meas_diag)+"    "+str(result_Pauli_all_comm)+"\n")
                
                # f.write("HF"+"    "+"CCSD"+"    "+"FCI_atom"+"    "+"FCI_emb"+"    "+"VQE_emb"+"\n")
                # f.write(str(mf.e_tot)+"    "+str(cc.e_tot)+"    "+str(emb_atom_fci.e_tot)+"    "+str(emb_fci.e_tot)+"    "+str(emb.e_tot)+"\n")
                
                f.write("HF"+"    "+"CCSD"+"    "+"FCI_emb"+"    "+"VQE_emb"+"\n")
                f.write(str(mf.e_tot)+"    "+str(cc.e_tot)+"    "+str(emb_fci.e_tot)+"    "+str(emb.e_tot)+"\n")
                f.close()
        else:
            f_name="NiO_HS_only_fci_solver_lat_con_"+str(atomic_dis)+"_nshots_"+str(Nshots)+"_k_"+str(kmesh[0])+str(kmesh[1])+str(kmesh[2])+"_basis_"+str(cell.basis)+"_"+bath_type+"_bath"+"magentics_"+magnetism+".dat"
            if os.path.isfile(f_name):
                f = open(f_name, "a")
                # f.write(str(mf.e_tot)+"    "+str(cc.e_tot)+"    "+str(fci.e_tot)+"    "+str(emb_atom_fci.e_tot)+"    "+str(emb_fci.e_tot)+"    "+str(emb.e_tot)+"\n")
                #f.write(str(mf.e_tot)+"    "+str(emb_atom_fci.e_tot)+"    "+str(emb_fci.e_tot)+"    "+str(emb.e_tot)+"\n")
                f.write(str(mf.e_tot)+"    "+str(emb_fci.e_tot)+"    "+str(emb.e_tot)+"\n")
                f.close()
            else:
                f = open(f_name, "a")
                # f.write("c0"+"    "+"e_fci"+"    "+"e_hf"+"    "+"vqe_energy"+"    "+"HF+CS"+"    "+"HF+meas_diag"+"    "+"E_pauli_all"+"\n")
                # f.write(str(c0)+"    "+str(e_fci)+"    "+str(myhf.e_tot)+"    "+str(vqe_energy)+"    "+str(e_proj_CS)+"    "+str(e_proj_meas_diag)+"    "+str(result_Pauli_all_comm)+"\n")
                f.write("HF"+"    "+"FCI_emb"+"    "+"VQE_emb"+"\n")
                f.write(str(mf.e_tot)+"    "+str(emb_fci.e_tot)+"    "+str(emb.e_tot)+"\n")
                f.close()



