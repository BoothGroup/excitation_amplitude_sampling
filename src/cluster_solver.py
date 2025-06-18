from qiskit_nature.second_q.operators import ElectronicIntegrals, FermionicOp ,SparseLabelOp, PolynomialTensor, tensor_ordering
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
#from qiskit_nature.second_q.circuit.library import UCCSD
from qiskit_nature.second_q.circuit.library.ansatzes.utils import generate_fermionic_excitations 


########### VQE Modules #######
# from qiskit.utils import algorithm_globals
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector


######### Optimiser Modules #########


# from qiskit.opflow import StateFn, PauliExpectation, CircuitSampler
# from qiskit.opflow import CircuitStateFn, PauliSumOp
from qiskit.quantum_info import SparsePauliOp


from qiskit.circuit.library import NLocal,EfficientSU2
######### PySCF + others #########
# from qiskit.extensions import *

import sys
sys.argv.append("--quiet")  # For Vayesta
import numpy as np
from pyscf import gto, scf, lib, ao2mo
from pyscf.fci import cistring
# from vayesta.lattmod import Hubbard1D, LatticeRHF
import itertools
import math
import array_to_latex as a2l
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
import pandas as pd 
##import matplotlib.pyplot as plt
from qiskit.quantum_info.operators import Operator
import qiskit

from src import classical_shadow
from src import swap_test
from src import observables
from src import solvers
# import classical_shadow
# import swap_test
# import observables
# import vqe_solver

import cmath
import pyscf
from pyscf import gto, scf, ao2mo, fci, ci
from pyscf.ci import cisd
from pyscf import cc as cupclus

# from qiskit import Aer
from qiskit.quantum_info import Statevector

#from src import Lan_coeffs

class cluster_solver():

    def __init__(self,h1e, h2e, norb, nelec, ovlp,hf_type="RHF",solver="VQE",noisy=False,t2s=None):
        # mol = gto.M()
        # mol.incore_anyway = True
        # mol.nelectron = nelec
        self.hf_type=hf_type
        self.norb=norb
        self.nelec=nelec
        self.nelec_T = sum(nelec)
        self.noisy = noisy
        self.mol_c = gto.Mole()
        self.mol_c.incore_anyway = True
        self.solver_name = solver
        if hf_type == "RHF":
            self.h1e=h1e
            self.h2e=h2e
            self.mol_c.nelectron = np.sum(nelec)
            self.mf_c = scf.RHF(self.mol_c)
            self.mf_c.get_hcore = lambda *args: h1e
            self.mf_c.get_ovlp = lambda *args: ovlp#np.eye(norb)
            self.mf_c._eri = ao2mo.restore(1, h2e, norb)
            self.mf_c.kernel()
            self.hf_E = self.mf_c.energy_tot()
            integrals = ElectronicIntegrals.from_raw_integrals(h1_a=h1e, h1_b = h1e, h2_aa=h2e, h2_bb= h2e, h2_ba= h2e , auto_index_order=True) 
            avg_norb = norb
            self.nelec_T = sum(nelec)
            self.nocc = self.nelec_T // 2
            self.nvir = self.norb - self.nocc
            self.SV_basis_0 = "0"*(self.norb-self.nocc) + "1"*(self.nocc) + "0"*(self.norb-self.nocc) + "1"*(self.nocc)
        elif hf_type == "UHF":
            self.h1e=h1e
            self.h2e=h2e
            self.h1ea = h1e[0]
            self.h1eb = h1e[1]
            self.h2eaa = h2e[0]
            self.h2eab = h2e[1]
            self.h2ebb = h2e[2]
            self.mol_c.nelec = nelec
            self.mf_c = scf.UHF(self.mol_c)
            self.mf_c.get_hcore = lambda *args: h1e
            self.mf_c.get_ovlp = lambda *args: ovlp#np.eye(norb)
            # self.mf_c._eri = h2e#ao2mo.restore(1, h2e, norb)
            # self.mf_c.kernel()
            self.nocc = nelec#(int(sum(self.mf_c.get_occ()[0])),int(sum(self.mf_c.get_occ()[1])))#self.nelec_T // 2
            self.nvir = (self.norb[0] - self.nocc[0],self.norb[1] - self.nocc[1])
            self.nvira = self.nvir[0]
            self.nvirb = self.nvir[1]

            self.norba = self.norb[0]
            self.norbb = self.norb[1]
            self.nocca = self.nocc[0]
            self.noccb = self.nocc[1]
            print("nocc, nelec: ", self.nocc,self.nelec)
            avg_norb=sum(norb)//2####assume same number of orbitals in a and b spin channels
            # self.hf_E = self.mf_c.energy_tot()
            integrals = ElectronicIntegrals.from_raw_integrals(h1_a=self.h1ea, h1_b = self.h1eb, h2_aa=self.h2eaa, h2_bb= self.h2ebb, h2_ba= self.h2eab.transpose(2,3,0,1) , auto_index_order=True)    #0,2,1,3
            self.SV_basis_0 = "0"*(self.norbb-self.noccb) + "1"*(self.noccb) + "0"*(self.norba-self.nocca) + "1"*(self.nocca) #qiskit.quantum_info.Statevector.from_instruction(self.initial_state) 
        self.h_elec = ElectronicEnergy(integrals).second_q_op()       
        mapper = JordanWignerMapper()
        self.qubit_ham = mapper.map(self.h_elec) # this performs the JW transformation to qubit representation
        self.num_qubits = self.qubit_ham.num_qubits
        if solver == "VQE":
            if hf_type=="UHF":
                NotImplementedError("not implemented for UHF")
            self.solver = solvers.vqe_solver(self.qubit_ham, avg_norb, nelec,mapper)
        elif solver == "FCI":
            if hf_type=="RHF":
                NotImplementedError("not implemented for RHF")
            self.solver = solvers.fci_solver(h1e,h2e,norb,nelec)
        elif solver == "ffsim":
            ### the LUCJ ansatz is only implemented for UHF but is very inefficient. The FFsim package is really designed for rhf, but it is not implemented for RHF here.
            if hf_type=="RHF":
                NotImplementedError("not implemented for RHF")
            self.solver = solvers.ffsim_vqe_solver(self.qubit_ham,norb,[self.nocca,self.noccb],t2s)    

        self.backend = AerSimulator(method='statevector')#Aer.get_backend('statevector_simulator')
        
        self.estimator=qiskit.primitives.Estimator()
        


    def solve(self,my_solver_options):
        if self.noisy:
            assert "Statevector" in my_solver_options
            assert not my_solver_options["Statevector"]

        self.GS_circ, self.GS_E = self.solver.solve(my_solver_options)


    def measure_classical_shadows_perp(self,Nshots, N_estimator, ensemble="Clifford",noise_alpha=0.0):    
        if self.hf_type=="RHF":
            sv_exact=list(Statevector(self.GS_circ))
            

            ### get the fci c0,c1,c2 and the permutational signs and 1 and 2 electron integrals
            class_c0,class_c1,class_c2,t1sign,f_ov, g_ovvo= self.classical_coeffs(self.h1e,self.h2e,self.norb,self.nelec,self.mf_c)
            # fci_e,ci_vec = pyscf.fci.direct_spin0.kernel(h1e, h2e, norb, nelec=(nocc, nocc), conv_tol=1.e-16)
            # print("cluster ci vec: ", ci_vec)

            self.c0_len=1
            self.c1_len=0
            self.c2_len=0
            c1_indices,c1_labels,c1_qubit_indices,c2_indices,c2_labels,c2_qubit_indices,fermi_vac,fermi_vac_split = self.get_c1_c2_calc(class_c1,class_c2)


            
            ### calculate the classical shadow 
            Shadow = classical_shadow.Classical_shadow(ensemble,self.num_qubits,[],[])
            if self.noisy:
                Shadow.calc_noisy_clifford_shadow_lists(self.GS_circ,N_estimator,int(Nshots/N_estimator))
            else:    
                Shadow.calc_shadow_lists(Statevector(1/np.sqrt(2)*(np.array(qiskit.quantum_info.Statevector.from_label("0"*self.num_qubits)) + np.array(Statevector(self.GS_circ)))),N_estimator,int(Nshots/N_estimator))




            op_list=[]

            index_a=0
            
            ###index for HF state
            index_b=int(self.SV_basis_0,2)
            op_list.append([1.,index_a,index_b])


            results_l,errors_l = Shadow.evaluate_operator_overlap_stabilizer_perp(op_list)
            # print(results_l)


            c0_cs=results_l[0]


            c1_cs=np.zeros((len(fermi_vac_split[0]),len(fermi_vac_split[1])))

            a_exc = generate_fermionic_excitations(num_excitations=1, num_spatial_orbitals=self.norb, num_particles=self.nelec, alpha_spin=True, beta_spin=False, max_spin_excitation=None, generalized=False, preserve_spin=True)
            a_exc_strings=excitations_tuples_to_strings(a_exc,fermi_vac)
            op_list=[]
            for a in a_exc_strings:
                index_a=0
                index_b=int(a,2)
                op_list.append([1.,index_a,index_b])
            results_l,errors_l = Shadow.evaluate_operator_overlap_stabilizer_perp(op_list)
            a_int_norm_results = [x for x in results_l]
            c1_cs_rs = (a_int_norm_results * t1sign).reshape(self.nocc, self.nvir)

            print("\n\n\n")

            ### construct the labels for operators of the form |index_a><index_b| + |index_b><index_a| (for the real part) + imaginary part
            op_list=[]
            for two_p in c2_labels:    
                SV_basis_1=qiskit.quantum_info.Statevector.from_label(two_p)
                
                index_b=int(two_p,2)
                index_a=0
                
                op_list.append([1.,index_a,index_b])

            ### evaluate overlaps
            results_l,errors_l = Shadow.evaluate_operator_overlap_stabilizer_perp(op_list)



            ### manipulate overlaps to match pyscf output
            c2_cs=np.zeros((len(fermi_vac_split[0]),len(fermi_vac_split[1]),len(fermi_vac_split[2]),len(fermi_vac_split[3])))
            for n in range(len(c2_labels)):
                c2_cs[c2_indices[n][0]][c2_indices[n][1]][c2_indices[n][2]][c2_indices[n][3]]=results_l[n]#/(c0_cs.real)
            # print(c2)
            c2_cs_rs=c2_cs.transpose(0, 2, 1, 3)
            c2_cs_rs = c2_cs_rs.reshape((self.nocc*self.nvir, self.nocc*self.nvir))
            c2_cs_rs = np.einsum("i,j,ij->ij", t1sign, t1sign, c2_cs_rs)
            c2_cs_rs = c2_cs_rs.reshape((self.nocc, self.nvir, self.nocc, self.nvir))
            c2_cs_rs = c2_cs_rs.transpose(0, 2, 1, 3)

            ### calculate energies
            e_singles = 2 * np.sum(f_ov * c1_cs_rs) * (1/c0_cs)
            e_doubles = 2 * np.einsum("ijab,iabj->", c2_cs_rs, g_ovvo) - np.einsum("ijab,ibaj->", c2_cs_rs, g_ovvo) * (1/c0_cs)
            # e_proj_CS_perp = myhf.e_tot + e_singles + e_doubles






            e_singles = 2 * np.sum(f_ov * c1_cs_rs)
            e_doubles = 2 * np.einsum("ijab,iabj->", c2_cs_rs, g_ovvo) - np.einsum("ijab,ibaj->", c2_cs_rs, g_ovvo)
        elif self.hf_type == "UHF":
            fermi_vac=self.SV_basis_0
            c0,c1,c2,t1signa,t1signb,t1addra,t1addrb,t2signa,t2signb,t2addra,t2addrb, nocca_comp,noccb_comp,nvira_comp,nvirb_comp, fova,fovb,gaa,gab,gbb = self.classical_coeffs_uhf()
            a_exc = generate_fermionic_excitations(num_excitations=1, num_spatial_orbitals=self.norba, num_particles=self.nelec, alpha_spin=True, beta_spin=False, max_spin_excitation=None, generalized=False, preserve_spin=True)
            print("a_exc: ",a_exc)
            print(self.norba,self.nelec)
            a_exc_strings=excitations_tuples_to_strings(a_exc,fermi_vac)
            print("a_exc_strings: ",a_exc_strings)
            b_exc = generate_fermionic_excitations(num_excitations=1, num_spatial_orbitals=self.norba, num_particles=self.nelec, alpha_spin=False, beta_spin=True, max_spin_excitation=None, generalized=False, preserve_spin=True)
            b_exc_strings=excitations_tuples_to_strings(b_exc,fermi_vac)
            print("b_exc: ",b_exc)
            print(self.norbb,self.nelec)
            print("b_exc_strings: ",b_exc_strings)

            all_exc = generate_fermionic_excitations(num_excitations=2, num_spatial_orbitals=self.norba, num_particles=self.nelec, alpha_spin=True, beta_spin=True, max_spin_excitation=None, generalized=False, preserve_spin=True)  
            aa_exc = generate_fermionic_excitations(num_excitations=2, num_spatial_orbitals=self.norba, num_particles=self.nelec, alpha_spin=True, beta_spin=False, max_spin_excitation=None, generalized=False, preserve_spin=True)
            bb_exc = generate_fermionic_excitations(num_excitations=2, num_spatial_orbitals=self.norba, num_particles=self.nelec, alpha_spin=False, beta_spin=True, max_spin_excitation=None, generalized=False, preserve_spin=True)
            aa_exc_strings=excitations_tuples_to_strings(aa_exc,fermi_vac)
            bb_exc_strings=excitations_tuples_to_strings(bb_exc,fermi_vac)
            for n in aa_exc:
                if n in all_exc: all_exc.remove(n)
            for n in bb_exc:
                if n in all_exc: all_exc.remove(n)    
            ab_exc = all_exc
            ab_exc_strings=excitations_tuples_to_strings(ab_exc,fermi_vac)
            
            Shadow = classical_shadow.Classical_shadow(ensemble,self.num_qubits,[],[])

            if self.noisy:
                Shadow.calc_noisy_clifford_shadow_lists(self.GS_circ,N_estimator,int(Nshots/N_estimator),noise_alpha=noise_alpha)
            else:    
                Shadow.calc_shadow_lists(Statevector(1/np.sqrt(2)*(np.array(qiskit.quantum_info.Statevector.from_label("0"*self.num_qubits)) + np.array(Statevector(self.GS_circ)))),N_estimator,int(Nshots/N_estimator))
            op_list=[]
            index_a=0
            index_b=int(self.SV_basis_0,2)
            op_list.append([1.,index_a,index_b])
            results_l,errors_l = Shadow.evaluate_operator_overlap_stabilizer_perp(op_list)
            c0_cs = results_l[0]


            op_list=[]
            for a in a_exc_strings:
                index_a=0
                index_b=int(a,2)
                op_list.append([1.,index_a,index_b])
            results_l,errors_l = Shadow.evaluate_operator_overlap_stabilizer_perp(op_list)
            a_int_norm_results = [x for x in results_l]
            c1a_cs = (a_int_norm_results * t1signa).reshape(self.nocca, self.nvira)

            op_list=[]
            for b in b_exc_strings:
                index_a=0
                index_b=int(b,2)
                op_list.append([1.,index_a,index_b])
            results_l,errors_l = Shadow.evaluate_operator_overlap_stabilizer_perp(op_list)
            b_int_norm_results = [x for x in results_l]
            c1b_cs = (b_int_norm_results * t1signb).reshape(self.noccb, self.nvirb)

            op_list=[]
            for ab in ab_exc_strings:
                index_a=0
                index_b=int(ab,2)
                op_list.append([1.,index_a,index_b])
            results_l,errors_l = Shadow.evaluate_operator_overlap_stabilizer_perp(op_list)
            ab_int_norm_results = [x for x in results_l]

            op_list=[]
            for aa in aa_exc_strings:
                index_a=0
                index_b=int(aa,2)
                op_list.append([1.,index_a,index_b])
            
            results_l,errors_l = Shadow.evaluate_operator_overlap_stabilizer_perp(op_list)
            aa_int_norm_results = [x for x in results_l]

            op_list=[]
            for bb in bb_exc_strings:
                index_a=0
                index_b=int(bb,2)
                op_list.append([1.,index_a,index_b])
   
            results_l,errors_l = Shadow.evaluate_operator_overlap_stabilizer_perp(op_list)
            bb_int_norm_results = [x for x in results_l]



            ###put into pyscf civec type array
            my_civec_cs=np.zeros((math.comb(self.norba,self.nocca),math.comb(self.norbb,self.noccb)))
            my_civec_cs[0,0] = c0_cs

            for a_count in range(len(a_exc_strings)):
                a = a_exc_strings[a_count]
                assert len(a)%2==0 ###even number of qubits
                ci_index2 = cistring.str2addr(self.norbb, self.noccb, bin(int(a[:len(a)//2],2)))
                ci_index1 = cistring.str2addr(self.norba, self.nocca, bin(int(a[len(a)//2:],2)))
                my_civec_cs[ci_index1,ci_index2] = a_int_norm_results[a_count]

            for b_count in range(len(b_exc_strings)):
                b = b_exc_strings[b_count]
                assert len(b)%2==0
                ci_index2 = cistring.str2addr(self.norbb, self.noccb, bin(int(b[:len(b)//2],2)))
                ci_index1 = cistring.str2addr(self.norba, self.nocca, bin(int(b[len(b)//2:],2)))
                my_civec_cs[ci_index1,ci_index2] = b_int_norm_results[b_count]    

            for ab_count in range(len(ab_exc_strings)):
                ab = ab_exc_strings[ab_count]
                assert len(ab)%2==0
                ci_index2 = cistring.str2addr(self.norbb, self.noccb, bin(int(ab[:len(ab)//2],2)))
                ci_index1 = cistring.str2addr(self.norba, self.nocca, bin(int(ab[len(ab)//2:],2)))
                my_civec_cs[ci_index1,ci_index2] = ab_int_norm_results[ab_count]

            for aa_count in range(len(aa_exc_strings)):
                aa = aa_exc_strings[aa_count]
                assert len(aa)%2==0
                ci_index2 = cistring.str2addr(self.norbb, self.noccb, bin(int(aa[:len(aa)//2],2)))
                ci_index1 = cistring.str2addr(self.norba, self.nocca, bin(int(aa[len(aa)//2:],2)))
                # print(ci_index1,ci_index2)
                my_civec_cs[ci_index1,ci_index2] = aa_int_norm_results[aa_count]   

            for bb_count in range(len(bb_exc_strings)):
                bb = bb_exc_strings[bb_count]
                assert len(bb)%2==0
                ci_index2 = cistring.str2addr(self.norbb, self.noccb, bin(int(bb[:len(bb)//2],2)))
                ci_index1 = cistring.str2addr(self.norba, self.nocca, bin(int(bb[len(bb)//2:],2)))
                # print(ci_index1,ci_index2)
                my_civec_cs[ci_index1,ci_index2] = bb_int_norm_results[bb_count]        


            na = pyscf.fci.cistring.num_strings(self.norba, self.nocca)
            nb = pyscf.fci.cistring.num_strings(self.norbb, self.noccb)

            # print("c1a: ",civec[t1addra, 0])
            # print("c1b: ",civec[0, t1addrb])
            c1a_cs = (my_civec_cs[t1addra, 0] * t1signa).reshape(self.nocca, self.nvira)
            c1b_cs = (my_civec_cs[0, t1addrb] * t1signb).reshape(self.noccb, self.nvirb)
            
            nocca_comb = self.nocca * (self.nocca - 1) // 2
            noccb_comb = self.noccb * (self.noccb - 1) // 2
            nvira_comb = self.nvira * (self.nvira - 1) // 2
            nvirb_comb = self.nvirb * (self.nvirb - 1) // 2
            c2aa_cs = (my_civec_cs[t2addra, 0] * t2signa).reshape(nocca_comb, nvira_comb)
            c2bb_cs = (my_civec_cs[0, t2addrb] * t2signb).reshape(noccb_comb, nvirb_comb)
            c2aa_cs = pyscf.cc.ccsd._unpack_4fold(c2aa_cs, self.nocca, self.nvira)
            c2bb_cs = pyscf.cc.ccsd._unpack_4fold(c2bb_cs, self.noccb, self.nvirb)
            c2ab_cs = np.einsum("i,j,ij->ij", t1signa, t1signb, my_civec_cs[t1addra[:, None], t1addrb])
            c2ab_cs = c2ab_cs.reshape(self.nocca, self.nvira, self.noccb, self.nvirb).transpose(0, 2, 1, 3)
            
            
            
            e_singles = (np.sum(fova * c1a_cs) + np.sum(fovb * c1b_cs)) * (1 / c0_cs)


            e_doubles = (
                            np.einsum("ijab,iajb", c2aa_cs, gaa) / 4
                            - np.einsum("ijab,ibja", c2aa_cs, gaa) / 4
                            + np.einsum("ijab,iajb", c2bb_cs, gbb) / 4
                            - np.einsum("ijab,ibja", c2bb_cs, gbb) / 4
                            + np.einsum("ijab,iajb", c2ab_cs, gab)
                        ) * (1 / c0_cs)
            # c1_cs_rs=(c1a_cs,c1b_cs)
            # c2_cs_rs=(c2aa_cs,c2ab_cs,c2bb_cs)
            # c0_cs = c0_cs
            c1_cs_rs=(c1a_cs*cmath.exp(complex(0,-cmath.phase(c0_cs))),c1b_cs*cmath.exp(complex(0,-cmath.phase(c0_cs))))
            c2_cs_rs=(c2aa_cs*cmath.exp(complex(0,-cmath.phase(c0_cs))),c2ab_cs,c2bb_cs*cmath.exp(complex(0,-cmath.phase(c0_cs))))
            c0_cs = c0_cs*cmath.exp(complex(0,-cmath.phase(c0_cs)))

        return (e_singles + e_doubles), [c0_cs, c1_cs_rs, c2_cs_rs]
    


    def measure_fci_civec_coeffs(self,Nshots, N_estimator, ensemble="Clifford"):   

        ### get c1s and c2s directly from the statevector but otherwise the same as the shadows procedure UHF only
        ### meant for validation

        if self.hf_type=="RHF":
            return NotImplementedError("RHF not implemented, can implement by restricting UHF code to only ab excitations")
        
        elif self.hf_type == "UHF":
            SV_exact = np.array(Statevector(self.GS_circ))
            fermi_vac=self.SV_basis_0
            c0,c1,c2,t1signa,t1signb,t1addra,t1addrb,t2signa,t2signb,t2addra,t2addrb, nocca_comp,noccb_comp,nvira_comp,nvirb_comp, fova,fovb,gaa,gab,gbb = self.classical_coeffs_uhf()
            a_exc = generate_fermionic_excitations(num_excitations=1, num_spatial_orbitals=self.norba, num_particles=self.nelec, alpha_spin=True, beta_spin=False, max_spin_excitation=None, generalized=False, preserve_spin=True)
            print("a_exc: ",a_exc)
            print(self.norba,self.nelec)
            a_exc_strings=excitations_tuples_to_strings(a_exc,fermi_vac)
            print("a_exc_strings: ",a_exc_strings)
            b_exc = generate_fermionic_excitations(num_excitations=1, num_spatial_orbitals=self.norba, num_particles=self.nelec, alpha_spin=False, beta_spin=True, max_spin_excitation=None, generalized=False, preserve_spin=True)
            b_exc_strings=excitations_tuples_to_strings(b_exc,fermi_vac)
            print("b_exc: ",b_exc)
            print(self.norbb,self.nelec)
            print("b_exc_strings: ",b_exc_strings)

            all_exc = generate_fermionic_excitations(num_excitations=2, num_spatial_orbitals=self.norba, num_particles=self.nelec, alpha_spin=True, beta_spin=True, max_spin_excitation=None, generalized=False, preserve_spin=True)  
            aa_exc = generate_fermionic_excitations(num_excitations=2, num_spatial_orbitals=self.norba, num_particles=self.nelec, alpha_spin=True, beta_spin=False, max_spin_excitation=None, generalized=False, preserve_spin=True)
            bb_exc = generate_fermionic_excitations(num_excitations=2, num_spatial_orbitals=self.norba, num_particles=self.nelec, alpha_spin=False, beta_spin=True, max_spin_excitation=None, generalized=False, preserve_spin=True)
            aa_exc_strings=excitations_tuples_to_strings(aa_exc,fermi_vac)
            bb_exc_strings=excitations_tuples_to_strings(bb_exc,fermi_vac)
            for n in aa_exc:
                if n in all_exc: all_exc.remove(n)
            for n in bb_exc:
                if n in all_exc: all_exc.remove(n)    
            ab_exc = all_exc
            ab_exc_strings=excitations_tuples_to_strings(ab_exc,fermi_vac)
            # Shadow = classical_shadow.Classical_shadow(Statevector(1/np.sqrt(2)*(np.array(qiskit.quantum_info.Statevector.from_label("0"*self.num_qubits)) + np.array(Statevector(self.GS_circ)))),N_estimator,int(Nshots/N_estimator),ensemble)
            
            
            op_list=[]
            index_a=0
            index_b=int(self.SV_basis_0,2)
            
            c0_cs = SV_exact[index_b]
            
            


            op_list=[]
            results_l=[]
            for a in a_exc_strings:
                index_a=0
                index_b=int(a,2)
                results_l.append(SV_exact[index_b])
            a_int_norm_results = [x for x in results_l]

            op_list=[]
            results_l=[]
            for b in b_exc_strings:
                index_a=0
                index_b=int(b,2)
                results_l.append(SV_exact[index_b])
            b_int_norm_results = [x for x in results_l]

            op_list=[]
            results_l=[]
            for ab in ab_exc_strings:
                index_a=0
                index_b=int(ab,2)
                results_l.append(SV_exact[index_b])
            ab_int_norm_results = [x for x in results_l]

            op_list=[]
            results_l=[]
            for aa in aa_exc_strings:
                index_a=0
                index_b=int(aa,2)
                results_l.append(SV_exact[index_b])
                op_list.append([1.,index_a,index_b])
            
            aa_int_norm_results = [x for x in results_l]

            op_list=[]
            results_l=[]
            for bb in bb_exc_strings:
                index_a=0
                index_b=int(bb,2)
                results_l.append(SV_exact[index_b])
   
            bb_int_norm_results = [x for x in results_l]


            my_civec_cs=np.zeros((math.comb(self.norba,self.nocca),math.comb(self.norbb,self.noccb)))
            my_civec_cs[0,0] = c0_cs#SV_exact[int("00110011",2)]

            for a_count in range(len(a_exc_strings)):
                a = a_exc_strings[a_count]
                assert len(a)%2==0 ###even number of qubits
                ci_index2 = cistring.str2addr(self.norbb, self.noccb, bin(int(a[:len(a)//2],2)))
                ci_index1 = cistring.str2addr(self.norba, self.nocca, bin(int(a[len(a)//2:],2)))
                # print(ci_index1,ci_index2,a_int_norm_results[a_count])
                my_civec_cs[ci_index1,ci_index2] = a_int_norm_results[a_count]

            for b_count in range(len(b_exc_strings)):
                b = b_exc_strings[b_count]
                assert len(b)%2==0
                ci_index2 = cistring.str2addr(self.norbb, self.noccb, bin(int(b[:len(b)//2],2)))
                ci_index1 = cistring.str2addr(self.norba, self.nocca, bin(int(b[len(b)//2:],2)))
                # print(ci_index1,ci_index2,a_int_norm_results[a_count])
                my_civec_cs[ci_index1,ci_index2] = b_int_norm_results[b_count]    

            for ab_count in range(len(ab_exc_strings)):
                ab = ab_exc_strings[ab_count]
                assert len(ab)%2==0
                ci_index2 = cistring.str2addr(self.norbb, self.noccb, bin(int(ab[:len(ab)//2],2)))
                ci_index1 = cistring.str2addr(self.norba, self.nocca, bin(int(ab[len(ab)//2:],2)))
                # print(ci_index1,ci_index2)
                my_civec_cs[ci_index1,ci_index2] = ab_int_norm_results[ab_count]

            for aa_count in range(len(aa_exc_strings)):
                aa = aa_exc_strings[aa_count]
                assert len(aa)%2==0
                ci_index2 = cistring.str2addr(self.norbb, self.noccb, bin(int(aa[:len(aa)//2],2)))
                ci_index1 = cistring.str2addr(self.norba, self.nocca, bin(int(aa[len(aa)//2:],2)))
                # print(ci_index1,ci_index2)
                my_civec_cs[ci_index1,ci_index2] = aa_int_norm_results[aa_count]   

            for bb_count in range(len(bb_exc_strings)):
                bb = bb_exc_strings[bb_count]
                assert len(bb)%2==0
                ci_index2 = cistring.str2addr(self.norbb, self.noccb, bin(int(bb[:len(bb)//2],2)))
                ci_index1 = cistring.str2addr(self.norba, self.nocca, bin(int(bb[len(bb)//2:],2)))
                # print(ci_index1,ci_index2)
                my_civec_cs[ci_index1,ci_index2] = bb_int_norm_results[bb_count]            


            na = pyscf.fci.cistring.num_strings(self.norba, self.nocca)
            nb = pyscf.fci.cistring.num_strings(self.norbb, self.noccb)

            ci = my_civec_cs.reshape(na, nb)
            # print("c1a: ",civec[t1addra, 0])
            # print("c1b: ",civec[0, t1addrb])
            c1a_cs = (my_civec_cs[t1addra, 0] * t1signa).reshape(self.nocca, self.nvira)
            c1b_cs = (my_civec_cs[0, t1addrb] * t1signb).reshape(self.noccb, self.nvirb)
            
            nocca_comb = self.nocca * (self.nocca - 1) // 2
            noccb_comb = self.noccb * (self.noccb - 1) // 2
            nvira_comb = self.nvira * (self.nvira - 1) // 2
            nvirb_comb = self.nvirb * (self.nvirb - 1) // 2
            c2aa_cs = (my_civec_cs[t2addra, 0] * t2signa).reshape(nocca_comb, nvira_comb)
            c2bb_cs = (my_civec_cs[0, t2addrb] * t2signb).reshape(noccb_comb, nvirb_comb)
            c2aa_cs = pyscf.cc.ccsd._unpack_4fold(c2aa_cs, self.nocca, self.nvira)
            c2bb_cs = pyscf.cc.ccsd._unpack_4fold(c2bb_cs, self.noccb, self.nvirb)
            c2ab_cs = np.einsum("i,j,ij->ij", t1signa, t1signb, my_civec_cs[t1addra[:, None], t1addrb])
            c2ab_cs = c2ab_cs.reshape(self.nocca, self.nvira, self.noccb, self.nvirb).transpose(0, 2, 1, 3)

  




            e_singles = (np.sum(fova * c1a_cs) + np.sum(fovb * c1b_cs)) * (1 / c0_cs)


            e_doubles = (
                            np.einsum("ijab,iajb", c2aa_cs, gaa) / 4
                            - np.einsum("ijab,ibja", c2aa_cs, gaa) / 4
                            + np.einsum("ijab,iajb", c2bb_cs, gbb) / 4
                            - np.einsum("ijab,ibja", c2bb_cs, gbb) / 4
                            + np.einsum("ijab,iajb", c2ab_cs, gab)
                        ) * (1 / c0_cs)
            c1_cs_rs=(c1a_cs*cmath.exp(complex(0,-cmath.phase(c0_cs))),c1b_cs*cmath.exp(complex(0,-cmath.phase(c0_cs))))
            c2_cs_rs=(c2aa_cs*cmath.exp(complex(0,-cmath.phase(c0_cs))),c2ab_cs,c2bb_cs*cmath.exp(complex(0,-cmath.phase(c0_cs))))
            c0_cs = c0_cs*cmath.exp(complex(0,-cmath.phase(c0_cs)))


        return (e_singles + e_doubles), [c0_cs, c1_cs_rs, c2_cs_rs]

    




    def classical_coeffs(self,h1e,h2e,norb,nelec,myhf):
        ###performs an fci calculation to get exact coefficients and check that the fci result agrees between the mixed and expectation value functinoals, this and returning the c0, c1 ,c2 can be removed without issue. However the t1sign, f_ov and g_ovvo are required 
        nelec_T = sum(nelec)
        nocc = nelec_T // 2
        nvir = norb - nocc

        

        # create an FCI solver based on the SCF object (enforcing singlet symmetry)
        cisolver = fci.direct_spin0.FCI()
        e1, civec = cisolver.kernel(h1e, h2e, norb, nelec)
        e_fci = e1
        #print('E(FCI) = %.12f' % e_fci)
        #print('Size of FCI wave function: ', civec.shape)
        # Note that the wave function is 2D. This is because it is a matrix of dimension 'alpha excitations' x 'beta excitations'

        # Now, extract the C0, C1 and C2 coefficients of the HF, singles and doubles excitations
        c0 = civec[0,0]
        #print('Coefficient of FCI wave function on Hartree--Fock configuration: ',c0)

        # === C1 amplitudes ===
        # Note that we only consider alpha -> alpha
        # or beta -> beta excitations, and in a spin-restricted system, then these *should*
        # be the same.
        # These functions extract out the indicies and permutational signs of
        # the *same spin* excitations of a given rank from the FCI vector
        t1addr, t1sign = cisd.tn_addrs_signs(norb, nocc, 1)
        t1addr = np.asarray(t1addr, dtype=int)
        # C1 are taken to be the beta -> beta excitations (which should be
        # the same as alpha -> alpha), by taking the first (alpha) index to be doubly occupied.
        c1 = civec[0, t1addr] * t1sign
        c1 = c1.reshape((nocc, nvir))
        # Intermediately normalize C1 amplitudes by the HF reference amplitude
        c1 = c1 / c0

        # === C2 amplitudes ===
        # For RHF, we want the (alpha, beta) -> (alpha, beta) 'mixed spin' excitation amplitudes.
        # Therefore, we can just take single excitations of alpha and
        # combine with the single excitations of beta.
        # For a spin-symmetry-broken reference state (UHF), we would also need to get the aa->aa
        # and bb->bb excitations, but we can leave these for the moment.
        c2 = np.einsum("i,j,ij->ij", t1sign, t1sign, civec[t1addr[:, None], t1addr])
        c2 = c2.reshape((nocc, nvir, nocc, nvir))
        c2 = c2.transpose(0, 2, 1, 3)
        # Normalize by reference weight
        c2 = c2 / c0

        # c2[w, x, y, z] corresponds to the excitation of an alpha electron from (spatial) orbital w -> y+nocc
        # and the excitation of a beta electron x -> z+nocc.

        # Build Fock matrix for mean-field reference
        fock = myhf.get_fock()
        # Rotate and extract the occupied:virtual part of the Fock matrix
        f_ov = np.linalg.multi_dot((myhf.mo_coeff[:,:nocc].T, fock, myhf.mo_coeff[:,nocc:]))
        # Contract with the singles amplitudes to get the singles contribution to the correlation energy
        # Note: should be zero contrib to energy if mean-field=Hartree--Fock
        e_singles = 2 * np.sum(f_ov * c1)
        #print('Singles contribution to projected correlation energy: ',e_singles)

        # Two-electron contribution to correlation energy
        # Get (ov|vo) integral slice in MO basis:
        g_ovvo = h2e[:nocc,nocc:,nocc:,:nocc]
        e_doubles = 2 * np.einsum("ijab,iabj->", c2, g_ovvo) - np.einsum("ijab,ibaj->", c2, g_ovvo)
        #print('Doubles contribution to projected correlation energy: ',e_doubles)

        e_proj = myhf.energy_tot() + e_singles + e_doubles
        #print('Projected correlation energy + MF energy = ',e_proj)
        # cc = cupclus.CCSD(myhf)
        # cc.kernel()
        # print('CCSD total energy: {}'.format(cc.e_tot))
        print("c0: ",c0)
        assert np.isclose(e_proj, e_fci),"e_proj: "+str(e_proj)+" e_fci: "+str(e_fci)+"  "+str(myhf.energy_tot())+"  "+str(e_singles)+"  "+str(e_doubles)
        #print('Projected energy and FCI energy are the same')

        # print("c0: ",c0)
        # print("c1: ",c1)
        # print("c2: ",c2)
        return c0,c1,c2,t1sign,f_ov,g_ovvo
    
    def classical_coeffs_uhf(self):
        ### similarly to the rhf case the FCI calculation could be removed and the c0,c1,c2 not returned


        e1, civec = pyscf.fci.direct_uhf.kernel(self.h1e, self.h2e, self.norb[0], self.nelec, conv_tol=1.e-14)
        norba, norbb = self.norb
        nocca, noccb = self.nocc
        nvira, nvirb = self.nvir

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
        c1a = (civec[t1addra, 0] * t1signa).reshape(nocca, nvira)
        c1b = (civec[0, t1addrb] * t1signb).reshape(noccb, nvirb)

        print(c1a)
        print(c1b)
        nocca_comp = nocca * (nocca - 1) // 2
        noccb_comp = noccb * (noccb - 1) // 2
        nvira_comp = nvira * (nvira - 1) // 2
        nvirb_comp = nvirb * (nvirb - 1) // 2
        c2aa = (civec[t2addra, 0] * t2signa).reshape(nocca_comp, nvira_comp)
        c2bb = (civec[0, t2addrb] * t2signb).reshape(noccb_comp, nvirb_comp)
        c2aa = pyscf.cc.ccsd._unpack_4fold(c2aa, nocca, nvira)
        c2bb = pyscf.cc.ccsd._unpack_4fold(c2bb, noccb, nvirb)
        c2ab = np.einsum("i,j,ij->ij", t1signa, t1signb, civec[t1addra[:, None], t1addrb])
        c2ab = c2ab.reshape(nocca, nvira, noccb, nvirb).transpose(0, 2, 1, 3)

        c0 = civec[0][0]
        c1a *= 1/c0
        c1b *= 1/c0
        c2aa *= 1/c0
        c2ab *= 1/c0
        c2bb *= 1/c0

        c1 = (c1a, c1b)
        c2 = (c2aa, c2ab, c2bb)
     

        eri_aa = self.h2e[0] 
        eri_bb = self.h2e[2]
        eri_ab = self.h2e[1]


        # Lower case alpha, upper case beta
        oooo = eri_aa[:nocca,:nocca,:nocca,:nocca].copy()
        ovoo = eri_aa[:nocca,nocca:,:nocca,:nocca].copy()
        ovov = eri_aa[:nocca,nocca:,:nocca,nocca:].copy()
        oovv = eri_aa[:nocca,:nocca,nocca:,nocca:].copy()
        ovvo = eri_aa[:nocca,nocca:,nocca:,:nocca].copy()
        ovvv = eri_aa[:nocca,nocca:,nocca:,nocca:].copy()
        vvvv = eri_aa[nocca:,nocca:,nocca:,nocca:].copy()

        OOOO = eri_bb[:noccb,:noccb,:noccb,:noccb].copy()
        OVOO = eri_bb[:noccb,noccb:,:noccb,:noccb].copy()
        OVOV = eri_bb[:noccb,noccb:,:noccb,noccb:].copy()
        OOVV = eri_bb[:noccb,:noccb,noccb:,noccb:].copy()
        OVVO = eri_bb[:noccb,noccb:,noccb:,:noccb].copy()
        OVVV = eri_bb[:noccb,noccb:,noccb:,noccb:].copy()
        VVVV = eri_bb[noccb:,noccb:,noccb:,noccb:].copy()

        ooOO = eri_ab[:nocca,:nocca,:noccb,:noccb].copy()
        ovOO = eri_ab[:nocca,nocca:,:noccb,:noccb].copy()
        ovOV = eri_ab[:nocca,nocca:,:noccb,noccb:].copy()
        ooVV = eri_ab[:nocca,:nocca,noccb:,noccb:].copy()
        ovVO = eri_ab[:nocca,nocca:,noccb:,:noccb].copy()
        ovVV = eri_ab[:nocca,nocca:,noccb:,noccb:].copy()
        vvVV = eri_ab[nocca:,nocca:,noccb:,noccb:].copy()


        fova = self.h1e[0][:nocca,nocca:]
        fovb = self.h1e[1][:noccb,noccb:]


        


        gaa = ovov
        gab = ovOV
        gbb = OVOV
        return c0,c1,c2,t1signa,t1signb,t1addra,t1addrb,t2signa,t2signb,t2addra,t2addrb, nocca_comp,noccb_comp,nvira_comp,nvirb_comp, fova,fovb,gaa,gab,gbb

       
    def one_p_excitations(self,fermi_vac_str):
        '''
        assumes receiving a string of ones and zeros correspsonding to fermi vac, with a JW transform that puts one spin sector in first n/2 qubits and the other spin sector in the sector in the other n/2
        '''
        if ( len(fermi_vac_str) // 2 * 2 ) != len(fermi_vac_str):
            sys.exit("length of fermi vac string is odd")
        zeros_list_up=[]
        ones_list_up=[]
        zeros_list_down=[]
        ones_list_down=[]
        for n in range(int(len(fermi_vac_str))//int(2)):
            #print(n,type(n))
            if fermi_vac_str[n]=="0":
                zeros_list_up.append(n)
            elif fermi_vac_str[n]=="1":
                ones_list_up.append(n)
            else:
                sys.exit("fermi vac string contains something other than 0 or 1")
        for n in np.arange(int(len(fermi_vac_str))//int(2),len(fermi_vac_str)):
            #print(n,type(n))
            if fermi_vac_str[n]=="0":
                zeros_list_down.append(n)
            elif fermi_vac_str[n]=="1":
                ones_list_down.append(n)        
            else:
                sys.exit("fermi vac string contains something other than 0 or 1")    

        #print(zeros_list_up,zeros_list_down)        
        #print(ones_list_up,ones_list_down)
        unique_combinations_up = []
        for n in ones_list_up:
            for m in zeros_list_up:
                unique_combinations_up.append([n,m])
        #print(unique_combinations_up)
        unique_combinations_down = []
        for n in ones_list_down:
            for m in zeros_list_down:
                unique_combinations_down.append([n,m])            
        #print(unique_combinations_down)
        one_p_excitations=[]
        
        for n in unique_combinations_up+unique_combinations_down:
            fermi_vac_str_list=list(fermi_vac_str)
            fermi_vac_str_list[n[0]]="0"
            fermi_vac_str_list[n[1]]="1"
            one_p_excitations.append("".join(str(x) for x in fermi_vac_str_list))

        return one_p_excitations  



    def two_p_excitations(self,fermi_vac_str,restricted=True):
        '''
        assumes receiving a string of ones and zeros correspsonding to fermi vac, with a JW transform that puts one spin sector in first n/2 qubits and the other spin sector in the sector in the other n/2
        '''
        if ( len(fermi_vac_str) // 2 * 2 ) != len(fermi_vac_str):
            sys.exit("length of fermi vac string is odd")
        zeros_list_up=[]
        ones_list_up=[]
        zeros_list_down=[]
        ones_list_down=[]
        for n in range(int(len(fermi_vac_str))//int(2)):
            #print(n,type(n))
            if fermi_vac_str[n]=="0":
                zeros_list_up.append(n)
            elif fermi_vac_str[n]=="1":
                ones_list_up.append(n)
            else:
                sys.exit("fermi vac string contains something other than 0 or 1")
        for n in np.arange(int(len(fermi_vac_str))//int(2),len(fermi_vac_str)):
            #print(n,type(n))
            if fermi_vac_str[n]=="0":
                zeros_list_down.append(n)
            elif fermi_vac_str[n]=="1":
                ones_list_down.append(n)        
            else:
                sys.exit("fermi vac string contains something other than 0 or 1")    

        #print(zeros_list_up,zeros_list_down)        
        #print(ones_list_up,ones_list_down)
        unique_combinations_up = []
        for n in ones_list_up:
            for m in zeros_list_up:
                unique_combinations_up.append([n,m])
        #print(unique_combinations_up)
        unique_combinations_down = []
        for n in ones_list_down:
            for m in zeros_list_down:
                unique_combinations_down.append([n,m])            
        #print(unique_combinations_down)
        two_p_excitations=[]
        #print(unique_combinations_up+unique_combinations_down)
        if restricted==False:
            all_unique_combs=unique_combinations_up+unique_combinations_down
            for n in range(len(all_unique_combs)):
                for m in np.arange(n,len(all_unique_combs)):
                    fermi_vac_str_list=list(fermi_vac_str)
                    if all_unique_combs[n][0]!=all_unique_combs[m][0] and all_unique_combs[n][1]!=all_unique_combs[m][1]:
                        fermi_vac_str_list[all_unique_combs[n][0]]="0"
                        fermi_vac_str_list[all_unique_combs[n][1]]="1"
                        fermi_vac_str_list[all_unique_combs[m][0]]="0"
                        fermi_vac_str_list[all_unique_combs[m][1]]="1"
                        two_p_excitations.append("".join(str(x) for x in fermi_vac_str_list))
        else:
            #only need aabb 2 particle excitations
            for n in range(len(unique_combinations_down)):
                for m in range(len(unique_combinations_up)):
                    fermi_vac_str_list=list(fermi_vac_str)
                    #if all_unique_combs[n][0]!=all_unique_combs[m][0] and all_unique_combs[n][1]!=all_unique_combs[m][1]:
                    fermi_vac_str_list[unique_combinations_down[n][0]]="0"
                    fermi_vac_str_list[unique_combinations_down[n][1]]="1"
                    fermi_vac_str_list[unique_combinations_up[m][0]]="0"
                    fermi_vac_str_list[unique_combinations_up[m][1]]="1"
                    two_p_excitations.append("".join(str(x) for x in fermi_vac_str_list))

        return two_p_excitations  

    def get_c1_c2_calc(self,class_c1,class_c2):

        c1_indices=[]
        c1_labels=[]
        c1_qubit_indices=[]
        c2_indices=[]
        c2_labels=[]
        c2_qubit_indices=[]

        labels=[str(bin(i))[2:] for i in range(2**self.num_qubits)]
        #print(labels)
        #print(labels[1])
        for i in range(len(labels)):
            if len(labels[i])<self.num_qubits:
                labels[i]="0"*(self.num_qubits-len(labels[i]))+labels[i]
        #print(labels)


        
        
        # if len(list(self.SV_basis_0.to_dict().keys()))!=1:
        #     sys.exit("initial state is not a cbs")


        ###make c1s
        fermi_vac=self.SV_basis_0
        o_exc=self.one_p_excitations(fermi_vac)
        fermi_vac_split=["".join(g) for k, g in itertools.groupby(fermi_vac)]


        self.c1_len=len(o_exc[:len(o_exc)//2])
        for one_p in o_exc[:len(o_exc)//2]:
            exc_from=[]
            exc_to=[]
            qubit_labels=[]
            for m in range(len(one_p)//2):
                if one_p[m]=="0" and fermi_vac[m]=="1":
                    exc_from.append(len(one_p)-1-m-len(one_p)//2)
                    qubit_labels.append(m)
                if one_p[m]=="1" and fermi_vac[m]=="0":
                    exc_to.append(len(one_p)-1-m-len(one_p)//2 - len(fermi_vac_split[1]))
                    qubit_labels.append(m)
            


            if abs(class_c1[exc_from[0]][exc_to[0]]) > 2e-6:# we are using classically calculated c1 coeffs to determine which we need to calculate and which are symmetry forbidden
                c1_indices.append([exc_from[0],exc_to[0]])
                c1_labels.append(one_p)
                c1_qubit_indices.append(qubit_labels)


        fermi_vac=self.SV_basis_0
        t_exc=self.two_p_excitations(fermi_vac)
        self.c2_len=len(t_exc)
        for two_p in t_exc:
            
            exc_from=[]
            exc_to=[]
            qubit_labels=[]
            for m in range(len(two_p)//2):
                if two_p[m]=="0" and fermi_vac[m]=="1":
                    exc_from.append(len(two_p)-1-m-len(two_p)//2)
                    qubit_labels.append(m)
                if two_p[m]=="1" and fermi_vac[m]=="0":
                    exc_to.append(len(two_p)-1-m-len(two_p)//2-len(fermi_vac_split[1]))
                    qubit_labels.append(m)
            for m in np.arange(len(two_p)//2,len(two_p)):
                if two_p[m]=="0" and fermi_vac[m]=="1":
                    exc_from.append(len(two_p)-1-m)
                    qubit_labels.append(m)
                if two_p[m]=="1" and fermi_vac[m]=="0":
                    exc_to.append(len(two_p)-1-m-len(fermi_vac_split[3]))   
                    qubit_labels.append(m)         


            if abs(class_c2[exc_from[0]][exc_from[1]][exc_to[0]][exc_to[1]]) > 2e-6: # we are using classically calculated c2 coeffs to determine which we need to calculate and which are symmetry forbidden
                c2_labels.append(two_p)
                c2_indices.append([exc_from[0],exc_from[1],exc_to[0],exc_to[1]])
                c2_qubit_indices.append(qubit_labels)
        return c1_indices,c1_labels,c1_qubit_indices,c2_indices,c2_labels,c2_qubit_indices,fermi_vac, fermi_vac_split
    

    def cbssp_unitary(self,stateA,stateB,imag=False):
        '''
        create superposition (sp) of computaional basis states (cbs), given by strings of 1s and 0s
        returns qiskit circuit creating the sp
        '''

        both1=[]
        a1=[]
        b1=[]
        if len(stateA) != len(stateB):
            sys.exit("states diff lengths")
        for n in range(len(stateA)):
            if stateA[n] =="1" and stateB[n]=="1":
                both1.append(len(stateA)-1-n)
            if stateA[n] =="1" and stateB[n]=="0":
                a1.append(len(stateA)-1-n)    
            if stateA[n] =="0" and stateB[n]=="1":
                b1.append(len(stateA)-1-n)    
        # print(both1)        
        # print(a1)
        # print(b1)
        sp_circ=qiskit.QuantumCircuit(len(stateA))
        for n in both1:
            sp_circ.x(n)
        if len(a1)!=0:
            sp_circ.h(a1[0])
            if imag:
                sp_circ.s(a1[0])
            for n in np.arange(1,len(a1[:])):
                #print(n,a1[0],a1[n])
                sp_circ.cnot(a1[0],a1[n])

            sp_circ.x(a1[0])
            for n in np.arange(0,len(b1[:])):
                #print(n)
                sp_circ.cnot(a1[0],b1[n])    
            sp_circ.x(a1[0])
        elif len(a1)==0 and len(b1)!=0:
            sp_circ.h(b1[0])
            if imag:
                sp_circ.s(b1[0])
            for n in np.arange(1,len(b1[:])):
                sp_circ.cnot(b1[0],b1[n])
        return sp_circ,a1[0]    
    


if __name__ == "__main__":
    mol = gto.M(
        #atom = 'H 0 0 0; H 0 0 1.0; H 0 0 2.0; H 0 0 3.0',  # in Angstrom
        #atom = 'H 0 0 0; H 0 0 2.',#; H 0 0 4.0; H 0 0 6.0',  # in Angstrom
        #atom = 'H 0 0 0; H 0 0 1.5; H 0 0 3.; H 0 0 4.5; H 0 0 6.; H 0 0 7.5',  # in Angstrom
        atom = 'H 0 0 0; H 0 0 2.',#; H 0 0 4.0; H 0 0 6.0',  # in Angstrom
        basis = 'sto-3g',
        symmetry = True,
        verbose = 4
    )
    nelec = mol.nelec
    myhf = mol.RHF().run()
    assert(myhf.converged)


    ci = pyscf.fci.FCI(myhf)
    ci.kernel()
    #print("nuclear repulsion: ",mol.energy_nuc()) 
    print("hf energy: ",myhf.e_tot)
    print('FCI energy: ',ci.e_tot,myhf.e_tot,mol.energy_nuc())  

    norb = myhf.mo_coeff.shape[1]
    nelec = mol.nelectron
    nocc = nelec // 2
    nvir = norb - nocc


    # Extract one and two-body terms of the hamiltonian, and transform them
    # into the MO basis
    h1e = np.linalg.multi_dot((myhf.mo_coeff.T, myhf.get_hcore(), myhf.mo_coeff))
    eri = ao2mo.incore.general(myhf._eri, (myhf.mo_coeff,)*4, compact=False)
    # Remove all of the permutational symmetry that is used to compress the integral array
    eri = ao2mo.restore(1, eri, norb)    

    solver=cluster_solver(h1e, eri, norb, (nocc,nocc))
    solver.solve()
    sv_E,sv_ci=solver.measure_state_vector()
    print(sv_E,sv_ci)
    # print(solver.qubit_ham)

    # fci_corr_e=ci.e_tot - myhf.e_tot - mol.energy_nuc()
    Nshots=1e6
    # print(solver.measure_paulis(Nshots,"all") - myhf.e_tot)
    # print(solver.measure_paulis(Nshots,"QBW")- myhf.e_tot)  
    # print(solver.measure_classical_shadows_pauli(Nshots,3) - myhf.e_tot)

    # print(solver.measure_classical_shadows(Nshots,3)[0]- mol.energy_nuc())
    # print(solver.measure_super_pos(Nshots) - mol.energy_nuc())
    meas_d_E,meas_d_ci=solver.measure_diag(Nshots)
    print(meas_d_E,meas_d_E - mol.energy_nuc(),meas_d_ci)
    

    #print("FCI corr energy: ",fci_corr_e)


def excitations_tuples_to_strings(p_exc,fermi_vac):
        p_exc_list =[]
        for exc_det_count in range(len(p_exc)):
            # print(p_exc[exc_det_count])
            # print(fermi_vac)
            fermi_vac_exc=fermi_vac
            for n_exc in range(len(p_exc[exc_det_count][0])):
                index1 = -1-p_exc[exc_det_count][0][n_exc]
                index2 = -1-p_exc[exc_det_count][1][n_exc]
                # print(fermi_vac_exc[index1])
                if index1==-1:
                    s = fermi_vac_exc[:index1] + "0"
                else:
                    s = fermi_vac_exc[:index1] + "0" + fermi_vac_exc[index1 + 1:]    

                if index2==-1:    
                    s2 = s[:index2] + "1"
                else:
                    s2 = s[:index2] + "1" + s[index2 + 1:]        
                # print(s)
                # print(s2)
                fermi_vac_exc = s2
            # print(fermi_vac_exc)   
            p_exc_list.append(fermi_vac_exc) 
        return p_exc_list