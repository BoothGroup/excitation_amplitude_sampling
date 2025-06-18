### comparison of different energy expectation values (Pauli grouping, mixed estimator using classical shadows and direct evaluation) with a ground state calculated with a FCI calculation

from qiskit_nature.second_q.operators import ElectronicIntegrals, FermionicOp ,SparseLabelOp, PolynomialTensor, tensor_ordering
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
#from qiskit_nature.second_q.circuit.library import UCCSD

########### UCC modules #######
#from qiskit_nature.second_q.circuit import QubitConverter
from qiskit_nature.second_q.circuit.library.ansatzes import UCC
from qiskit_nature.second_q.circuit.library.initial_states import HartreeFock
from qiskit_nature.second_q.circuit.library.ansatzes.utils import generate_fermionic_excitations 


########### VQE Modules #######
# from qiskit.utils import algorithm_globals
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import UnitaryGate

########### OLD VQE Modules #######
from qiskit_algorithms import VQE

# from qiskit import Aer


######### Optimiser Modules #########
from qiskit_algorithms.optimizers import COBYLA , SLSQP , L_BFGS_B


# from qiskit.opflow import StateFn, PauliExpectation, CircuitSampler
# from qiskit.opflow import CircuitStateFn, PauliSumOp
from qiskit.quantum_info import SparsePauliOp
# from qiskit.utils import algorithm_globals
# from qiskit.utils import QuantumInstance
# from qiskit import Aer
from qiskit_aer import AerSimulator

from qiskit.circuit.library import NLocal,EfficientSU2
######### PySCF + others #########

import sys
sys.argv.append("--quiet")  # For Vayesta
import numpy as np
from pyscf import gto, scf, lib, ao2mo
import itertools
import math
import array_to_latex as a2l
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
import pandas as pd 
#import matplotlib.pyplot as plt
from qiskit.quantum_info.operators import Operator
import qiskit
from pyscf import cc as cupclus

from src import classical_shadow
from src import swap_test
from src import observables
import cmath
import pyscf
from pyscf import gto, scf, ao2mo, fci, ci
import os.path
from pyscf.fci import cistring
from itertools import permutations




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



def one_p_excitations(fermi_vac_str):
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



def two_p_excitations(fermi_vac_str,restricted=True):
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

run_options = {"method": "statevector"}
estimator=AerEstimator(run_options=run_options)

nsite=int(sys.argv[1])
Nshots=int(float(sys.argv[2]))

num_qubits=nsite*2

c0_list=[]
E_frac_err_CS=[]
corr_E_frac_err_CS=[]
vqe_E_frac_err_CS=[]
corr_vqe_E_frac_err_CS=[]

E_frac_err_meas=[]
corr_E_frac_err_meas=[]
vqe_E_frac_err_meas=[]
corr_vqe_E_frac_err_meas=[]

E_frac_err_meas_diag=[]
corr_E_frac_err_meas_diag=[]
vqe_E_frac_err_meas_diag=[]
corr_vqe_E_frac_err_meas_diag=[]

E_frac_err_paulis_all_comm=[]
corr_E_frac_err_paulis_all_comm=[]
vqe_E_frac_err_paulis_all_comm=[]
corr_vqe_E_frac_err_paulis_all_comm=[]

E_frac_err_paulis_qb_comm=[]
corr_E_frac_err_paulis_qb_comm=[]
vqe_E_frac_err_paulis_qb_comm=[]
corr_vqe_E_frac_err_paulis_qb_comm=[]


E_frac_err_CS_pauli=[]
corr_E_frac_err_CS_pauli=[]
vqe_E_frac_err_CS_pauli=[]
corr_vqe_E_frac_err_CS_pauli=[]




for d in list(np.arange(float(sys.argv[3]),float(sys.argv[4]),float(sys.argv[5]))):#+[12.,16.]:
    my_atom=[("H 0. 0. %f" % xyz) for xyz in [d*x for x in list(range(nsite))]]
    '''

    Setting up two site Hubbard model

    '''

    num_e = nsite




    mol = gto.M(
        #atom = 'H 0 0 0; H 0 0 1.0; H 0 0 2.0; H 0 0 3.0',  # in Angstrom
        #atom = 'H 0 0 0; H 0 0 2.',#; H 0 0 4.0; H 0 0 6.0',  # in Angstrom
        #atom = 'H 0 0 0; H 0 0 1.55; H 0 0 3.1; H 0 0 4.65',  # in Angstrom
        #atom = 'H 0 0 0; H 0 0 2.',#; H 0 0 4.0; H 0 0 6.0',  # in Angstrom
        atom=my_atom,
        basis = 'sto-3g',
        symmetry = True,
        verbose = 3
    )
    nelec = mol.nelec
    myhf = mol.RHF().run()
    assert(myhf.converged)




    cc = cupclus.CCSD(myhf)
    cc.kernel()
    print('CCSD total energy: {}'.format(cc.e_tot))
    CCSD_energy = cc.e_tot

    # ci = pyscf.fci.FCI(myhf)
    # ci.kernel()
    # print('FCI total energy: {}'.format(ci.e_tot))

    

    norb = myhf.mo_coeff.shape[1]
    nelec = mol.nelectron
    nocc = nelec // 2
    nvir = norb - nocc
    
    h1e = np.linalg.multi_dot((myhf.mo_coeff.T, myhf.get_hcore(), myhf.mo_coeff))
    eri = ao2mo.incore.general(myhf._eri, (myhf.mo_coeff,)*4, compact=False)
    # Remove all of the permutational symmetry that is used to compress the integral array
    eri = ao2mo.restore(1, eri, norb)
    cisolver = fci.direct_spin0.FCI()
    e1, civec = cisolver.kernel(h1e, eri, norb, nelec)
    e_fci = e1 + mol.energy_nuc()


    # fermi_vac=list(SV_basis_0_dict.keys())[0]
    fermi_vac = "0"*nvir + "1"*nocc + "0"*nvir + "1"*nocc

    a_perms = list(set([''.join(p) for p in permutations(fermi_vac[len(fermi_vac)//2:])]))
    b_perms = list(set([''.join(p) for p in permutations(fermi_vac[:len(fermi_vac)//2])]))
    SV_exact=np.zeros(2**(norb+norb))
    all_addr = [y+x for x in a_perms for y in b_perms]
    for addr_string in all_addr:
        addr = (cistring.str2addr(norb, nocc, bin(int(addr_string[len(addr_string)//2:],2))),cistring.str2addr(norb, nocc, bin(int(addr_string[:len(addr_string)//2],2))))
        if abs(civec[addr])>1.e-10:
            # if abs(SV_exact[int(addr_string,2)]-civec[addr])>1e-4:
            #     print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # print(SV_exact[int(addr_string,2)],civec[addr])
            SV_exact[int(addr_string,2)] = civec[addr]



        
    vqe_energy = e_fci

    print("VQE energy: ",vqe_energy)





    backend =backend = AerSimulator(method = 'statevector') #Aer.get_backend('statevector_simulator')

    one_body = myhf.mo_coeff.T @ myhf.get_hcore() @ myhf.mo_coeff
    qiskit_eri = ao2mo.kernel(myhf._eri, (myhf.mo_coeff, myhf.mo_coeff,myhf.mo_coeff,myhf.mo_coeff), compact=False)
    two_body = qiskit_eri.reshape((myhf.mo_coeff.shape[-1],) * 4)
    
    # Constructing the electronic hamiltonian in second quantised representation
    integrals = ElectronicIntegrals.from_raw_integrals(h1_a=one_body, h1_b = one_body, h2_aa=two_body, h2_bb= two_body, h2_ba= two_body , auto_index_order=True) 
    
    # Defining the many body electronic hamiltionian in second quantised representation
    nuclear_repulsion_energy = myhf.energy_nuc()
    h_elec = ElectronicEnergy(integrals, constants = {'nuclear_repulsion_energy':nuclear_repulsion_energy}).second_q_op()       
    mapper = JordanWignerMapper()
    qubit_ham = mapper.map(h_elec) 
    

    # create an FCI solver based on the SCF object (enforcing singlet symmetry)
    
    print('E(FCI) = %.12f' % e_fci)
    print('Size of FCI wave function: ', civec.shape)
    # Note that the wave function is 2D. This is because it is a matrix of dimension 'alpha excitations' x 'beta excitations'

    # Now, extract the C0, C1 and C2 coefficients of the HF, singles and doubles excitations
    c0 = civec[0,0]
    print('Coefficient of FCI wave function on Hartree--Fock configuration: ',c0)

    # === C1 amplitudes ===
    # Note that we only consider alpha -> alpha
    # or beta -> beta excitations, and in a spin-restricted system, then these *should*
    # be the same.
    # These functions extract out the indicies and permutational signs of
    # the *same spin* excitations of a given rank from the FCI vector
    t1addr, t1sign = ci.cisd.tn_addrs_signs(norb, nocc, 1)
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
    print('Singles contribution to projected correlation energy: ',e_singles)

    # Two-electron contribution to correlation energy
    # Get (ov|vo) integral slice in MO basis:
    g_ovvo = eri[:nocc,nocc:,nocc:,:nocc]
    g_ovvo_T = g_ovvo.transpose(0, 2, 1, 3)
    #print("g_ovvo: ",g_ovvo)
    e_doubles = 2 * np.einsum("ijab,iabj->", c2, g_ovvo) - np.einsum("ijab,ibaj->", c2, g_ovvo)
    print('Doubles contribution to projected correlation energy: ',e_doubles)

    e_proj = myhf.e_tot + e_singles + e_doubles
    print('Projected correlation energy + MF energy = ',e_proj)
    assert math.isclose(e_proj, e_fci,rel_tol=1e-5),"e_proj: "+str(e_proj)+" e_fci: "+str(e_fci)
    print('Projected energy and FCI energy are the same')

    print("c0: ",c0)
    # print("c1: ",c1)
    # print("c2: ",c2)



    c0_list.append(c0)


    



    # SV_basis_0=Statevector.from_int(int(fermi_vac,2),2**num_qubits)
    # SV_basis_0_dict=SV_basis_0.to_dict()
    # SV_basis_0=0.
    # if len(list(SV_basis_0_dict.keys()))!=1:
    #     sys.exit("initial state is not a cbs")





    
    fermi_vac_split=["".join(g) for k, g in itertools.groupby(fermi_vac)]
    







    ### Classical shadow computation of energy  ########################################################
    ####################################################################################################
    Shadow = classical_shadow.Classical_shadow("Clifford",num_qubits,[],[])


    Shadow.calc_shadow_lists(Statevector(1/np.sqrt(2)*(np.array(qiskit.quantum_info.Statevector.from_label("0"*num_qubits)) + np.array(SV_exact))),3,int(Nshots/3))


    index_a=0#"0"*SV_basis_0.num_qubits#int(list(SV_basis_0.to_dict().keys())[0],2)

    ###index for HF state
    index_b=int(fermi_vac,2)
    op_list=[]
    op_list.append([1.,index_a,index_b])
    results_l,errors_l = Shadow.evaluate_operator_overlap_stabilizer_perp(op_list)
    # print(results_l)
    c0_cs=results_l[0]

    a_exc = generate_fermionic_excitations(num_excitations=1, num_spatial_orbitals=norb, num_particles=mol.nelec, alpha_spin=True, beta_spin=False, max_spin_excitation=None, generalized=False, preserve_spin=True)
    a_exc_strings=excitations_tuples_to_strings(a_exc,fermi_vac)
    op_list=[]
    for a in a_exc_strings:
        index_a=0#"0"*SV_basis_0.num_qubits#int(list(SV_basis_0.to_dict().keys())[0],2)
        index_b=int(a,2)
        op_list.append([1.,index_a,index_b])
    results_l=[]

    results_l,errors_l = Shadow.evaluate_operator_overlap_stabilizer_perp(op_list)
    a_int_norm_results = [x for x in results_l]
    # print(a_int_norm_results)
    t1addr,t1sign = pyscf.ci.cisd.tn_addrs_signs(norb, nocc, 1)
    c1_cs_rs = (a_int_norm_results * t1sign).reshape(nocc, nvir)
    # print(c1_cs_rs)

    ab_exc = generate_fermionic_excitations(num_excitations=2, num_spatial_orbitals=norb, num_particles=mol.nelec, alpha_spin=True, beta_spin=True, max_spin_excitation=1, generalized=False, preserve_spin=True)
    # ab_exc = generate_fermionic_excitations(num_excitations=2, max_spin_excitation=1, num_spatial_orbitals=norb, num_particles=nelec, alpha_spin=True, beta_spin=True, generalized=False, preserve_spin=True)
    ab_exc_strings=excitations_tuples_to_strings(ab_exc,fermi_vac)
    op_list=[]
    for ab in ab_exc_strings:
        index_a=0#"0"*SV_basis_0.num_qubits#int(list(SV_basis_0.to_dict().keys())[0],2)
        index_b=int(ab,2)
        op_list.append([1.,index_a,index_b])

    results_l,errors_l = Shadow.evaluate_operator_overlap_stabilizer_perp(op_list)
    ab_int_norm_results = [x for x in results_l]
    # print(ab_int_norm_results)
    c2_cs_ab = np.einsum("i,j,ij->ij", t1sign, t1sign, np.reshape(ab_int_norm_results,(len(t1addr),len(t1addr))))
    c2_cs_ab = np.reshape(c2_cs_ab,(nocc, nvir, nocc, nvir)).transpose(0, 2, 1, 3)





    e_singles = 2 * np.sum(f_ov * c1_cs_rs/c0_cs)
    print('Singles shadow contribution to projected correlation energy: ',e_singles)

    # Two-electron contribution to correlation energy
    e_doubles = 2 * np.einsum("ijab,iabj->", c2_cs_ab/c0_cs, g_ovvo) - np.einsum("ijab,ibaj->", c2_cs_ab/c0_cs, g_ovvo)
    print('Doubles shadow contribution to projected correlation energy: ',e_doubles)

    e_proj_CS_perp = myhf.e_tot + e_singles + e_doubles









    ### direct measurements of overlaps biased using classical coupled cluster###
    #############################################################################
    t1=cc.t1
    t2=cc.t2
    c0_cc=1.0
    c1_cc = c0_cc * t1
    c2_cc = c0_cc * (t2 + np.einsum("ia,jb->ijab", t1, t1))


    c0_len=1
    c1_len=0
    c2_len=0
    c1_indices=[]
    c1_labels=[]
    c1_qubit_indices=[]
    c2_indices=[]
    c2_labels=[]
    c2_qubit_indices=[]

    labels=[str(bin(i))[2:] for i in range(2**num_qubits)]
    #print(labels)
    #print(labels[1])
    for i in range(len(labels)):
        if len(labels[i])<num_qubits:
            labels[i]="0"*(num_qubits-len(labels[i]))+labels[i]
    #print(labels)


    o_exc=one_p_excitations(fermi_vac)
    c1_len=len(o_exc[:len(o_exc)//2])
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
        #print("one_p: ",one_p)
        SV_basis_1=qiskit.quantum_info.Statevector.from_label(one_p)
        # P_i = np.outer(np.array(SV_basis_0),np.array(SV_basis_1))
        # P_p= 0.5*(P_i + P_i.transpose().conjugate())
        # op_list.append(P_p)
        if abs(c1_cc[exc_from[0]][exc_to[0]]) > 2e-6:
            c1_indices.append([exc_from[0],exc_to[0]])
            c1_labels.append(one_p)
            c1_qubit_indices.append(qubit_labels)


    # fermi_vac=list(SV_basis_0_dict.keys())[0]
    t_exc=two_p_excitations(fermi_vac)#list(SV_basis_0.to_dict().keys())[0])
    c2_len=len(t_exc)
    for two_p in t_exc:
        #print(two_p[:len(two_p)//2],two_p[len(two_p)//2:])
        #print(fermi_vac[:len(fermi_vac)//2],fermi_vac[len(fermi_vac)//2:])
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

        #print(exc_from,exc_to,"\n")        
    ###make c2s
    #for two_p in two_p_excitations(list(SV_basis_0.to_dict().keys())[0]):
        # SV_basis_1=qiskit.quantum_info.Statevector.from_label(two_p)
        # P_i = np.outer(np.array(SV_basis_0),np.array(SV_basis_1))
        # P_p= 0.5*(P_i + P_i.transpose().conjugate())
        # op_list.append(P_p)
        if abs(c2_cc[exc_from[0]][exc_from[1]][exc_to[0]][exc_to[1]]) > 2e-6:
            #c1_indices.append([exc_from[0],exc_to[0]])
            c2_labels.append(two_p)
            c2_indices.append([exc_from[0],exc_from[1],exc_to[0],exc_to[1]])
            c2_qubit_indices.append(qubit_labels)



    ab_exc = generate_fermionic_excitations(num_excitations=2, num_spatial_orbitals=norb, num_particles=mol.nelec, alpha_spin=True, beta_spin=True, max_spin_excitation=1, generalized=False, preserve_spin=True)
    print("number ab exc: ",len(ab_exc),len(c2_labels))


    c0_cc_est=np.sqrt(1/(1+(np.sum(np.square(np.array(c2_cc)))+np.sum(np.square(np.array(c1_cc))))))
    h_ijab = (2.*g_ovvo - g_ovvo_T).transpose(0,3,1,2)
    h_ijab_square = np.square(h_ijab)
    numerator=2 * np.einsum("ijab,iabj->", c2_cc, g_ovvo) - np.einsum("ijab,ibaj->", c2_cc, g_ovvo)
    # c2_cc_square = np.square(np.array(c2_cc*c0_cc_est))*0.25 + 0.5
    c2_cc_square = (np.square(np.array(c2_cc)*c0_cc_est)+c0_cc_est**2)*0.25 - np.square(np.array(c2_cc)*(c0_cc_est**2))


    # delta_A=np.einsum("ijab,ijab->", c2_cc_square, h_ijab_square)
    delta_A=0
    for m_c2 in range(len(c2_labels)):
        delta_A+=h_ijab_square[c2_indices[m_c2][0],c2_indices[m_c2][1],c2_indices[m_c2][2],c2_indices[m_c2][3]] * c2_cc_square[c2_indices[m_c2][0],c2_indices[m_c2][1],c2_indices[m_c2][2],c2_indices[m_c2][3]]
    Ldelta_A = delta_A*len(c2_labels)
    delta_B=c0_cc_est**2 - c0_cc_est**4
    # shots_ratio = 1 + np.sqrt(( (numerator**2/c0_cc_est**2) - 1) * (delta_B**2/delta_A**2))
    shots_ratio = 1 + np.sqrt(( (numerator**2)) * (delta_B/Ldelta_A))

    if cc.converged:
        Nshots_c2 = int(Nshots//shots_ratio)
        Nshots_c0 = Nshots - Nshots_c2
        Nshots_c2=int(Nshots_c2//len(c2_labels))

    else:
        Nshots_c0 = int(Nshots//(len(c2_labels)+1))
        Nshots_c2 = int(Nshots//(len(c2_labels)+1))


    basis_measure_circ=qiskit.QuantumCircuit(num_qubits)
    basis_measure_circ.initialize(SV_exact)
    basis_measure_circ.measure_all()

    c0_meas_diag=0
    while c0_meas_diag==0:
        basis_result = backend.run(basis_measure_circ,shots=Nshots_c0).result()
        if fermi_vac in basis_result.get_counts():
            c0_meas_diag=np.sqrt(basis_result.get_counts()[fermi_vac]/Nshots_c0)
        else:
            c0_meas_diag=0  
        

    

    # print(c0_meas_diag,c0)


    c2_meas_diag=np.zeros((len(fermi_vac_split[0]),len(fermi_vac_split[1]),len(fermi_vac_split[2]),len(fermi_vac_split[3])))

    sigma_plus=np.ndarray((2,2),buffer=np.array([0.,0.,1.,0.]))
    sigma_minus=np.ndarray((2,2),buffer=np.array([0.,1.,0.,0.]))
    op_matrix=np.array(1)
    op_matrix=np.kron(op_matrix,sigma_plus)
    op_matrix=np.kron(op_matrix,sigma_minus)
    op_matrix=np.kron(op_matrix,sigma_plus)
    op_matrix=np.kron(op_matrix,sigma_minus)
    op_matrix=0.5*(op_matrix+op_matrix.conjugate().transpose())
    eig_vals,eig_vecs=np.linalg.eig(op_matrix)

    for n2_count in range(len(c2_labels)):

        if Nshots_c2 != 0:
            basis_measure_circ=qiskit.QuantumCircuit(num_e*2)
            basis_measure_circ.initialize(SV_exact)
            basis_change_U=UnitaryGate(eig_vecs.conjugate().transpose())
            basis_measure_circ.unitary(basis_change_U,[num_qubits-1-x for x in c2_qubit_indices[n2_count]])#may need to reverse qubit ordering
            basis_measure_circ.measure_all()
            basis_result = backend.run(basis_measure_circ,shots=Nshots_c2).result()
            counts=basis_result.get_counts()
            exp_val=0
            for m in counts:
                untouched_states_correct=True #check if the ones we meaure in basis have the correct values (same as in vacuum state)
                for m2 in range(len(m)):
                    if m2 not in c2_qubit_indices[n2_count] and m[m2] != fermi_vac[m2]:
                        untouched_states_correct=False
                if untouched_states_correct:
                    diag_subspace_measurements=[m[x] for x in c2_qubit_indices[n2_count]]
                    diag_subspace_measurements.reverse()
                    # print(diag_subspace_measurements,int("".join(str(y) for y in diag_subspace_measurements), 2),counts[m])
                    exp_val+=counts[m]*eig_vals[int("".join(str(y) for y in diag_subspace_measurements), 2)]
            exp_val=exp_val/Nshots_c2
            # print(exp_val)  
            c2_meas_diag[c2_indices[n2_count][0]][c2_indices[n2_count][1]][c2_indices[n2_count][2]][c2_indices[n2_count][3]]=exp_val/(c0_meas_diag**2)      
        else:
            c2_meas_diag[c2_indices[n2_count][0]][c2_indices[n2_count][1]][c2_indices[n2_count][2]][c2_indices[n2_count][3]]=0.0        




    # print(c2)
    c2_meas_diag_rs=c2_meas_diag.transpose(0, 2, 1, 3)
    c2_meas_diag_rs = c2_meas_diag_rs.reshape((nocc*nvir, nocc*nvir))
    c2_meas_diag_rs = np.einsum("i,j,ij->ij", t1sign, t1sign, c2_meas_diag_rs)
    c2_meas_diag_rs = c2_meas_diag_rs.reshape((nocc, nvir, nocc, nvir))
    c2_meas_diag_rs = c2_meas_diag_rs.transpose(0, 2, 1, 3)
    # print(c2_meas_diag_rs)
    # print((c2-c2_meas_diag_rs)/c2)

    #e_singles = 2 * np.sum(f_ov * c1_meas_rs)
    e_doubles = 2 * np.einsum("ijab,iabj->", c2_meas_diag_rs, g_ovvo) - np.einsum("ijab,ibaj->", c2_meas_diag_rs, g_ovvo)
    e_proj_meas_diag = myhf.e_tot + e_doubles
    print("MF energy: ",myhf.e_tot)
    print('E(FCI) = %.12f' % e_fci)
    print('Projected correlation energy + MF energy = ',e_proj_meas_diag)

    print(abs(e_proj_meas_diag-e_fci)/abs(e_fci))
    print(abs(e_proj_meas_diag-e_fci)/abs(e_fci-myhf.e_tot))
    print(abs(e_proj_meas_diag-vqe_energy)/abs(vqe_energy))






    ### measure groups of commuting paulis with QWC commuting #####
    ###############################################################
    new_circuit=qiskit.QuantumCircuit(num_e*2)
    new_circuit.initialize(SV_exact)
    # new_circuit.measure_all()
    pauli_groups=qubit_ham.group_commuting(qubit_wise=True)
    shots_per_group=Nshots//len(pauli_groups)
    print(len(pauli_groups),shots_per_group,(shots_per_group*len(pauli_groups)))
    result_Pauli_qb_comm=nuclear_repulsion_energy
    if len(pauli_groups)>0 and shots_per_group!=0:
        for n in pauli_groups:
            job=estimator.run(new_circuit,n,shots=shots_per_group)
            #print(job.result().values)
            result_Pauli_qb_comm+=job.result().values[0]





    ### grouping all possible commuting (with heurisitc algorithm) ####
    ###################################################################

    pauli_groups=qubit_ham.group_commuting()
    shots_per_group=Nshots//len(pauli_groups)
    print(len(pauli_groups),shots_per_group,(shots_per_group*len(pauli_groups)))
    result_Pauli_all_comm=nuclear_repulsion_energy
    if len(pauli_groups)>0 and shots_per_group!=0:
        for n in pauli_groups:
            job=estimator.run(new_circuit,n,shots=shots_per_group)
            #print(job.result().values)
            result_Pauli_all_comm+=job.result().values[0]



    f_name="results_comp_Hydrogen1D_FCI_SV_nsite_"+str(nsite)+"_nshots_"+str(int(Nshots))+"_d_"+str(d)+"_fixed_runs.dat"
    if os.path.isfile(f_name):
        f = open(f_name, "a")
        f.write(str(c0)+"    "+str(e_fci)+"    "+str(myhf.e_tot)+"    "+str(vqe_energy)+"    "+str(e_proj_CS_perp)+"    "+str(e_proj_meas_diag)+"    "+str(result_Pauli_all_comm)+"    "+str(result_Pauli_qb_comm)+"    "+str(CCSD_energy)+"\n")
        # f.write(str(c0)+"    "+str(e_fci)+"    "+str(myhf.e_tot)+"    "+str(vqe_energy)+"    "+str(e_proj_CS)+"    "+str(e_proj_meas_diag)+"    "+str(result_Pauli_all_comm)+"\n")
        f.close()
    else:
        f = open(f_name, "a")
        # f.write("c0"+"    "+"e_fci"+"    "+"e_hf"+"    "+"vqe_energy"+"    "+"HF+CS"+"    "+"HF+meas_diag"+"    "+"E_pauli_all"+"\n")
        # f.write(str(c0)+"    "+str(e_fci)+"    "+str(myhf.e_tot)+"    "+str(vqe_energy)+"    "+str(e_proj_CS)+"    "+str(e_proj_meas_diag)+"    "+str(result_Pauli_all_comm)+"\n")
        f.write("c0"+"    "+"e_fci"+"    "+"e_hf"+"    "+"ansatz_energy"+"    "+"HF+CS"+"    "+"HF+meas_diag"+"    "+"pauli_all"+"    "+"pauli_qbw"+"    "+"CCSD"+"\n")
        f.write(str(c0)+"    "+str(e_fci)+"    "+str(myhf.e_tot)+"    "+str(vqe_energy)+"    "+str(e_proj_CS_perp)+"    "+str(e_proj_meas_diag)+"    "+str(result_Pauli_all_comm)+"    "+str(result_Pauli_qb_comm)+"    "+str(CCSD_energy)+"\n")
        f.close()



