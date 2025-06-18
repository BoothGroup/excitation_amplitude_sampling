### data collection for properties via mapping to CCSD ansatz

import sys
sys.argv.append("--quiet")  # For Vayesta
import numpy as np
import vayesta
# from vayesta.lattmod import Hubbard1D, LatticeRHF
# import ewf
from vayesta import ewf
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


import cmath
import pyscf
from pyscf import gto, scf, ao2mo, fci, ci
import os.path

import ebcc
from ebcc import REBCC, Space, GEBCC

from src import classical_shadow, observables, cluster_solver

from qiskit_nature.second_q.operators import ElectronicIntegrals, FermionicOp ,SparseLabelOp, PolynomialTensor, tensor_ordering
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.circuit.library.ansatzes.utils import generate_fermionic_excitations 

from pyscf.fci import cistring


from ebcc.ext.fci import _coefficients_to_amplitudes_restricted, _ci_vector_to_coefficients_restricted
from typing import Iterator, Union
from numpy.typing import NDArray
from numpy import floating, integer
# from ebcc import util


import copy


from pyscf.fci import cistring
from itertools import permutations



from qiskit_aer import AerSimulator
import qiskit.quantum_info

# import julia
# julia.install()
# from julia.api import Julia
# jl = Julia(compiled_modules=False)
# from julia import Main
# from julia import ITensors

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


def spin_spin_correlation_mixed(rdm1, rdm2, P_A, P_B):
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
    # rdm2 = -(rdm2 / 6 + rdm2.transpose(0, 3, 2, 1) / 3)
    # print();print()
    # print(rdm2.shape)
    # print(rdm2)properties_comp
    # Apply projection operators to the one-body RDMs and sum over indices i and j
    one_body_term = np.einsum('ik,jk,ij->', P_A, P_B, rdm1)

    # Apply projection operators to the two-body RDMs and sum over indices i, j, k, and l
    two_body_term = np.einsum('ij,kl,ijkl->', P_A, P_B, rdm2)

    # Combine terms according to the corrected equation
    SzAszB = 0.25 * one_body_term + 0.5 * two_body_term

    return SzAszB.real


backend = AerSimulator(method = 'statevector') 

def pauli_rdm12(Nshots,qubit_wise=True,backend=AerSimulator(method = 'statevector') ):
    my_pauli_list = []

    rdm1_pauli_list=[[0. for i in range(norb)] for j in range(norb)]#[[None]*norb]*norb#np.zeros([num_qubits,num_qubits])
    rdm2_pauli_list=[[[[0. for i in range(norb)] for j in range(norb)] for k in range(norb)] for l in range(norb)]#np.zeros([num_qubits,num_qubits,num_qubits,num_qubits])
    # print(np.shape(rdm2_pauli_list))
    for n in range(norb):
        for m in range(norb):
            
            op = FermionicOp(
                {
                    "+_{0} -_{1}".format(m,n): 2.0,

                    # "+_{0} -_{1}".format(m+norb,n+norb): 1.0,
                },
                num_spin_orbitals=num_qubits,
            )

            # print(op,mapper.map(op))
            # print(mapper.map(op).paulis)
            # print(my_pauli_list)
            rdm1_pauli_list[n][m] = copy.deepcopy(mapper.map(op))
            # print("n,m: ",n,m)
            # print("0,0: ",rdm1_pauli_list[0][0])
            # print("0,1: ",rdm1_pauli_list[0][1])
            if len(my_pauli_list) == 0:
                # print("True!")
                my_pauli_list = mapper.map(op).paulis
                # print(mapper.map(op).coeffs)
            else:
                my_pauli_list = my_pauli_list + mapper.map(op).paulis
                # print(mapper.map(op).coeffs)


    print("start 2rdm ops")
    for n in range(norb):
        for m in range(norb):
            for a in range(norb):
                for b in range(norb):
            
                    op = FermionicOp(
                        {
                            "+_{0} +_{2} -_{3} -_{1}".format(m+norb,n+norb,a,b): 1.0,

                            # "+_{0} +_{2} -_{3} -_{1}".format(m,n,a+norb,b+norb): 1.0,
                            
                            # "+_{0} +_{2} -_{3} -_{1}".format(m+norb,n+norb,a+norb,b+norb): 1.0,

                            # "+_{0} +_{2} -_{3} -_{1}".format(m,n,a,b): 1.0,
                        },
                        num_spin_orbitals=num_qubits,
                    )

                    # print(m,n,a,b)
                    rdm2_pauli_list[m][n][a][b] = copy.deepcopy(mapper.map(op))
                    # print(mapper.map(op).paulis)
                    # print(my_pauli_list)
                    if len(my_pauli_list) == 0:
                        # print("True!")
                        my_pauli_list = mapper.map(op).paulis
                        # print(mapper.map(op).coeffs)
                    else:
                        my_pauli_list = my_pauli_list + mapper.map(op).paulis
                        # print(mapper.map(op).coeffs)            


    print("group commutinmg  ",len(my_pauli_list),len(list(set(my_pauli_list))))
    qiskit_pauli_list = qiskit.quantum_info.PauliList(list(set(my_pauli_list))).group_commuting(qubit_wise=qubit_wise)
    print("group commutinmg done", len(qiskit_pauli_list))
    diag_circs=[]


    

    result_strings=[]

    #### make the circuits which diagonalise a commuting group & the diagonal ops (H if it is an X op, S^dag H if it is a Y otherwise already diagonal)
    # for commuting_group in qiskit_pauli_list:

    def get_diag_qbw_circ_op(commuting_group):
        group_diag_circ = ["I"]*num_qubits
        group_results=[]
        for pauli_string in commuting_group:

            result_circ=["I"]*num_qubits
            for p in range(len(pauli_string)):
                # print(type(pauli_string[p].to_label()))
                if pauli_string[p].to_label() != "I":
                    result_circ[p]="Z"
                if pauli_string[p].to_label()=="X":
                    group_diag_circ[p]="H"
                elif pauli_string[p].to_label()=="Y":
                    group_diag_circ[p]="SH"
            result_circ.reverse()
            group_results.append(result_circ)
        return group_diag_circ,group_results    
        # result_strings.append(group_results)    
        # diag_circs.append(group_diag_circ)
    # print(diag_circs)            


    print("start exp vals")
    all_exp_vals=[]
    shots_per_group = Nshots//len(qiskit_pauli_list)

    for n in range(len(qiskit_pauli_list[:])):
        qc=qiskit.QuantumCircuit(num_qubits)
        # print(n.to_labels())
        # print(diag_results(tableaus(n.to_labels()),True)[0][:,2*num_qubits:2*num_qubits+1])
        # result_strings = diag_results(tableaus(n.to_labels()),True)[1]
        if qubit_wise:
            diag_gates, result_string = get_diag_qbw_circ_op(qiskit_pauli_list[n])#diag_results(tableaus(n.to_labels()),True)[2]
        if not qubit_wise:
            sys.exit("Non QWC not implemented")    
        # print(diag_results(tableaus(n.to_labels()))[2])
        # print(diag_gates)
        for gate_n in range(len(diag_gates)):
            if diag_gates[gate_n] == "H":
                qc.h(gate_n)
            elif diag_gates[gate_n] == "SH":
                qc.sdg(gate_n)
                qc.h(gate_n)   
            elif diag_gates[gate_n] == "I":                           
                pass
            else:
                print("************************************* PROBLEM ****************************************")
                raise Exception("Unknown gate type"+str(diag_gates[gate_n]))
            
        # op = qiskit.quantum_info.Operator(qc)  
        # # print(qc)
        # op_mat = op.data  #.reverse_qargs()
        # # print(op.data)
        # for count in range(len(qiskit_pauli_list[n])):
        #     # print(qiskit_pauli_list[n][count],result_string[count])
        #     # print(np.allclose((op_mat.conjugate().transpose() @ qiskit.quantum_info.Pauli("".join(result_strings[n][count][::])).to_matrix() @ op_mat), qiskit_pauli_list[n][count].to_matrix()))
        #     assert(np.allclose((op_mat.conjugate().transpose() @ qiskit.quantum_info.Pauli("".join(result_string[count][::])).to_matrix() @ op_mat), qiskit_pauli_list[n][count].to_matrix()))
        
        measure_circ=qiskit.QuantumCircuit(num_qubits)
        measure_circ.initialize(SV_exact)
        measure_circ.compose(qc, inplace=True)
        measure_circ.measure_all()
        basis_result = backend.run(measure_circ,shots=shots_per_group).result()
        counts = basis_result.get_counts()
        # print(counts)
        exp_vals=[0. for throwaway_count in range(len(result_string))]
        for diag_string_n in range(len(result_string)):
            diag_string = result_string[diag_string_n]
            for key in counts.keys():
                # print(key, counts[key])
                sgn = 1
                for measure_binary, IZ in zip(key,diag_string):
                    # print(measure_binary, IZ) 
                    # ###reverse one of them?
                    if IZ=="Z" and measure_binary=="1":
                        sgn *= -1
                        # print("triggered")
                exp_vals[diag_string_n] += float(sgn * counts[key])/shots_per_group


        # print(exp_vals)
        # print(result_string)    
        all_exp_vals.append(exp_vals)    



        # print("\n\n***********************************************************************************************************************************")
        # print("***********************************************************************************************************************************\n\n")


    qiskit_pauli_list_list=[]
    for m in qiskit_pauli_list:
        qiskit_pauli_list_list.append(m.to_labels())


    def index_2d(myList, v):
        for i, x in enumerate(myList):
            if v in x:
                return (i, x.index(v))
    # rdm1_qwc=[]
    rdm1_qwc = [[0. for i in range(norb)] for j in range(norb)]#[[None]*norb]*norb#np.zeros([num_qubits,num_qubits])


    for n in range(norb):
        for m in range(norb):
            if rdm1_pauli_list[n][m]:
                for p in range(len(rdm1_pauli_list[n][m].paulis)):
                    index = index_2d(qiskit_pauli_list_list,rdm1_pauli_list[n][m].paulis[p].to_label())
                    # print(all_exp_vals[index[0]][index[1]])
                    rdm1_qwc[n][m]+=all_exp_vals[index[0]][index[1]] * rdm1_pauli_list[n][m].coeffs[p]
                    # if n == 3 and m==0:
                    #     print(rdm1_pauli_list[n][m].paulis[p])
                    #     print(all_exp_vals[index[0]][index[1]] * rdm1_pauli_list[n][m].coeffs[p],all_exp_vals[index[0]][index[1]] , rdm1_pauli_list[n][m].coeffs[p])
    # rdm1_qwc

    rdm2_qwc = [[[[0. for i in range(norb)] for j in range(norb)] for k in range(norb)] for l in range(norb)]#[[0.]*norb]*norb#np.zeros([norb,norb])
    for n in range(norb):
        for m in range(norb):
            for a in range(norb):
                for b in range(norb):
                    if rdm2_pauli_list[n][m][a][b]:
                        for p in range(len(rdm2_pauli_list[n][m][a][b].paulis)):
                            index = index_2d(qiskit_pauli_list_list,rdm2_pauli_list[n][m][a][b].paulis[p].to_label())
                            # print(all_exp_vals[index[0]][index[1]])
                            rdm2_qwc[n][m][a][b]+=all_exp_vals[index[0]][index[1]] * rdm2_pauli_list[n][m][a][b].coeffs[p]    

    return rdm1_qwc,rdm2_qwc            

# rdm1_qwc,rdm2_qwc = pauli_rdm12(Nshots=1e4)    











max_order=3


def _tn_addrs_signs(norb: int, nocc: int, order: int) -> tuple[NDArray[integer], NDArray[integer]]:
    """Get the addresses and signs for the given order.

    Args:
        norb: Number of orbitals.
        nocc: Number of occupied orbitals.
        order: Order of the excitation.

    Returns:
        Addresses and signs for the given order.
    """
    addrs, signs = pyscf.ci.cisd.tn_addrs_signs(norb, nocc, order)
    return np.asarray(addrs, dtype=np.int64), np.asarray(signs, dtype=np.int64)











# def adapt2pyscf_c(spins: str,cn,c0) -> NDArray[floating]:
#     """Get the C amplitudes for a given spin configuration."""
#     # Find the spins
#     nalph = spins.count("a")
#     nbeta = spins.count("b")

#     # Get the addresses and signs
#     addrsi: Union[int, NDArray[integer]] = 0
#     addrsj: Union[int, NDArray[integer]] = 0
#     signsi: Union[int, NDArray[integer]] = 1
#     signsj: Union[int, NDArray[integer]] = 1
#     if nalph != 0:
#         addrsi = addrs[nalph]
#         signsi = signs[nalph]
#     if nbeta != 0:
#         addrsj = addrs[nbeta]
#         signsj = signs[nbeta]
#     if nalph != 0 and nbeta != 0:
#         addrsi, addrsj = np.ix_(addrsi, addrsj)  # type: ignore
#         signsi = signsi[:, None]  # type: ignore
#         signsj = signsj[None, :]  # type: ignore

#     # Get the amplitudes
#     # print(addrsi,addrsj)
#     # print(signsi,signsj)
#     cn = cn * signsi * signsj
#     print("cn: \n",cn.shape)
#     # Decompress the axes
#     shape = tuple(
#         space.size(char) for char in ("O" * nalph + "V" * nalph + "O" * nbeta + "V" * nbeta)
#     )
#     subscript = "i" * nalph + "a" * nalph + "j" * nbeta + "b" * nbeta
#     cn = ebcc.util.decompress_axes(subscript, cn, shape=shape)
#     # print(cn)
#     # Transpose the axes
#     subscript_target = ""
#     for spin in spins:
#         subscript_target += "i" if spin == "a" else "j"
#     for spin in spins:
#         subscript_target += spin  # a->a and b->b
#     perm = ebcc.util.get_string_permutation(subscript, subscript_target)
#     cn = np.transpose(cn, perm)

#     # Scale by reference energy
#     cn /= c0

#     return cn



# Main.include("../Quantum_cluster/src/julia_functions.jl")

nsite = int(float(sys.argv[2]))
d = 1.5

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


occupied = myhf.mo_occ > 0
active = np.zeros_like(occupied)
nocc = np.sum(occupied)
#active[nocc - 4 : nocc + 4] = True  # First four HOMOs and LUMOs
active[:] = True
frozen = ~active
space = Space(
    occupied,
    frozen,
    active,
)





norb = myhf.mo_coeff.shape[1]
nelec = mol.nelec
nocc=nsite//2
nvir=norb-nocc
# nocc = sum(nelec)
# nvir = norb - nocc

# print("ns: ",norb, nelec,nocc, nvir)
print("ns: ",norb, nelec)

h1e = np.linalg.multi_dot((myhf.mo_coeff.T, myhf.get_hcore(), myhf.mo_coeff))
eri = ao2mo.incore.general(myhf._eri, (myhf.mo_coeff,)*4, compact=False)
# Remove all of the permutational symmetry that is used to compress the integral array
eri = ao2mo.restore(1, eri, norb)
cisolver = fci.direct_spin0.FCI()
e1, civec = cisolver.kernel(h1e, eri, norb, nelec)
e_fci = e1 + mol.energy_nuc()
print(e_fci,e1)

fci_coeffs = _ci_vector_to_coefficients_restricted(civec,space,max_order=max_order)

one_body = myhf.mo_coeff.T @ myhf.get_hcore() @ myhf.mo_coeff
eri = ao2mo.kernel(myhf._eri, (myhf.mo_coeff, myhf.mo_coeff,myhf.mo_coeff,myhf.mo_coeff), compact=False)
two_body = eri.reshape((myhf.mo_coeff.shape[-1],) * 4)

# Constructing the electronic hamiltonian in second quantised representation
integrals = ElectronicIntegrals.from_raw_integrals(h1_a=one_body, h1_b = one_body, h2_aa=two_body, h2_bb= two_body, h2_ba= two_body , auto_index_order=True) 

# Defining the many body electronic hamiltionian in second quantised representation

h_elec = ElectronicEnergy(integrals, constants = {'nuclear_repulsion_energy':mol.energy_nuc()}).second_q_op()       
mapper = JordanWignerMapper()
qubit_ham = mapper.map(h_elec)


addrs: dict[int, NDArray[integer]] = {}
signs: dict[int, NDArray[integer]] = {}
for order in range(1, max_order + 1):
    addrs[order], signs[order] = _tn_addrs_signs(norb, nocc, order)

def _get_c(spins: str,civec) -> NDArray[floating]:
    """Get the C amplitudes for a given spin configuration."""
    # Find the spins
    nalph = spins.count("a")
    nbeta = spins.count("b")

    # Get the addresses and signs
    addrsi: Union[int, NDArray[integer]] = 0
    addrsj: Union[int, NDArray[integer]] = 0
    signsi: Union[int, NDArray[integer]] = 1
    signsj: Union[int, NDArray[integer]] = 1
    if nalph != 0:
        addrsi = addrs[nalph]
        signsi = signs[nalph]
    if nbeta != 0:
        addrsj = addrs[nbeta]
        signsj = signs[nbeta]
    if nalph != 0 and nbeta != 0:
        addrsi, addrsj = np.ix_(addrsi, addrsj)  # type: ignore
        signsi = signsi[:, None]  # type: ignore
        signsj = signsj[None, :]  # type: ignore

    # Get the amplitudes
    # print(addrsi,addrsj)
    # print(signsi,signsj)
    cn = civec[addrsi, addrsj] * signsi * signsj
    # print("cn: \n",cn.shape)
    cn0=civec[addrsi, addrsj]

    cn0_s = civec[addrsi, addrsj] * signsi * signsj 
    # Decompress the axes
    shape = tuple(
        space.size(char) for char in ("O" * nalph + "V" * nalph + "O" * nbeta + "V" * nbeta)
    )
    subscript = "i" * nalph + "a" * nalph + "j" * nbeta + "b" * nbeta
    cn = ebcc.util.decompress_axes(subscript, cn, shape=shape)
    # print(cn)
    # Transpose the axes
    subscript_target = ""
    for spin in spins:
        subscript_target += "i" if spin == "a" else "j"
    for spin in spins:
        subscript_target += spin  # a->a and b->b
    perm = ebcc.util.get_string_permutation(subscript, subscript_target)
    cn = np.transpose(cn, perm)

    # Scale by reference energy
    cn /= civec[0, 0]
    cn0_s /= civec[0, 0]

    return cn, cn0, cn0_s


n_qubits = 2.*norb

# my_ham=[]
# for h_i in h_elec:
#     # print(h_i)
#     # print([x.split("_") for x in h_i.split()],":  ",Lan_coeff.h_elec[h_i])
#     qb_op=(h_elec[h_i],)
#     for op_str in [x.split("_") for x in h_i.split()][::-1]:
#         if op_str[0]=="+":
#             qb_op=qb_op+("S-",int(op_str[1])+1)
#             # print(op_str[1])
#             for z_qb in range(int(op_str[1])):
#                 qb_op = qb_op + ("Z",z_qb+1)
#         elif op_str[0]=="-":
#             # print(op_str[1])
#             qb_op=qb_op+("S+",int(op_str[1])+1)
#             for z_qb in range(int(op_str[1])):
#                 qb_op = qb_op + ("Z",z_qb+1)
#     # print("qb_op: ",qb_op)  
#     # print("")      
#     my_ham.append(qb_op)
# # print("my ham:  ",my_ham)    
# dmrg_energy,psi = Main.my_dmrg(n_qubits,my_ham,200,maxdim=1024)

# itensor_sv = Main.get_SV_from_mps(psi)

# itensor_sv_bit_flipped=[0.]*len(itensor_sv)
# for n in range(len(itensor_sv)):
#     bins="0"*int((n_qubits-len(str(bin(n))[2:])))+str(bin(n))[2:]
#     bit_flipped_bin="0"*len(bins)
#     for i in range(len(bins)):
#         if bins[i] == "0":
#             bit_flipped_bin = bit_flipped_bin[:i]+"1"+bit_flipped_bin[i+1:]


#     # print(n,bins,bit_flipped_bin)
#     itensor_sv_bit_flipped[int(bit_flipped_bin,2)]=itensor_sv[n]
# SV_exact = itensor_sv_bit_flipped
# print(SV_exact)

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

from qiskit.quantum_info import Statevector

num_qubits = 2*norb
Shadow = classical_shadow.Classical_shadow("Clifford",num_qubits,[],[])
# Shadow_tau.calc_shadow_lists(Statevector(1/np.sqrt(2)*(np.array(qiskit.quantum_info.Statevector.from_label("0"*num_qubits)) + np.array(SV_exact))),no_estimators_K = 3,N_per_estimator=int((Nshots)//3))
# Shadow.calc_shadow_lists((Statevector(1/np.sqrt(2)*(np.array(qiskit.quantum_info.Statevector.from_label("0"*num_qubits)) + np.array(Statevector(GS_circ)))),N_estimator,int(Nshots/N_estimator)))
Nshots=int(float(sys.argv[1]))

Shadow.calc_shadow_lists(Statevector(1/np.sqrt(2)*(np.array(qiskit.quantum_info.Statevector.from_label("0"*num_qubits)) + np.array(SV_exact))),3,int(Nshots/3))



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
fermi_vac = "0"*(norb//2) + "1"*(norb//2) + "0"*(norb//2) + "1"*(norb//2)
index_a=0#"0"*SV_basis_0.num_qubits#int(list(SV_basis_0.to_dict().keys())[0],2)

###index for HF state
index_b=int(fermi_vac,2)
op_list=[]
op_list.append([1.,index_a,index_b])
results_l,errors_l = Shadow.evaluate_operator_overlap_stabilizer_perp(op_list)
# print(results_l)
c0_cs=results_l[0]

a_exc = generate_fermionic_excitations(num_excitations=1, num_spatial_orbitals=norb, num_particles=nelec, alpha_spin=True, beta_spin=True, max_spin_excitation=None, generalized=False, preserve_spin=True)
a_exc_strings=excitations_tuples_to_strings(a_exc,fermi_vac)
op_list=[]
for a in a_exc_strings:
    index_a=0#"0"*SV_basis_0.num_qubits#int(list(SV_basis_0.to_dict().keys())[0],2)
    index_b=int(a,2)
    op_list.append([1.,index_a,index_b])

results_l,errors_l = Shadow.evaluate_operator_overlap_stabilizer_perp(op_list)
a_int_norm_results = [x for x in results_l]
print(a_int_norm_results)
# t1addr,t1sign = pyscf.ci.cisd.tn_addrs_signs(norb, nocc, 1)
# c1_cs_rs = (a_int_norm_results * t1sign).reshape(nocc, nvir)
# print(c1_cs_rs)

ab_exc = generate_fermionic_excitations(num_excitations=2, num_spatial_orbitals=norb, num_particles=nelec, alpha_spin=True, beta_spin=True, max_spin_excitation=None, generalized=False, preserve_spin=True)
# ab_exc = generate_fermionic_excitations(num_excitations=2, max_spin_excitation=1, num_spatial_orbitals=norb, num_particles=nelec, alpha_spin=True, beta_spin=True, generalized=False, preserve_spin=True)
ab_exc_strings=excitations_tuples_to_strings(ab_exc,fermi_vac)
op_list=[]
for ab in ab_exc_strings:
    index_a=0#"0"*SV_basis_0.num_qubits#int(list(SV_basis_0.to_dict().keys())[0],2)
    index_b=int(ab,2)
    op_list.append([1.,index_a,index_b])

results_l,errors_l = Shadow.evaluate_operator_overlap_stabilizer_perp(op_list)
ab_int_norm_results = [x for x in results_l]
print(ab_int_norm_results)
# c2_cs_ab = np.einsum("i,j,ij->ij", t1sign, t1sign, np.reshape(ab_int_norm_results,(len(t1addr),len(t1addr))))
# c2_cs_ab = np.reshape(c2_cs_ab,(nocc, nvir, nocc, nvir)).transpose(0, 2, 1, 3)
# print(c2_cs_ab/SV_exact[int("00110011",2)])


# aba_exc = generate_fermionic_excitations(num_excitations=3, num_spatial_orbitals=norb, num_particles=nelec, alpha_spin=True, beta_spin=True, max_spin_excitation=None, generalized=False, preserve_spin=True)
# ab_exc = generate_fermionic_excitations(num_excitations=2, max_spin_excitation=1, num_spatial_orbitals=norb, num_particles=nelec, alpha_spin=True, beta_spin=True, generalized=False, preserve_spin=True)
# aba_exc2 = []
# for _exc in aba_exc:
#     # print(_exc[0] > )
#     total=0
#     for _exc0 in _exc[0]:
#         if _exc0 >= norb:
#             total+=1
#     # print(total)
#     if total <= 1:
#         aba_exc2.append(_exc)
# aba_exc_strings=excitations_tuples_to_strings(aba_exc[::],fermi_vac)
# op_list=[]
# for aba in aba_exc_strings:
#     index_a=0#"0"*SV_basis_0.num_qubits#int(list(SV_basis_0.to_dict().keys())[0],2)
#     index_b=int(aba,2)
#     op_list.append([1.,index_a,index_b])
# results_l=[]
# for n in range(len(op_list)):
#     # print(op_list[n][2])
#     # results_l.append(SV_exact[op_list[n][2]])
#     results_l,errors_l = Shadow.evaluate_operator_overlap_stabilizer_perp(op_list)
# aba_int_norm_results = [x for x in results_l] 


# e4_exc = generate_fermionic_excitations(num_excitations=4, num_spatial_orbitals=norb, num_particles=nelec, alpha_spin=True, beta_spin=True, generalized=False, preserve_spin=True)
# # ab_exc = generate_fermionic_excitations(num_excitations=2, max_spin_excitation=1, num_spatial_orbitals=norb, num_particles=nelec, alpha_spin=True, beta_spin=True, generalized=False, preserve_spin=True)

# e4_exc_strings=excitations_tuples_to_strings(e4_exc[::1],fermi_vac)
# op_list=[]
# for e4 in e4_exc_strings:
#     index_a=0#"0"*SV_basis_0.num_qubits#int(list(SV_basis_0.to_dict().keys())[0],2)
#     index_b=int(e4,2)
#     op_list.append([1.,index_a,index_b])
# results_l=[]
# for n in range(len(op_list)):
#     # print(op_list[n][2])
#     # results_l.append(SV_exact[op_list[n][2]])
#     results_l,errors_l = Shadow.evaluate_operator_overlap_stabilizer_perp(op_list)
# e4_int_norm_results = [x for x in results_l] 



print(norb,nelec)
my_civec_cs=np.zeros((math.comb(norb,nelec[0]),math.comb(norb,nelec[1])))
my_civec_cs[0,0] = c0_cs#SV_exact[int("00110011",2)]

for a_count in range(len(a_exc_strings)):
    a = a_exc_strings[a_count]
    assert len(a)%2==0
    ci_index2 = cistring.str2addr(norb, nelec[0], bin(int(a[:len(a)//2],2)))
    ci_index1 = cistring.str2addr(norb, nelec[0], bin(int(a[len(a)//2:],2)))
    # print(ci_index1,ci_index2,a_int_norm_results[a_count])
    my_civec_cs[ci_index1,ci_index2] = a_int_norm_results[a_count]

for ab_count in range(len(ab_exc_strings)):
    ab = ab_exc_strings[ab_count]
    assert len(ab)%2==0
    ci_index2 = cistring.str2addr(norb, nelec[0], bin(int(ab[:len(ab)//2],2)))
    ci_index1 = cistring.str2addr(norb, nelec[0], bin(int(ab[len(ab)//2:],2)))
    # print(ci_index1,ci_index2)
    my_civec_cs[ci_index1,ci_index2] = ab_int_norm_results[ab_count]


# for aba_count in range(len(aba_exc_strings)):
#     aba = aba_exc_strings[aba_count]
#     assert len(aba)%2==0
#     ci_index2 = cistring.str2addr(norb, nelec[0], bin(int(aba[:len(aba)//2],2)))
#     ci_index1 = cistring.str2addr(norb, nelec[0], bin(int(aba[len(aba)//2:],2)))
#     # print(ci_index1,ci_index2)
#     my_civec_cs[ci_index1,ci_index2] = aba_int_norm_results[aba_count] 


# for e4_count in range(len(e4_exc_strings)):
#     e4 = e4_exc_strings[e4_count]
#     assert len(e4)%2==0
#     ci_index2 = cistring.str2addr(norb, nelec[0], bin(int(e4[:len(e4)//2],2)))
#     ci_index1 = cistring.str2addr(norb, nelec[0], bin(int(e4[len(e4)//2:],2)))
#     # print(ci_index1,ci_index2, e4_int_norm_results[e4_count])
#     my_civec_cs[ci_index1,ci_index2] = e4_int_norm_results[e4_count]        
    # print(my_civec_cs[ci_index1,ci_index2])
# print(np.allclose(_get_c("a",my_civec_cs)[0],c1_cs_rs/SV_exact[int("00110011",2)]))
# print(np.allclose(_get_c("ab",my_civec_cs)[0],c2_cs_ab/SV_exact[int("00110011",2)]))
# print(np.allclose(_get_c("aba",my_civec_cs)[0],_get_c("aba",civec)[0],atol=1e-6))
# print(np.allclose(_get_c("abab",my_civec_cs)[0],_get_c("abab",civec)[0],atol=1e-6))
# print(np.allclose(_get_c("abaa",my_civec_cs)[0],_get_c("abaa",civec)[0],atol=1e-6))  


occupied = myhf.mo_occ > 0
active = np.zeros_like(occupied)
nocc = np.sum(occupied)
active[:] = True  # First four HOMOs and LUMOs
# active[nocc - 1 : nocc + 1] = True  # First four HOMOs and LUMOs
frozen = ~active
space = Space(
    occupied,
    frozen,
    active,
)

# Run an FCI calculation in the active space
mo = mo=myhf.mo_coeff[:, space.active]

h1e = np.einsum("pq,pi,qj->ij", myhf.get_hcore(), mo, mo)

h2e = ao2mo.kernel(myhf._eri, mo, compact=False).reshape((mo.shape[-1],) * 4)
ci = fci.direct_spin1.FCI()
ci.kernel(h1e, h2e, space.nact, space.naocc * 2)

h1 = np.linalg.multi_dot((myhf.mo_coeff.T, myhf.get_hcore(), myhf.mo_coeff))
# Get 2-electron electron repulsion integrals, transformed into MO basis
eri = ao2mo.incore.general(myhf._eri, (myhf.mo_coeff,)*4, compact=False)
# Previous representation exploited permutational symmetry in storage. Change this to a 4D array.
# Integrals now stored as h2[p,q,r,s] = (pq|rs) = <pr|qs>. Note 8-fold permutational symmetry.
h2 = ao2mo.restore(1, eri, norb)

# Notes from ebcc example: 
# - There are public fci_to_amplitudes_{spin} functions, but I use the private functions for
#   finer control.
# - If you're using custom C amplitudes there may be conventional differences. This can be worked
#   out if needed.
# - The `coefficients` object can be initialised from NumPy arrays as
#   `coefficients = ebcc.util.Namespace(c1=c1, c2=c2, c3=c3)`, scaled by `c0` I guess?

# Extract the C amplitudes from the FCI calculation
coefficients = _ci_vector_to_coefficients_restricted(civec, space, max_order=3)

# Get the T amplitudes
amplitudes = _coefficients_to_amplitudes_restricted(coefficients, max_order=3)

# Initialise a ebcc_CCSD calculation and get the 1RDM
ci_ebcc_ccsd = REBCC(myhf, ansatz="CCSD", space=space)
ci_ebcc_ccsd.amplitudes = amplitudes
ci_ebcc_ccsd.solve_lambda()
ci_rdm1_sd = ci_ebcc_ccsd.make_rdm1_f()
ci_rdm2_sd = ci_ebcc_ccsd.make_rdm2_f()

ci_ebcc_ccsdt = REBCC(myhf, ansatz="CCSDT", space=space)
ci_ebcc_ccsdt.amplitudes = amplitudes
ci_ebcc_ccsdt.solve_lambda()
ci_rdm1_sdt = ci_ebcc_ccsdt.make_rdm1_f()
ci_rdm2_sdt = ci_ebcc_ccsdt.make_rdm2_f()

np.set_printoptions(precision=6, suppress=True, linewidth=200, edgeitems=10)

print();print()

print("E: ")
print(ci_ebcc_ccsd.energy() + myhf.e_tot)
# print(ci_ebcc_ccsdt.energy() + myhf.e_tot)
print(ci.e_tot + myhf.energy_nuc())

print();print()

print("1-rdm: ")
print(ci_rdm1_sd)
# print(ci_rdm1_sdt)
print(ci.make_rdm1(ci.ci, space.nact, space.naocc * 2))



coefficients = _ci_vector_to_coefficients_restricted(my_civec_cs, space, max_order=2)

# Get the T amplitudes
amplitudes = _coefficients_to_amplitudes_restricted(coefficients, max_order=2)

# Initialise a ebcc_CCSD calculation and get the 1RDM
ebcc_ccsd = REBCC(myhf, ansatz="CCSD", space=space)
ebcc_ccsd.amplitudes = amplitudes
ebcc_ccsd.solve_lambda()
rdm1_sd = ebcc_ccsd.make_rdm1_f()
rdm2_sd = ebcc_ccsd.make_rdm2_f()

# ebcc_ccsdt = REBCC(myhf, ansatz="CCSDT", space=space)
# ebcc_ccsdt.amplitudes = amplitudes
# ebcc_ccsdt.solve_lambda()
# rdm1_sdt = ebcc_ccsdt.make_rdm1_f()
# rdm2_sdt = ebcc_ccsdt.make_rdm2_f()

np.set_printoptions(precision=6, suppress=True, linewidth=200, edgeitems=10)

print();print()

print("E: ")
print(ebcc_ccsd.energy() + myhf.e_tot)
#print(ebcc_ccsdt.energy() + myhf.e_tot)
print(ci.e_tot + myhf.energy_nuc())

print();print()

print("1-rdm: ")
print(rdm1_sd)
#print(rdm1_sdt)
print(ci.make_rdm1(ci.ci, space.nact, space.naocc * 2))




emb = ewf.EWF(myhf, bath_options=dict(bathtype="full"), solver="FCI")
emb.kernel()
projectors = emb._get_atom_projectors()[2]

print("projectors done")

# print(len(rdm2_qwc))

rdm1_qwc,rdm2_qwc = pauli_rdm12(Nshots=Nshots) 
np_rdm1_qwc = np.array(rdm1_qwc)
np_rdm2_qwc = np.array(rdm2_qwc)
np_qwc_rdm2_sep = (-np_rdm2_qwc).transpose(0,3,2,1).real
np_qwc_rdm2_sum = 2.*(np_qwc_rdm2_sep - 2.*np_qwc_rdm2_sep.transpose(0,3,2,1))
print("qwc made")

rdm2_fci = cisolver.make_rdm2(cisolver.ci, space.nact, space.naocc * 2)
rdm2_fci = -(rdm2_fci / 6 + rdm2_fci.transpose(0, 3, 2, 1) / 3)
rdm2_qwc_s = -(np_rdm2_qwc / 6 + np_rdm2_qwc.transpose(0, 3, 2, 1) / 3)



two_e_energy = np.einsum('ijkl,ijkl->', two_body, cisolver.make_rdm2(cisolver.ci, norb, nelec)) * 0.5
one_e_energy = np.einsum('ij,ij->', one_body, cisolver.make_rdm1(cisolver.ci, norb, nelec))
fci_rdm_energy = two_e_energy + one_e_energy + mol.energy_nuc()
print('Energy from RDMs: ', fci_rdm_energy)
assert(np.isclose(fci_rdm_energy, e1 + mol.energy_nuc()))

print("fci rdm1: \n",cisolver.make_rdm1(cisolver.ci, norb, nelec))
print()
print("fci rdm2: \n",cisolver.make_rdm2(cisolver.ci, norb, nelec)[1][0])
print()
print("fci rdm2: \n",cisolver.make_rdm2(cisolver.ci, norb, nelec)[0][0])

two_e_energy = np.einsum('ijkl,ijkl->', two_body, np_qwc_rdm2_sum) * 0.5
one_e_energy = np.einsum('ij,ij->', one_body, np_rdm1_qwc)
qwc_rdm_energy = two_e_energy + one_e_energy + mol.energy_nuc()
print('Energy from RDMs: ', qwc_rdm_energy)

print("qwc rdm1: \n",np_rdm1_qwc)
print()
print("qwc rdm2: \n",np_qwc_rdm2_sum[1][0])
print()
print("qwc rdm2: \n",np_qwc_rdm2_sum[0][0])



two_e_energy = np.einsum('ijkl,ijkl->', two_body, ci_rdm2_sd) * 0.5
one_e_energy = np.einsum('ij,ij->', one_body, ci_rdm1_sd)
ci_sd_rdm_energy = two_e_energy + one_e_energy + mol.energy_nuc()
print('Energy from RDMs: ', ci_sd_rdm_energy)


two_e_energy = np.einsum('ijkl,ijkl->', two_body, ci_rdm2_sdt) * 0.5
one_e_energy = np.einsum('ij,ij->', one_body, ci_rdm1_sdt)
ci_sdt_rdm_energy = two_e_energy + one_e_energy + mol.energy_nuc()
print('Energy from RDMs: ', ci_sdt_rdm_energy)


two_e_energy = np.einsum('ijkl,ijkl->', two_body, rdm2_sd) * 0.5
one_e_energy = np.einsum('ij,ij->', one_body, rdm1_sd)
cs_sd_rdm_energy = two_e_energy + one_e_energy + mol.energy_nuc()
print('Energy from RDMs: ', cs_sd_rdm_energy)

print("cs rdm1: \n",rdm1_sd)
print()
print("cs rdm2: \n",rdm2_sd[1][0])
print()
print("cs rdm2: \n",rdm2_sd[0][0])

# print("      FCI               CS->CCSD               CS->CCSDT","     FCI->CCSD               FCI->CCSDT")



# print(spin_spin_correlation(cisolver.make_rdm1(cisolver.ci, space.nact, space.naocc * 2),cisolver.make_rdm2(cisolver.ci, space.nact, space.naocc * 2),projectors[0],projectors[3]),spin_spin_correlation(rdm1_sd,rdm2_sd,projectors[0],projectors[3]),spin_spin_correlation(rdm1_sdt,rdm2_sdt,projectors[0],projectors[3]), spin_spin_correlation(ci_rdm1_sd,ci_rdm2_sd,projectors[0],projectors[3]),spin_spin_correlation(ci_rdm1_sdt,ci_rdm2_sdt,projectors[0],projectors[3]))
# print(spin_spin_correlation(cisolver.make_rdm1(cisolver.ci, space.nact, space.naocc * 2),cisolver.make_rdm2(cisolver.ci, space.nact, space.naocc * 2),projectors[0],projectors[2]),spin_spin_correlation(rdm1_sd,rdm2_sd,projectors[0],projectors[2]),spin_spin_correlation(rdm1_sdt,rdm2_sdt,projectors[0],projectors[2]), spin_spin_correlation(ci_rdm1_sd,ci_rdm2_sd,projectors[0],projectors[2]),spin_spin_correlation(ci_rdm1_sdt,ci_rdm2_sdt,projectors[0],projectors[2]))
# print(spin_spin_correlation(cisolver.make_rdm1(cisolver.ci, space.nact, space.naocc * 2),cisolver.make_rdm2(cisolver.ci, space.nact, space.naocc * 2),projectors[0],projectors[1]),spin_spin_correlation(rdm1_sd,rdm2_sd,projectors[0],projectors[1]),spin_spin_correlation(rdm1_sdt,rdm2_sdt,projectors[0],projectors[1]), spin_spin_correlation(ci_rdm1_sd,ci_rdm2_sd,projectors[0],projectors[1]),spin_spin_correlation(ci_rdm1_sdt,ci_rdm2_sdt,projectors[0],projectors[1]))
# print(spin_spin_correlation(cisolver.make_rdm1(cisolver.ci, space.nact, space.naocc * 2),cisolver.make_rdm2(cisolver.ci, space.nact, space.naocc * 2),projectors[0],projectors[0]),spin_spin_correlation(rdm1_sd,rdm2_sd,projectors[0],projectors[0]),spin_spin_correlation(rdm1_sdt,rdm2_sdt,projectors[0],projectors[0]), spin_spin_correlation(ci_rdm1_sd,ci_rdm2_sd,projectors[0],projectors[0]),spin_spin_correlation(ci_rdm1_sdt,ci_rdm2_sdt,projectors[0],projectors[0]))

# print(spin_spin_correlation(myhf.make_rdm1(),myhf.make_rdm2(),projectors[0],projectors[3]),spin_spin_correlation(myhf.make_rdm1(),myhf.make_rdm2(),projectors[0],projectors[2]),spin_spin_correlation(myhf.make_rdm1(),myhf.make_rdm2(),projectors[0],projectors[1]),spin_spin_correlation(myhf.make_rdm1(),myhf.make_rdm2(),projectors[0],projectors[0]))
# print(spin_spin_correlation(cc.make_rdm1(),cc.make_rdm2(),projectors[0],projectors[3]),spin_spin_correlation(cc.make_rdm1(),cc.make_rdm2(),projectors[0],projectors[2]),spin_spin_correlation(cc.make_rdm1(),cc.make_rdm2(),projectors[0],projectors[1]),spin_spin_correlation(cc.make_rdm1(),cc.make_rdm2(),projectors[0],projectors[0]))

# print(spin_spin_correlation_mixed(rdm1_qwc,-np_rdm2_qwc.transpose(0, 3, 2, 1),projectors[0],projectors[3]),spin_spin_correlation_mixed(rdm1_qwc,-np_rdm2_qwc.transpose(0, 3, 2, 1),projectors[0],projectors[2]),spin_spin_correlation_mixed(rdm1_qwc,-np_rdm2_qwc.transpose(0, 3, 2, 1),projectors[0],projectors[1]),spin_spin_correlation_mixed(rdm1_qwc,-np_rdm2_qwc.transpose(0, 3, 2, 1),projectors[0],projectors[0]))

# for n in range(nsite):
#     # f_name="NiO_lat_con_"+str(atomic_dis)+"_nshots_"+str(Nshots)+"_k_"+str(kmesh[0])+str(kmesh[1])+str(kmesh[2])+"_basis_"+str(cell.basis)+"_"+bath_type+"_bath"+"magentics_"+magnetism+".dat"
#     f_name = "SzSz_0"+str(n)+"_H_"+str(nsite)+"_d_"+str(d)+"_nshots_"+str(Nshots)+"and_Etot_all_ci.dat"
#     if os.path.isfile(f_name):
#         f = open(f_name, "a")
#         # f.write(str(mf.e_tot)+"    "+str(cc.e_tot)+"    "+str(fci.e_tot)+"    "+str(emb_atom_fci.e_tot)+"    "+str(emb_fci.e_tot)+"    "+str(emb.e_tot)+"\n")
#         # f.write(str(spin_spin_correlation(myhf.make_rdm1(),myhf.make_rdm2(),projectors[0],projectors[n]))+"    "+str(spin_spin_correlation(cc.make_rdm1(),cc.make_rdm2(),projectors[0],projectors[n]))+"    "+str(spin_spin_correlation(cisolver.make_rdm1(cisolver.ci, space.nact, space.naocc * 2),cisolver.make_rdm2(cisolver.ci, space.nact, space.naocc * 2),projectors[0],projectors[n]))+"    "+str(spin_spin_correlation(rdm1_sd,rdm2_sd,projectors[0],projectors[n]))+"    "+str(spin_spin_correlation(rdm1_sdt,rdm2_sdt,projectors[0],projectors[n]))+"     "+str(spin_spin_correlation(ci_rdm1_sd,ci_rdm2_sd,projectors[0],projectors[n]))+"    "+str(spin_spin_correlation(ci_rdm1_sdt,ci_rdm2_sdt,projectors[0],projectors[n]))+"    "+str(spin_spin_correlation_mixed(rdm1_qwc,-np_rdm2_qwc.transpose(0, 3, 2, 1),projectors[0],projectors[n]))+"\n")
#         f.write(str(spin_spin_correlation(myhf.make_rdm1(),myhf.make_rdm2(),projectors[0],projectors[n]))+"    "+str(spin_spin_correlation(cc.make_rdm1(),cc.make_rdm2(),projectors[0],projectors[n]))+"    "+str(spin_spin_correlation(cisolver.make_rdm1(cisolver.ci, space.nact, space.naocc * 2),cisolver.make_rdm2(cisolver.ci, space.nact, space.naocc * 2),projectors[0],projectors[n]))+"    "+str(spin_spin_correlation(rdm1_sd,rdm2_sd,projectors[0],projectors[n]))+"    "+str(spin_spin_correlation(rdm1_sdt,rdm2_sdt,projectors[0],projectors[n]))+"     "+str(spin_spin_correlation(ci_rdm1_sd,ci_rdm2_sd,projectors[0],projectors[n]))+"    "+str(spin_spin_correlation(ci_rdm1_sdt,ci_rdm2_sdt,projectors[0],projectors[n]))+"    "+str(spin_spin_correlation_mixed(rdm1_qwc,-np_rdm2_qwc.transpose(0, 3, 2, 1),projectors[0],projectors[n]))+"    "+str(e_fci)+"    "+str(ebcc_ccsd.energy() + myhf.e_tot)+"    "+str(ci_ebcc_ccsd.energy() + myhf.e_tot)+"\n")
#         f.close()
#     else:
#         f = open(f_name, "a")
#         # f.write("c0"+"    "+"e_fci"+"    "+"e_hf"+"    "+"vqe_energy"+"    "+"HF+CS"+"    "+"HF+meas_diag"+"    "+"E_pauli_all"+"\n")
#         # f.write(str(c0)+"    "+str(e_fci)+"    "+str(myhf.e_tot)+"    "+str(vqe_energy)+"    "+str(e_proj_CS)+"    "+str(e_proj_meas_diag)+"    "+str(result_Pauli_all_comm)+"\n")
#         f.write("HF"+"    "+"CCSD"+"    "+"FCI"+"    "+"CS_CCSD"+"    "+"CS_CCSDT"+"    "+"FCI_CCSD"+"    "+"FCI_CCSDT"+"    "+"Pauli_group_qwc"+"     "+"e_fci"+"    "+"e_cs_ccsd"+"    e_fci_ccsd"+"\n")
#         f.write(str(spin_spin_correlation(myhf.make_rdm1(),myhf.make_rdm2(),projectors[0],projectors[n]))+"    "+str(spin_spin_correlation(cc.make_rdm1(),cc.make_rdm2(),projectors[0],projectors[n]))+"    "+str(spin_spin_correlation(cisolver.make_rdm1(cisolver.ci, space.nact, space.naocc * 2),cisolver.make_rdm2(cisolver.ci, space.nact, space.naocc * 2),projectors[0],projectors[n]))+"    "+str(spin_spin_correlation(rdm1_sd,rdm2_sd,projectors[0],projectors[n]))+"    "+str(spin_spin_correlation(rdm1_sdt,rdm2_sdt,projectors[0],projectors[n]))+"     "+str(spin_spin_correlation(ci_rdm1_sd,ci_rdm2_sd,projectors[0],projectors[n]))+"    "+str(spin_spin_correlation(ci_rdm1_sdt,ci_rdm2_sdt,projectors[0],projectors[n]))+"    "+str(spin_spin_correlation_mixed(rdm1_qwc,-np_rdm2_qwc.transpose(0, 3, 2, 1),projectors[0],projectors[n]))+"    "+str(e_fci)+"    "+str(ebcc_ccsd.energy() + myhf.e_tot)+"    "+str(ci_ebcc_ccsd.energy() + myhf.e_tot)+"\n")
#         f.close()


from pyscf.ci import cisd

c0 = my_civec_cs[0,0]
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
c1 = my_civec_cs[0, t1addr] * t1sign
c1 = c1.reshape((nocc, nvir))
# Intermediately normalize C1 amplitudes by the HF reference amplitude
c1 = c1 / c0

# === C2 amplitudes ===
# For RHF, we want the (alpha, beta) -> (alpha, beta) 'mixed spin' excitation amplitudes.
# Therefore, we can just take single excitations of alpha and
# combine with the single excitations of beta.
# For a spin-symmetry-broken reference state (UHF), we would also need to get the aa->aa
# and bb->bb excitations, but we can leave these for the moment.
c2 = np.einsum("i,j,ij->ij", t1sign, t1sign, my_civec_cs[t1addr[:, None], t1addr])
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



for n in range(nsite):
    # f_name="NiO_lat_con_"+str(atomic_dis)+"_nshots_"+str(Nshots)+"_k_"+str(kmesh[0])+str(kmesh[1])+str(kmesh[2])+"_basis_"+str(cell.basis)+"_"+bath_type+"_bath"+"magentics_"+magnetism+".dat"
    f_name = "SzSz_0"+str(n)+"_H_"+str(nsite)+"_d_"+str(d)+"_nshots_"+str(Nshots)+"and_Etot_rdm_E_all_ci_fci_GS.dat"
    if os.path.isfile(f_name):
        f = open(f_name, "a")
        # f.write(str(mf.e_tot)+"    "+str(cc.e_tot)+"    "+str(fci.e_tot)+"    "+str(emb_atom_fci.e_tot)+"    "+str(emb_fci.e_tot)+"    "+str(emb.e_tot)+"\n")
        # f.write(str(spin_spin_correlation(myhf.make_rdm1(),myhf.make_rdm2(),projectors[0],projectors[n]))+"    "+str(spin_spin_correlation(cc.make_rdm1(),cc.make_rdm2(),projectors[0],projectors[n]))+"    "+str(spin_spin_correlation(cisolver.make_rdm1(cisolver.ci, space.nact, space.naocc * 2),cisolver.make_rdm2(cisolver.ci, space.nact, space.naocc * 2),projectors[0],projectors[n]))+"    "+str(spin_spin_correlation(rdm1_sd,rdm2_sd,projectors[0],projectors[n]))+"    "+str(spin_spin_correlation(rdm1_sdt,rdm2_sdt,projectors[0],projectors[n]))+"     "+str(spin_spin_correlation(ci_rdm1_sd,ci_rdm2_sd,projectors[0],projectors[n]))+"    "+str(spin_spin_correlation(ci_rdm1_sdt,ci_rdm2_sdt,projectors[0],projectors[n]))+"    "+str(spin_spin_correlation_mixed(rdm1_qwc,-np_rdm2_qwc.transpose(0, 3, 2, 1),projectors[0],projectors[n]))+"\n")
        # f.write(str(spin_spin_correlation(myhf.make_rdm1(),myhf.make_rdm2(),projectors[0],projectors[n]))+"    "+str(spin_spin_correlation(cc.make_rdm1(),cc.make_rdm2(),projectors[0],projectors[n]))+"    "+str(spin_spin_correlation(cisolver.make_rdm1(cisolver.ci, space.nact, space.naocc * 2),cisolver.make_rdm2(cisolver.ci, space.nact, space.naocc * 2),projectors[0],projectors[n]))+"    "+str(spin_spin_correlation(rdm1_sd,rdm2_sd,projectors[0],projectors[n]))+"    "+str(spin_spin_correlation(rdm1_sdt,rdm2_sdt,projectors[0],projectors[n]))+"     "+str(spin_spin_correlation(ci_rdm1_sd,ci_rdm2_sd,projectors[0],projectors[n]))+"    "+str(spin_spin_correlation(ci_rdm1_sdt,ci_rdm2_sdt,projectors[0],projectors[n]))+"    "+str(spin_spin_correlation_mixed(rdm1_qwc,-np_rdm2_qwc.transpose(0, 3, 2, 1),projectors[0],projectors[n]))+"    "+str(e_fci)+"    "+str(ebcc_ccsd.energy() + myhf.e_tot)+"    "+str(ci_ebcc_ccsd.energy() + myhf.e_tot)+"\n")
        f.write(str(spin_spin_correlation(myhf.make_rdm1(),myhf.make_rdm2(),projectors[0],projectors[n]))+"    "+str(spin_spin_correlation(cc.make_rdm1(),cc.make_rdm2(),projectors[0],projectors[n]))+"    "+str(spin_spin_correlation(cisolver.make_rdm1(cisolver.ci, space.nact, space.naocc * 2),cisolver.make_rdm2(cisolver.ci, space.nact, space.naocc * 2),projectors[0],projectors[n]))+"    "+str(spin_spin_correlation(rdm1_sd,rdm2_sd,projectors[0],projectors[n]))+"     "+str(spin_spin_correlation(ci_rdm1_sd,ci_rdm2_sd,projectors[0],projectors[n]))+"    "+str(spin_spin_correlation(ci_rdm1_sdt,ci_rdm2_sdt,projectors[0],projectors[n]))+"    "+str(spin_spin_correlation(rdm1_qwc,np_qwc_rdm2_sum,projectors[0],projectors[n]))+"    "+str(e_fci)+"    "+str(ebcc_ccsd.energy() + myhf.e_tot)+"    "+str(ci_ebcc_ccsd.energy() + myhf.e_tot)+"    "+str(fci_rdm_energy) +"    "+str(qwc_rdm_energy) +"    "+str(ci_sd_rdm_energy) +"    "+str(ci_sdt_rdm_energy) +"    "+str(cs_sd_rdm_energy)+"    "+str(my_civec_cs[0][0])+"    "+str(my_civec_cs[1][0])+"    "+str(my_civec_cs[0][1])+"    "+str(my_civec_cs[1][1])+"    "+str(e_proj)+"\n")
        f.close()
    else:
        f = open(f_name, "a")
        # f.write("c0"+"    "+"e_fci"+"    "+"e_hf"+"    "+"vqe_energy"+"    "+"HF+CS"+"    "+"HF+meas_diag"+"    "+"E_pauli_all"+"\n")
        # f.write(str(c0)+"    "+str(e_fci)+"    "+str(myhf.e_tot)+"    "+str(vqe_energy)+"    "+str(e_proj_CS)+"    "+str(e_proj_meas_diag)+"    "+str(result_Pauli_all_comm)+"\n")
        # f.write("HF"+"    "+"CCSD"+"    "+"FCI"+"    "+"CS_CCSD"+"    "+"CS_CCSDT"+"    "+"FCI_CCSD"+"    "+"FCI_CCSDT"+"    "+"Pauli_group_qwc"+"     "+"e_fci"+"    "+"e_cs_ccsd"+"    e_fci_ccsd"+"    "+"fci_rdm_energy"+"    "+"qwc_rdm_energy"+"    "+"ci_sd_rdm_energy"+"    "+"ci_sdt_rdm_energy"+"    "+"cs_sd_rdm_energy"+"\n")
        f.write("HF"+"    "+"CCSD"+"    "+"FCI"+"    "+"CS_CCSD"+"    "+"FCI_CCSD"+"    "+"FCI_CCSDT"+"    "+"Pauli_group_qwc"+"     "+"e_fci"+"    "+"e_cs_ccsd"+"    e_fci_ccsd"+"    "+"fci_rdm_energy"+"    "+"qwc_rdm_energy"+"    "+"ci_sd_rdm_energy"+"    "+"ci_sdt_rdm_energy"+"    "+"cs_sd_rdm_energy"+"    "+"c0"+"    "+"c10"+"    "+"c01"+"    "+"c2"+"    "+"e_proj"+"\n")
        # f.write(str(spin_spin_correlation(myhf.make_rdm1(),myhf.make_rdm2(),projectors[0],projectors[n]))+"    "+str(spin_spin_correlation(cc.make_rdm1(),cc.make_rdm2(),projectors[0],projectors[n]))+"    "+str(spin_spin_correlation(cisolver.make_rdm1(cisolver.ci, space.nact, space.naocc * 2),cisolver.make_rdm2(cisolver.ci, space.nact, space.naocc * 2),projectors[0],projectors[n]))+"    "+str(spin_spin_correlation(rdm1_sd,rdm2_sd,projectors[0],projectors[n]))+"    "+str(spin_spin_correlation(rdm1_sdt,rdm2_sdt,projectors[0],projectors[n]))+"     "+str(spin_spin_correlation(ci_rdm1_sd,ci_rdm2_sd,projectors[0],projectors[n]))+"    "+str(spin_spin_correlation(ci_rdm1_sdt,ci_rdm2_sdt,projectors[0],projectors[n]))+"    "+str(spin_spin_correlation_mixed(rdm1_qwc,-np_rdm2_qwc.transpose(0, 3, 2, 1),projectors[0],projectors[n]))+"    "+str(e_fci)+"    "+str(ebcc_ccsd.energy() + myhf.e_tot)+"    "+str(ci_ebcc_ccsd.energy() + myhf.e_tot)+"\n")
        f.write(str(spin_spin_correlation(myhf.make_rdm1(),myhf.make_rdm2(),projectors[0],projectors[n]))+"    "+str(spin_spin_correlation(cc.make_rdm1(),cc.make_rdm2(),projectors[0],projectors[n]))+"    "+str(spin_spin_correlation(cisolver.make_rdm1(cisolver.ci, space.nact, space.naocc * 2),cisolver.make_rdm2(cisolver.ci, space.nact, space.naocc * 2),projectors[0],projectors[n]))+"    "+str(spin_spin_correlation(rdm1_sd,rdm2_sd,projectors[0],projectors[n]))+"     "+str(spin_spin_correlation(ci_rdm1_sd,ci_rdm2_sd,projectors[0],projectors[n]))+"    "+str(spin_spin_correlation(ci_rdm1_sdt,ci_rdm2_sdt,projectors[0],projectors[n]))+"    "+str(spin_spin_correlation(rdm1_qwc,np_qwc_rdm2_sum,projectors[0],projectors[n]))+"    "+str(e_fci)+"    "+str(ebcc_ccsd.energy() + myhf.e_tot)+"    "+str(ci_ebcc_ccsd.energy() + myhf.e_tot)+"    "+str(fci_rdm_energy) +"    "+str(qwc_rdm_energy) +"    "+str(ci_sd_rdm_energy) +"    "+str(ci_sdt_rdm_energy) +"    "+str(cs_sd_rdm_energy)+"    "+str(my_civec_cs[0][0])+"    "+str(my_civec_cs[1][0])+"    "+str(my_civec_cs[0][1])+"    "+str(my_civec_cs[1][1])+"    "+str(e_proj)+"\n")
        f.close()

