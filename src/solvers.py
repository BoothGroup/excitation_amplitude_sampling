from qiskit_nature.second_q.circuit.library.ansatzes import UCC
from qiskit_nature.second_q.circuit.library.initial_states import HartreeFock
from qiskit_aer.primitives import Estimator as AerEstimator
# from qiskit import Aer
from qiskit_algorithms.optimizers import COBYLA , SLSQP , L_BFGS_B, ADAM
from qiskit_algorithms import VQE
from qiskit.quantum_info import Statevector

import pyscf
from pyscf.fci import cistring
from itertools import permutations

import numpy as np


import scipy.optimize

import ffsim
import qiskit

def cbssp_unitary(stateA,stateB,imag=False):
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
            sp_circ.cx(a1[0],a1[n])

        sp_circ.x(a1[0])
        for n in np.arange(0,len(b1[:])):
            #print(n)
            sp_circ.cx(a1[0],b1[n])    
        sp_circ.x(a1[0])
    elif len(a1)==0 and len(b1)!=0:
        sp_circ.h(b1[0])
        if imag:
            sp_circ.s(b1[0])
        for n in np.arange(1,len(b1[:])):
            sp_circ.cx(b1[0],b1[n])
    return sp_circ,a1[0]

class vqe_solver():

    def __init__(self,qubit_ham,num_orbitals,num_particles,mapper,ansatz_label= "UCC"):
        self.qubit_ham=qubit_ham

        self.num_orbitals= num_orbitals
        self.num_particles=num_particles
        self.mapper=mapper

        self.estimator = AerEstimator(approximation=True)
        '''
        Constructing the ansatz 

        '''

        # Make a qiskit HF reference state

        self.initial_state = HartreeFock(
        num_spatial_orbitals= self.num_orbitals,
        num_particles=num_particles , #nsite(1,1)
        qubit_mapper=mapper,
        )


        print("vqe num_particles = ",num_particles)

        # UCC ansatz
        #print("np: ", num_particles)
        if ansatz_label == "UCC":
            self.ansatz = UCC(
                excitations="sd",
                num_particles= num_particles , # nelec
                num_spatial_orbitals=self.num_orbitals,
                initial_state=self.initial_state,
                qubit_mapper= mapper,
                reps=1,
            )




    def solve(self,solver_options):
        #  maxiter=1e3,random_initial_params=False, attempts=16
        # seed=algorithm_globals.random
        #qi = QuantumInstance(Aer.get_backend('statevector_simulator'), seed_transpiler=seed, seed_simulator=seed,shots=100000)

        # slsqp = SLSQP(maxiter=maxiter)
        # cob = COBYLA(maxiter= maxiter)
        # BFGS = L_BFGS_B(maxiter= maxiter)
        # adam = ADAM(maxiter=maxiter)
        if "maxiter" in solver_options:
            maxiter=solver_options["maxiter"]
        else:
            maxiter=1e3

        if "random_initial_params" in solver_options:
            random_initial_params=solver_options["random_initial_params"]
        else:
            random_initial_params=False

        if "attempts" in solver_options:
            attempts=solver_options["attempts"]
        else:
            attempts=16

        if "Statevector" in solver_options:
            state_vec=solver_options["Statevector"]
        else:
            state_vec=True
        
        
        cobyla = COBYLA(maxiter=maxiter)
        
        #print("pre vqe")
         
        vqe = VQE(self.estimator,self.ansatz, optimizer=cobyla, initial_point = np.zeros(self.ansatz.num_parameters))
        result = vqe.compute_minimum_eigenvalue(operator= self.qubit_ham)
        #print("post min eigen")
        #energy = result.optimal_value.real 

        self.vqe_energy = result.optimal_value.real
       
        #print("return optimal params")
        self.optimal_point=result.optimal_point


        if random_initial_params:
            for initial_param_count in range(attempts):
                #print("param count: ", initial_param_count)
                vqe = VQE(self.estimator,self.ansatz, optimizer=cobyla, initial_point = np.random.normal(0.0,np.pi/4.,self.ansatz.num_parameters))
                result = vqe.compute_minimum_eigenvalue(operator= self.qubit_ham)
                if result.optimal_value.real < self.vqe_energy:
                    #print("Found better than 0 initial params", self.vqe_energy, result.optimal_value.real)
                    self.vqe_energy = result.optimal_value.real 
                    self.optimal_point=result.optimal_point
        print("vqe_energy: ",self.vqe_energy) 
        print("optimal params: ", self.optimal_point)
        qubits = qiskit.QuantumRegister(2 * self.num_orbitals, name="q")

        if state_vec:
            return Statevector(self.ansatz.assign_parameters(self.optimal_point)),self.vqe_energy
        else:
            operator = self.ansatz.assign_parameters(self.optimal_point)
            circuit = qiskit.QuantumCircuit(qubits)
            sp_circ,a1 = cbssp_unitary("0"*(self.num_orbitals-self.num_particles[1]) + "1"*self.num_particles[1] + "0"*(self.num_orbitals-self.num_particles[0]) + "1"*self.num_particles[0],"0"*(self.num_orbitals+self.num_orbitals))
            circuit.append(sp_circ,qubits)
            circuit.append(operator,qubits)
            return circuit,self.vqe_energy


class ffsim_vqe_solver():
    def __init__(self,qubit_ham,norb,nocc,t2s):
        self.qubit_ham = qubit_ham
        self.norb = norb
        assert norb[0]==norb[1]
        self.nocc = nocc
        print("ffsim nocc: ",self.nocc)
        self.t2s = t2s

    def solve(self,solver_options):
        if solver_options["Statevector"]:
            state_vec=solver_options["Statevector"]
        else:
            state_vec=True
        
        n_reps=1

        
        pairs_aa = [(p, p + 1) for p in range(self.norb[0] - 1)]#+[(p, p + 2) for p in range(norb[0] - 2)]
        pairs_bb = [(p, p + 1) for p in range(self.norb[0] - 1)]#+[(p, p + 2) for p in range(norb[0] - 2)]
        pairs_ab = [(p, p) for p in range(self.norb[0])]#+[(p, p+1) for p in range(norb[0]-1)]
        interaction_pairs = (pairs_aa, pairs_ab, pairs_bb)

        operator = ffsim.UCJOpSpinUnbalanced.from_t_amplitudes(
                self.t2s, n_reps=n_reps, interaction_pairs=interaction_pairs
            )



        qubits = qiskit.QuantumRegister(2 * self.norb[0], name="q")

        # hamiltonian = ffsim.MolecularHamiltonian()


        # # Compute the energy ⟨ψ|H|ψ⟩ of the ansatz state
        # hamiltonian = ffsim.linear_operator(mol_hamiltonian, norb=norb, nelec=nelec)
        # energy = np.real(np.vdot(ansatz_state, hamiltonian @ ansatz_state))
        # print(f"Energy at initialization: {energy}")



        


        def fun(x):
            # Initialize the ansatz operator from the parameter vector
            operator = ffsim.UCJOpSpinUnbalanced.from_parameters(x, norb=self.norb[0], n_reps=n_reps)
            # Apply the ansatz operator to the reference state
            circuit = qiskit.QuantumCircuit(qubits)
            circuit.append(ffsim.qiskit.PrepareHartreeFockJW(self.norb[0], self.nocc),qubits)#[self.nocc[1],self.nocc[0]]
            circuit.append(ffsim.qiskit.UCJOpSpinUnbalancedJW(operator),qubits)

            # final_state = ffsim.apply_unitary(reference_state, operator, norb=norb[0], nelec=nocc)
            # Return the energy ⟨ψ|H|ψ⟩ of the ansatz state
            final_state = np.array(Statevector(circuit))
            return np.real(np.vdot(final_state, self.qubit_ham.to_matrix() @ final_state))


        result = scipy.optimize.minimize(
            fun, x0=operator.to_parameters(), method="COBYLA", options=dict(maxiter=4e2), tol=1e-3
        )


        operator = ffsim.UCJOpSpinUnbalanced.from_parameters(result.x, norb=self.norb[0], n_reps=n_reps)
        
        print("ffsim energy: ",result.fun)

        params = result.x    
        if state_vec:
            circuit = qiskit.QuantumCircuit(qubits)
            circuit.append(ffsim.qiskit.PrepareHartreeFockJW(self.norb[0], self.nocc),qubits)
            circuit.append(ffsim.qiskit.UCJOpSpinUnbalancedJW(operator),qubits)
            return Statevector(circuit),result.fun
        else:
            circuit = qiskit.QuantumCircuit(qubits)
            sp_circ,a1 = cbssp_unitary("0"*(self.norb[1]-self.nocc[1]) + "1"*self.nocc[1] + "0"*(self.norb[0]-self.nocc[0]) + "1"*self.nocc[2],"0"*(self.norb[1]+self.norb[0]))
            circuit.append(sp_circ,qubits)#[self.norb[0]:]+qubits[:self.norb[0]]
            circuit.append(ffsim.qiskit.UCJOpSpinUnbalancedJW(operator),qubits)#[self.norb[0]:]+qubits[:self.norb[0]]
            return circuit,result.fun


class fci_solver():

    def __init__(self,h1e,h2e,norb,nocc):
        self.h1e = h1e
        self.h2e = h2e
        # print("h1e",h1e)
        # print("h2e",h2e)
        self.norb = norb
        # print("norb: ",norb)
        # nelec = mf.mol.nelectron
        self.nocc = nocc#(int(sum(mf.get_occ()[0])),int(sum(mf.get_occ()[1])))
        self.nvir = (norb[0] - nocc[0],norb[1] - nocc[1])
        
        self.nelec = nocc

        self.norba, self.norbb = self.norb
        self.nocca, self.noccb = self.nocc
        self.nvira, self.nvirb = self.nvir
        assert self.norb[0]==self.norb[1]
        

    
    def solve(self,solver_options):
        # t1addra, t1signa = pyscf.ci.cisd.tn_addrs_signs(norba, nocca, 1)
        # t1addrb, t1signb = pyscf.ci.cisd.tn_addrs_signs(norbb, noccb, 1)
        # t2addra, t2signa = pyscf.ci.cisd.tn_addrs_signs(norba, nocca, 2)
        # t2addrb, t2signb = pyscf.ci.cisd.tn_addrs_signs(norbb, noccb, 2)
        
        fci_energy, civec = pyscf.fci.direct_uhf.kernel(self.h1e, self.h2e, self.norba, self.nelec, conv_tol=1.e-14)
        fermi_vac = "0"*(self.norbb-self.noccb) + "1"*(self.noccb) + "0"*(self.norba-self.nocca) + "1"*(self.nocca)
        print("energy fci: ",fci_energy)

        a_perms = list(set([''.join(p) for p in permutations(fermi_vac[len(fermi_vac)//2:])]))
        b_perms = list(set([''.join(p) for p in permutations(fermi_vac[:len(fermi_vac)//2])]))
        SV_exact=np.zeros(2**(self.norba+self.norbb))
        all_addr = [y+x for x in a_perms for y in b_perms]
        for addr_string in all_addr:
            addr = (cistring.str2addr(self.norba, self.nocca, bin(int(addr_string[len(addr_string)//2:],2))),cistring.str2addr(self.norbb, self.noccb, bin(int(addr_string[:len(addr_string)//2],2))))
            if abs(civec[addr])>1.e-10:
                # if abs(SV_exact[int(addr_string,2)]-civec[addr])>1e-4:
                #     print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                # print(SV_exact[int(addr_string,2)],civec[addr])
                SV_exact[int(addr_string,2)] = civec[addr]        
        return Statevector(SV_exact), fci_energy        