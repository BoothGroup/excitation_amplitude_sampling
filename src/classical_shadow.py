from sys import exit
import numpy as np
import qiskit 
from qiskit.quantum_info import random_pauli
import random
from copy import deepcopy
import qiskit.quantum_info
import qulacs
import sys
import math
from qiskit_nature.second_q.operators import FermionicOp
# from qiskit.extensions import HamiltonianGate
from qiskit.circuit.library import HamiltonianGate
from qiskit.quantum_info import Pauli
from qiskit_aer import AerSimulator

from qulacs.converter import convert_QASM_to_qulacs_circuit
from qulacs.gate import PauliRotation
import textwrap
from qiskit import qasm2
from src import stabilizer
import stim
# import julia
# julia.install()
# from julia.api import Julia
# jl = Julia(compiled_modules=False)
# from julia import Main
# from julia import ITensors

from itertools import groupby
import qiskit_aer.noise as noise
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeBrisbane
from qiskit import transpile

brisbane_backend = FakeBrisbane()
brisbane_noise_model = NoiseModel.from_backend(brisbane_backend)
# Get basis gates from noise model
eagle_basis_gates = brisbane_noise_model.basis_gates

# Get coupling map from backend
coupling_map = brisbane_backend.configuration().coupling_map

def allEqual(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)

def allUnique(x):
    seen = set()
    return not any(i in seen or seen.add(i) for i in x)

def to_matrix(alist, n,m):
    return [alist[i:i+n] for i in range(0, n*m, n)]

# def create_householder(v):
#     dim = v.size
#     phase=np.angle(v[0])
#     v_norm=v/np.dot(v, v.conjugate().transpose())
#     v_norm=cmath.rect(1, -phase)* v_norm
#     #print("normed: ",v_norm)
#     # Return identity if v is a multiple of e1
#     if v_norm[0] == 1:
#         return np.identity(dim)*cmath.rect(1, phase)
#     e1 = np.zeros(dim)
#     e1[0] = 1
#     w = v_norm - e1
#     #print("W: ",w)
#     return cmath.rect(1, phase)*(np.identity(dim) - 2*((np.outer(w, w.transpose().conjugate()))/(np.dot(w, w.conjugate().transpose()))))



def givens_rotation(i, j, theta, n):
    """
    Construct a Givens rotation matrix of size n x n.
    """
    G = np.eye(n)
    c = np.cos(theta)
    s = np.sin(theta)
    G[i, i] = c
    G[j, j] = c
    G[i, j] = s
    G[j, i] = -s
    return G

def decompose_to_givens_rotation(U):
    """
    Decompose the given special orthogonal matrix U into Givens rotations.
    """
    n = U.shape[0]  # Size of the matrix
    rotations = []  # List to store Givens rotations

    # Initialize U as identity matrix
    U_decomposed = np.eye(n)

    # Iterate to make U diagonal
    for j in range(n):
        for i in range(0, j, +1):  # Iterate over non-zero off-diagonal elements
            #print(i,j)
            if abs(U[j,i]) > 1e-9:  # Check if the element is non-zero
                #print(i,j,U[j,i], U[j, j],U[i, i],np.arctan2(U[j,i], U[j, j]))
                # Compute the angle for the Givens rotation
                if abs(U[j,j])<1e-9:
                    theta = np.arctan2(U[j,i], abs(U[j,j]))
                else:    
                    theta = np.arctan2(U[j,i], (U[j,j]))
                #print("c,s: ",np.cos(theta), np.sin(theta), )
                # Construct Givens rotation matrix
                G = givens_rotation(i, j, theta, n)
                # Update U_decomposed
                U_decomposed = np.dot(U_decomposed, G.T)
                # Update U with Givens rotation
                U = np.dot(G, U)
                #print("U: ",U)#U[j,i],U[i,i],U[j,j])
                # Store Givens rotation in the list
                rotations.append((j, i, theta))
    # print("U final: ", U)
    # print("U final: ", np.diag(U))
    # print("U decomp final: ", U_decomposed)
    if -1 in list(np.diag(U)):
        negative_indices = [i for i, x in enumerate(list(np.diag(U))) if abs(x+1)<1e-9]
        if len(negative_indices)%2 == 0:
            for i_neg in np.arange(0,len(negative_indices),2):
                rotations.append((negative_indices[i_neg],negative_indices[i_neg+1],np.pi))
        else:
            sys.exit("trying to decompose a negative determinant orthogonal matrix not implemented")        
    return rotations





def maj_num_to_matrix(maj1,maj2,mapper,nqubits):
    if maj1%2==0 and maj2%2 == 0:
        #print(f"G({i}, {j}, {theta})")
        my_str1=f"-_{math.floor(maj1/2.)} -_{math.floor(maj2/2.)}"
        my_str2=f"+_{math.floor(maj1/2.)} -_{math.floor(maj2/2.)}"
        my_str3=f"-_{math.floor(maj1/2.)} +_{math.floor(maj2/2.)}"
        my_str4=f"+_{math.floor(maj1/2.)} +_{math.floor(maj2/2.)}"
        f_op = FermionicOp({my_str1: +1., my_str2: +1., my_str3: +1., my_str4: +1.},nqubits)
    elif maj1%2==1 and maj2%2 == 1:
        #print(f"G({i}, {j}, {theta})")
        my_str1=f"-_{math.floor(maj1/2.)} -_{math.floor(maj2/2.)}"
        my_str2=f"+_{math.floor(maj1/2.)} -_{math.floor(maj2/2.)}"
        my_str3=f"-_{math.floor(maj1/2.)} +_{math.floor(maj2/2.)}"
        my_str4=f"+_{math.floor(maj1/2.)} +_{math.floor(maj2/2.)}"
        f_op = FermionicOp({my_str1: -1., my_str2: +1., my_str3: +1., my_str4: -1.},nqubits)
    elif maj1%2==0 and maj2%2 == 1:
        #print(f"G({i}, {j}, {theta})")
        my_str1=f"-_{math.floor(maj1/2.)} -_{math.floor(maj2/2.)}"
        my_str2=f"+_{math.floor(maj1/2.)} -_{math.floor(maj2/2.)}"
        my_str3=f"-_{math.floor(maj1/2.)} +_{math.floor(maj2/2.)}"
        my_str4=f"+_{math.floor(maj1/2.)} +_{math.floor(maj2/2.)}"
        f_op = FermionicOp({my_str1: -complex(0,1), my_str2: -complex(0,1), my_str3: +complex(0,1), my_str4: +complex(0,1)},nqubits)
    elif maj1%2==1 and maj2%2 == 0:
        #print(f"G({i}, {j}, {theta})")
        my_str1=f"-_{math.floor(maj1/2.)} -_{math.floor(maj2/2.)}"
        my_str2=f"+_{math.floor(maj1/2.)} -_{math.floor(maj2/2.)}"
        my_str3=f"-_{math.floor(maj1/2.)} +_{math.floor(maj2/2.)}"
        my_str4=f"+_{math.floor(maj1/2.)} +_{math.floor(maj2/2.)}"
        f_op = FermionicOp({my_str1: -complex(0,1), my_str2: +complex(0,1), my_str3: -complex(0,1), my_str4: +complex(0,1)},nqubits)
    #print(f_op)
    # print("RDM_paulis: ",mapper.map(f_op).paulis,mapper.map(f_op).coeffs)    
    RDM_pauli=Pauli(mapper.map(f_op).paulis[0]).to_matrix()*mapper.map(f_op).coeffs[0]
    return RDM_pauli



def get_noise_model(alpha):
  # Brisbane 2 Jan 2025
  t1 = 234.2e-6 #micro S
  t2 =  119.27e-6 #micro
  sx_time = 0.06e-6 #micro (60 nano)
  ecr_time = 0.66e-6 #660 nano

  ro_err = alpha * 1.57e-2
  ecr_err = alpha * 8.707e-3
  sx_err = alpha * 2.547e-4


  ro_probabilities = [[1.-ro_err,ro_err],[ro_err,1.-ro_err]]
  ro_err_model = noise.ReadoutError(ro_probabilities)

  sx_err_model = noise.depolarizing_error(sx_err, 1)
  ecr_err_model = noise.depolarizing_error(ecr_err, 2)

  # sx_thermal_err_model = noise.thermal_relaxation_error(t1,t2,sx_time)
  # ecr_thermal_err_model = noise.thermal_relaxation_error(t1,t2,ecr_time).expand(noise.thermal_relaxation_error(t1,t2,ecr_time))

  sx_thermal_err_model = noise.QuantumError(qiskit.quantum_info.SuperOp([[1.00000000e+00+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            alpha*2.56158475e-04+0.j],
          [0.00000000e+00+0.j, 1. - alpha*(1. - 9.99497066e-01)+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j],
          [0.00000000e+00+0.j, 0.00000000e+00+0.j, 1. - alpha*(1. - 9.99497066e-01)+0.j,
            0.00000000e+00+0.j],
          [0.00000000e+00+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            1. - alpha*(1. - 9.99743842e-01)+0.j]],
          input_dims=(2,), output_dims=(2,)))

  ecr_thermal_err_model = noise.QuantumError(qiskit.quantum_info.SuperOp([[1.00000000e+00+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j, 0.00000000e+00+0.j, alpha*2.81413706e-03+0.j,
            0.00000000e+00+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j, alpha*2.81413706e-03+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            alpha*7.91936737e-06+0.j],
          [0.00000000e+00+0.j, 1. - alpha*(1. - 9.94481619e-01)+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j, 0.00000000e+00+0.j, alpha*2.79860758e-03+0.j,
            0.00000000e+00+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j],
          [0.00000000e+00+0.j, 0.00000000e+00+0.j, 1. - alpha*(1. - 9.94481619e-01)+0.j,
            0.00000000e+00+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j, alpha*2.79860758e-03+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j],
          [0.00000000e+00+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            1. - alpha*(1. - 9.88993691e-01)+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j],
          [0.00000000e+00+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j, 1. - alpha*(1. - 9.94481619e-01)+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j, 0.00000000e+00+0.j, alpha*2.79860758e-03+0.j,
            0.00000000e+00+0.j],
          [0.00000000e+00+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j, 0.00000000e+00+0.j, 1. - alpha*(1. - 9.97185863e-01)+0.j,
            0.00000000e+00+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            alpha*2.80621769e-03+0.j],
          [0.00000000e+00+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            1. - alpha*(1. - 9.88993691e-01)+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j],
          [0.00000000e+00+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j, 1. - alpha*(1. - 9.91683012e-01)+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j],
          [0.00000000e+00+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j, 0.00000000e+00+0.j, 1. - alpha*(1. - 9.94481619e-01)+0.j,
            0.00000000e+00+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j, alpha*2.79860758e-03+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j],
          [0.00000000e+00+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            1. - alpha*(1. - 9.88993691e-01)+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j],
          [0.00000000e+00+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j, 1. - alpha*(1. - 9.97185863e-01)+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            alpha*2.80621769e-03+0.j],
          [0.00000000e+00+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j, 0.00000000e+00+0.j, 1. - alpha*(1. - 9.91683012e-01)+0.j,
            0.00000000e+00+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j],
          [0.00000000e+00+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            1. - alpha*(1. - 9.88993691e-01)+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j],
          [0.00000000e+00+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j, 1. - alpha*(1. - 9.91683012e-01)+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j],
          [0.00000000e+00+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j, 0.00000000e+00+0.j, 1. - alpha*(1. - 9.91683012e-01)+0.j,
            0.00000000e+00+0.j],
          [0.00000000e+00+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            0.00000000e+00+0.j, 0.00000000e+00+0.j, 0.00000000e+00+0.j,
            1. - alpha*(1. - 9.94379645e-01)+0.j]],
          input_dims=(2, 2), output_dims=(2, 2)))

  custom_noise_model = noise.NoiseModel(basis_gates=eagle_basis_gates)
  custom_noise_model.add_all_qubit_quantum_error(sx_err_model,["sx"])
  custom_noise_model.add_all_qubit_quantum_error(ecr_err_model,["ecr"])
  custom_noise_model.add_all_qubit_quantum_error(sx_thermal_err_model,["sx"])
  custom_noise_model.add_all_qubit_quantum_error(ecr_thermal_err_model,["ecr"])

  custom_noise_model.add_all_qubit_readout_error(ro_err_model)

  return custom_noise_model




class Classical_shadow():

    def __init__(self,channel,num_qubits,meas,Us):
        self.channel = channel
        self.shadow_measurements_central=meas
        self.shadow_Us_central=Us
        self.num_qubits = num_qubits

    def calc_shadow_lists(self,state_as_vector,no_estimators_K=1,N_per_estimator=200,use_QASM_qulacs_converter=True, mapper=None):
        #print("got to here")
        #print("K: ",no_estimators_K)
        #print("N: ",N_per_estimator)
        # self.channel=channel
        self.backend = AerSimulator(method="statevector",precision="double")#qiskit.Aer.get_backend('statevector_simulator')
        self.total_samples=no_estimators_K*N_per_estimator
        #print("got to here 2")
        self.no_estimators_K = no_estimators_K
        self.N_per_estimator = N_per_estimator
        self.shadow_list=[]
        self.shadow_measurements=[]
        self.shadow_Us=[]
        self.num_qubits=state_as_vector.num_qubits
        for n in range(no_estimators_K):
            #self.shadow_list.append(np.zeros((2**int(self.num_qubits),2**int(self.num_qubits)),dtype=complex))
            self.shadow_measurements.append([])
            self.shadow_Us.append([])                        
        # self.state_as_unitary = state_as_unitary
        # optimal_circ_save=state_as_unitary.copy()
        self.shadow_measurements_central=[]
        self.shadow_Us_central=[]
        self.identity_1qb=np.identity(2)
        state_as_vector_list=list(state_as_vector)
        state_as_vector_reordered=[0.]*len(state_as_vector_list)
        for n in range(len(state_as_vector_list)):
            bins="0"*(self.num_qubits-len(str(bin(n))[2:]))+str(bin(n))[2:]
            state_as_vector_reordered[int(bins[::-1],2)]=state_as_vector_list[n]
        if self.channel == "Clifford":
            # print("Begin shadow collection: ", "N samples: ",self.total_samples)
            for p in range(no_estimators_K):
                for shadow_collection_counter in range(N_per_estimator):
                    
                    # elif simulator == "MPS" or "mps":

                    if use_QASM_qulacs_converter:
                        # rand_clifford=qiskit.quantum_info.random_clifford(self.num_qubits,seed=np.random.randint(1e16))
                        random_tableau = stim.Tableau.random(self.num_qubits)
                        symp_mat = [str(x).replace("_","I") for x in[random_tableau.x_output(k) for k in range(len(random_tableau))]] + [str(x).replace("_","I") for x in random_tableau.to_stabilizers()]
                        rand_clifford = qiskit.quantum_info.Clifford(symp_mat)
                        rand_cliff_circ = rand_clifford.to_circuit()
                        # qasm = rand_cliff_circ.qasm() 
                        qasm = qasm2.dumps(rand_cliff_circ) ###convert clifford circuit to qasm
                        qasm = textwrap.dedent(qasm).strip() ### remove whitespace
                        circuit = convert_QASM_to_qulacs_circuit(qasm.splitlines())  ### convert qasm string to qulacs circuit
                        
                        state = qulacs.QuantumState(self.num_qubits)
                        state.load(state=state_as_vector_list)  ### initialise state with given statevector
                        #gate_initial.update_quantum_state(state)
                        circuit.update_quantum_state(state)   ### apply circuit to statevector
                        data = state.sampling(1,random_seed=np.random.randint(1e9)) ###collect a single sample
                        data=str([format(value, "b").zfill(self.num_qubits) for value in data][0]) ###convert integer to a binary string filling trailing zeros if necessary
                        

                    else:
                        # rand_clifford=qiskit.quantum_info.random_clifford(self.num_qubits)
                        random_tableau = stim.Tableau.random(self.num_qubits)
                        symp_mat = [str(x).replace("_","I") for x in[random_tableau.x_output(k) for k in range(len(random_tableau))]] + [str(x).replace("_","I") for x in random_tableau.to_stabilizers()]
                        rand_clifford = qiskit.quantum_info.Clifford(symp_mat)
                        #print(rand_clifford)
                        #print(type(rand_clifford.to_matrix()[0][0]))
                        
                        gate_cliff=qulacs.gate.DenseMatrix([int(list_el) for list_el in list(np.arange(self.num_qubits-1,-0.1,-1))],rand_clifford.to_matrix())
                        #gate_cliff=qulacs.gate.DenseMatrix([int(list_el) for list_el in list(np.arange(0,self.num_qubits))],rand_clifford.to_matrix())
                        
                        #gate_initial=qulacs.gate.DenseMatrix([int(list_el) for list_el in list(np.arange(self.num_qubits-1,-0.1,-1))],qiskit.quantum_info.Operator(state_as_unitary).data)
                        
                        state = qulacs.QuantumState(self.num_qubits)
                        state.load(state=state_as_vector_reordered)
                        #gate_initial.update_quantum_state(state)
                        gate_cliff.update_quantum_state(state)
                        data = state.sampling(1)
                        data=str([format(value, "b").zfill(self.num_qubits) for value in data][0])[::-1]



                    #print("data: ",data)
                    self.shadow_measurements[p].append(data)
                    self.shadow_Us[p].append(rand_clifford)
                    
                    #         measured_SV=qiskit.quantum_info.Statevector.from_label(data)
                    #print(measured_SV)

                    #meas_circuit=state_as_unitary.decompose(reps=3).compose(rand_clifford.to_circuit())
                    #meas_circuit.measure_all()
                    #job = self.backend.run(meas_circuit,shots=1)

                    # meas_circuit=qiskit.circuit.QuantumCircuit(int(self.num_qubits))
                    # meas_circuit.initialize(params=state_as_vector)
                    # meas_circuit.compose(rand_clifford.to_circuit(),inplace=True)
                    # #cliff_op=rand_clifford.to_instruction()
                    # #print("no qubits: ",cliff_op.num_qubits, "  ",meas_circuit.num_qubits)
                    # #meas_circuit.append(cliff_op,range(meas_circuit.num_qubits))
                    # meas_circuit.measure_all()
                    # job = self.backend.run(meas_circuit,shots=1)
                    
                    # result = job.result()
                    #measured_SV=qiskit.quantum_info.Statevector.from_label(list(result.get_counts().keys())[0])#

                    #         measurement_DM=qiskit.quantum_info.DensityMatrix(measured_SV.evolve(rand_clifford.adjoint().to_matrix()))####.to_circuit().inverse()))#.inverse()))
                    #         M_m1_measurement_DM=(((2**self.num_qubits)  +  1)*np.array(measurement_DM)) - self.identity
                    
                    #         self.shadow_list[p]+=(M_m1_measurement_DM)
                    ##if shadow_collection_counter>0 and shadow_collection_counter%((no_estimators_K*N_per_estimator)/10) == 0:
                    ##    print(int(shadow_collection_counter/((no_estimators_K*N_per_estimator)/10)),"/10")
            self.shadow_measurements_central=np.reshape(self.shadow_measurements,-1)
            if no_estimators_K==1 and N_per_estimator==1:
                self.shadow_Us_central=self.shadow_Us[0][0]
            else:        
                # self.shadow_Us_central=np.reshape(self.shadow_Us,-1)
                self.shadow_Us_central = [x1 for xs in self.shadow_Us for x1 in xs]

        elif self.channel == "Pauli":
            print("Begin shadow collection: ", "N samples: ",self.total_samples)
            #for shadow_collection_counter in range(no_estimators_K*N_per_estimator):
            for p in range(no_estimators_K):
                for shadow_collection_counter in range(N_per_estimator):
                    pauli_meas=["X","Y","Z"]
                    U_string1=""
                    U_string2=""
                    for qubit_no in range(self.num_qubits):
                        meas_basis_int=random.randint(0,2)
                        if meas_basis_int == 0:
                            U_string1 += "H"
                            U_string2 += "I"

                        elif meas_basis_int == 1:
                            U_string1 += "H"
                            U_string2 += "S"    
                        elif meas_basis_int == 2:
                            U_string1 += "I"
                            U_string2 += "I"
                        else:
                            sys.exit()       


                    
                    rand_clifford1=qiskit.quantum_info.Clifford.from_label(U_string1)
                    rand_clifford2=qiskit.quantum_info.Clifford.from_label(U_string2)
                    rand_clifford= rand_clifford2.adjoint().compose(rand_clifford1)#np.matmul(rand_clifford1.to_matrix(),rand_clifford2.to_matrix().conjugate().transpose())

                    #rand_clifford=rand_clifford2.to_circuit().inverse().compose(rand_clifford1.to_circuit())

                    gate_cliff=qulacs.gate.DenseMatrix([int(list_el) for list_el in list(np.arange(self.num_qubits-1,-0.1,-1))],rand_clifford.to_matrix())
                    #gate_initial=qulacs.gate.DenseMatrix([int(list_el) for list_el in list(np.arange(self.num_qubits-1,-0.1,-1))],qiskit.quantum_info.Operator(state_as_unitary).data)

                    state = qulacs.QuantumState(self.num_qubits)
                    #state.set_zero_state()
                    #gate_initial.update_quantum_state(state)
                    state.load(state=state_as_vector_reordered)
                    gate_cliff.update_quantum_state(state)
                    data = state.sampling(1)
                    data=str([format(value, "b").zfill(self.num_qubits) for value in data][0])[::-1]

                    # meas_circuit=state_as_unitary.decompose(reps=3).compose(rand_clifford)

                    # meas_circuit.measure_all()

                    # job = self.backend.run(meas_circuit,shots=1)
                    
                    # meas_circuit=qiskit.circuit.QuantumCircuit(int(self.num_qubits))
                    # meas_circuit.initialize(params=state_as_vector)
                    # #meas_circuit.compose(rand_clifford,inplace=True)
                    # meas_circuit.compose(rand_clifford,inplace=True)
                    # meas_circuit.measure_all()
                    # job = self.backend.run(meas_circuit,shots=1)
                    
                    # result = job.result()
                    self.shadow_measurements[p].append(data)
                    self.shadow_Us[p].append([U_string1,U_string2])

                    #            SV_qubit_list=[]
                    #            DM_qubit_list=[]
                    #for qubit_no in range(len(list(result.get_counts().keys())[0])):
                    #            for qubit_no in range(len(data)):
                        #print("qubit no: ",qubit_no)
                        # SV=qiskit.quantum_info.Statevector.from_label(list(result.get_counts().keys())[0][qubit_no])
                        #            SV=qiskit.quantum_info.Statevector.from_label(data[qubit_no])
                        #            SV_qubit_list.append(SV)
                        #            SV=SV.evolve(qiskit.quantum_info.Clifford.from_label(U_string1[qubit_no]))
                        #            SV=SV.evolve(qiskit.quantum_info.Clifford.from_label(U_string2[qubit_no]))
                        
                        #            DM=qiskit.quantum_info.DensityMatrix(SV)
                        #            DM_qubit_list.append(3.*DM - self.identity_1qb)    

                    #            DM_full=deepcopy(DM_qubit_list[0])
                    #            for qubit_no in np.arange(1,len(DM_qubit_list)):
                        #            DM_full=DM_full.tensor(DM_qubit_list[qubit_no])
                        

                    ##self.shadow_list.append(np.array(DM_full))
                    #            self.shadow_list[p]+=(np.array(DM_full))
                #if shadow_collection_counter>0 and shadow_collection_counter%((no_estimators_K*N_per_estimator)/10) == 0:
                #    print(int(shadow_collection_counter/((no_estimators_K*N_per_estimator)/10)),"/10")
            self.shadow_measurements_central=np.reshape(self.shadow_measurements,-1)
            # self.shadow_Us_central=np.reshape(self.shadow_Us,-1)
            self.shadow_Us_central = [x1 for xs in self.shadow_Us for x1 in xs]
        elif self.channel == "Alternating":
            ### https://arxiv.org/pdf/2307.12912 ###
            print("Begin shadow collection: ", "N samples: ",self.total_samples)
            #for shadow_collection_counter in range(no_estimators_K*N_per_estimator):
            for p in range(no_estimators_K):
                for shadow_collection_counter in range(N_per_estimator):
                    P = np.eye(2*self.num_qubits)
                    np.random.shuffle(P) 
                    while np.linalg.det(P) == -1: #want det == 1
                        np.random.shuffle(P)
                     
                    rotations=decompose_to_givens_rotation(P)
                    #print("rotations: ",rotations)    
                    f_op_tot=[]#FermionicOp({"+_0 -_0": 0})
                    for  i, j, theta in rotations[:]: #create mapping for Majorana operators using the definition in terms of C/A operators and the qiskit mapper
                        if i%2==0 and j%2 == 0:
                            #print(f"G({i}, {j}, {theta})")
                            my_str1=f"-_{math.floor(i/2.)} -_{math.floor(j/2.)}"
                            my_str2=f"+_{math.floor(i/2.)} -_{math.floor(j/2.)}"
                            my_str3=f"-_{math.floor(i/2.)} +_{math.floor(j/2.)}"
                            my_str4=f"+_{math.floor(i/2.)} +_{math.floor(j/2.)}"
                            f_op = FermionicOp({my_str1: +1., my_str2: +1., my_str3: +1., my_str4: +1.},int(len(P)/2))
                            f_op_tot.append([(theta/2.)*f_op,int(i/2.),int(j/2.)])
                        elif i%2==1 and j%2 == 1:
                            #print(f"G({i}, {j}, {theta})")
                            my_str1=f"-_{math.floor(i/2.)} -_{math.floor(j/2.)}"
                            my_str2=f"+_{math.floor(i/2.)} -_{math.floor(j/2.)}"
                            my_str3=f"-_{math.floor(i/2.)} +_{math.floor(j/2.)}"
                            my_str4=f"+_{math.floor(i/2.)} +_{math.floor(j/2.)}"
                            f_op = FermionicOp({my_str1: -1., my_str2: +1., my_str3: +1., my_str4: -1.},int(len(P)/2))
                            f_op_tot.append([(theta/2.)*f_op,int(i/2.),int(j/2.)])
                        elif i%2==0 and j%2 == 1:
                            #print(f"G({i}, {j}, {theta})")
                            my_str1=f"-_{math.floor(i/2.)} -_{math.floor(j/2.)}"
                            my_str2=f"+_{math.floor(i/2.)} -_{math.floor(j/2.)}"
                            my_str3=f"-_{math.floor(i/2.)} +_{math.floor(j/2.)}"
                            my_str4=f"+_{math.floor(i/2.)} +_{math.floor(j/2.)}"
                            f_op = FermionicOp({my_str1: -complex(0,1), my_str2: -complex(0,1), my_str3: +complex(0,1), my_str4: +complex(0,1)},int(len(P)/2))
                            # print(complex(0,1)*f_op)  
                            f_op_tot.append([(theta/2.)*f_op,int(i/2.),int(j/2.)])
                        elif i%2==1 and j%2 == 0:
                            #print(f"G({i}, {j}, {theta})")
                            my_str1=f"-_{math.floor(i/2.)} -_{math.floor(j/2.)}"
                            my_str2=f"+_{math.floor(i/2.)} -_{math.floor(j/2.)}"
                            my_str3=f"-_{math.floor(i/2.)} +_{math.floor(j/2.)}"
                            my_str4=f"+_{math.floor(i/2.)} +_{math.floor(j/2.)}"
                            f_op = FermionicOp({my_str1: -complex(0,1), my_str2: +complex(0,1), my_str3: -complex(0,1), my_str4: +complex(0,1)},int(len(P)/2))
                            # print(complex(0,1)*f_op)
                            f_op_tot.append([(theta/2.)*f_op,int(i/2.),int(j/2.)])
                    #print("f_op_tot: ",f_op_tot)


                    # U=np.eye(2**int(len(P)/2.))
                    # for p_op in f_op_tot:
                    #     jwm_p_op=mapper.map(p_op[0])#
                    #     if len(jwm_p_op.paulis)>1:
                    #         sys.exit("more than one pauli in mapped Given's rotation")
                    #     V= HamiltonianGate(jwm_p_op.paulis[0], -jwm_p_op.coeffs[0].imag).to_matrix()#np.cosh(jwm_p_op.coeffs[0].imag)*np.eye(2**int(len(P)/2.)) + np.sinh(jwm_p_op.coeffs[0].imag) * Pauli(jwm_p_op.paulis[0]).to_matrix()
                    #     U= U @ V


                    def qiskit_pauli_str_to_qulacs_target(pauli_str):
                        pauli = list(pauli_str)
                        target_list=[]
                        pauli_index=[]
                        for n in range(len(pauli)):
                            if pauli[n] == "X":
                                target_list.append(n)
                                pauli_index.append(1)
                            elif pauli[n] == "Y":
                                target_list.append(n)
                                pauli_index.append(2)   
                            elif pauli[n] == "Z":
                                target_list.append(n)
                                pauli_index.append(3)    
                        return pauli_index,target_list
                                                        
                    qlcs_qc=qulacs.QuantumCircuit(self.num_qubits)
                    for p_op in f_op_tot[::-1]:
                        jwm_p_op=mapper.map(p_op[0])#
                        if len(jwm_p_op.paulis)>1:
                            sys.exit("more than one pauli in mapped Given's rotation")
                        #print(jwm_p_op.paulis[0],jwm_p_op.coeffs[0], p_op[1],p_op[2])
                        # print(jwm_p_op.paulis[0], jwm_p_op.coeffs[0].imag)
                        # V= HamiltonianGate(jwm_p_op.paulis[0], -jwm_p_op.coeffs[0].imag)#np.cosh(jwm_p_op.coeffs[0].imag)*np.eye(2**int(len(P)/2.)) + np.sinh(jwm_p_op.coeffs[0].imag) * Pauli(jwm_p_op.paulis[0]).to_matrix()
                        pauli_index,target_list = qiskit_pauli_str_to_qulacs_target(jwm_p_op.paulis[0].to_label())
                        rot_gate = PauliRotation(target_list,pauli_index,angle=2.*jwm_p_op.coeffs[0].imag)
                        qlcs_qc.add_gate(rot_gate)



                    gate_cliff=  qlcs_qc #qulacs.gate.DenseMatrix([int(list_el) for list_el in list(np.arange(self.num_qubits-1,-0.1,-1))],U)
                    #gate_initial=qulacs.gate.DenseMatrix([int(list_el) for list_el in list(np.arange(self.num_qubits-1,-0.1,-1))],qiskit.quantum_info.Operator(state_as_unitary).data)

                    state = qulacs.QuantumState(self.num_qubits)
                    #state.set_zero_state()
                    #gate_initial.update_quantum_state(state)
                    state.load(state=state_as_vector_reordered)
                    # print(state,state.get_squared_norm())


                    gate_cliff.update_quantum_state(state)
                    # print(state,state.get_squared_norm())
                    # print(state.sampling(20))
                    data = state.sampling(1)


                    data=str([format(value, "b").zfill(self.num_qubits) for value in data][0])[::-1]

                    
                    self.shadow_Us[p].append(P)  
                    self.shadow_measurements[p].append(data)
                    #print("P,data: ",P,data)
                    # state_qiskit= qiskit.quantum_info.Statevector(U @ np.array(state_as_vector))
                    # # print(state_qiskit)
                    # # print(state_qiskit.sample_counts(20))
                    # data = state_qiskit.sample_counts(1)
                    # # print(list(data.keys())[0],type(list(data.keys())[0]))


                    #self.shadow_measurements[p].append(list(data.keys())[0])
            self.shadow_measurements_central=np.reshape(self.shadow_measurements,-1)
            # self.shadow_Us_central=np.reshape(self.shadow_Us,-1) 
            self.shadow_Us_central = [x1 for xs in self.shadow_Us for x1 in xs]         

        else:
            sys.exit("invalid measurement channel for Classical shadows")


    def calc_noisy_clifford_shadow_lists(self,circuit,no_estimators_K=1,N_per_estimator=200,noise_alpha=0.0):

        self.total_samples=no_estimators_K*N_per_estimator
        self.no_estimators_K = no_estimators_K
        self.N_per_estimator = N_per_estimator
        self.shadow_list=[]
        self.shadow_measurements=[]
        self.shadow_Us=[]
        self.num_qubits=circuit.num_qubits
        for n in range(no_estimators_K):
            #self.shadow_list.append(np.zeros((2**int(self.num_qubits),2**int(self.num_qubits)),dtype=complex))
            self.shadow_measurements.append([])
            self.shadow_Us.append([])                        
        # self.state_as_unitary = state_as_unitary
        # optimal_circ_save=state_as_unitary.copy()
        self.shadow_measurements_central=[]
        self.shadow_Us_central=[]


        custom_noise_model = get_noise_model(noise_alpha)

        backend = AerSimulator(
                                noise_model=custom_noise_model,
                                coupling_map=brisbane_backend.coupling_map,
                                basis_gates=eagle_basis_gates)
        # transpiled_circuit = transpile(circ, brisbane_backend)
        
        if self.channel == "Clifford":
            # print("Begin shadow collection: ", "N samples: ",self.total_samples)
            for p in range(no_estimators_K):
                for shadow_collection_counter in range(N_per_estimator):
                    
                    # elif simulator == "MPS" or "mps":

                    random_tableau = stim.Tableau.random(self.num_qubits)
                    symp_mat = [str(x).replace("_","I") for x in[random_tableau.x_output(k) for k in range(len(random_tableau))]] + [str(x).replace("_","I") for x in random_tableau.to_stabilizers()]
                    rand_clifford = qiskit.quantum_info.Clifford(symp_mat)
                    rand_cliff_circ = rand_clifford.to_circuit()
                    
                    circuit_plus_cliff = circuit.compose(rand_cliff_circ)
                    # print("types: ",type(circuit),type(rand_cliff_circ),type(circuit_plus_cliff))
                    circuit_plus_cliff.measure_all()
                    qc_compiled = transpile(circuit_plus_cliff, brisbane_backend,optimization_level=1)
                    
                    result = backend.run(qc_compiled,shots=1).result()

                    counts = result.get_counts(0)

                    data = list(counts.keys())[0]



                    #print("data: ",data)
                    self.shadow_measurements[p].append(data)
                    self.shadow_Us[p].append(rand_clifford)
                    

            self.shadow_measurements_central=np.reshape(self.shadow_measurements,-1)
            if no_estimators_K==1 and N_per_estimator==1:
                self.shadow_Us_central=self.shadow_Us[0][0]
            else:        
                # self.shadow_Us_central=np.reshape(self.shadow_Us,-1)
                self.shadow_Us_central = [x1 for xs in self.shadow_Us for x1 in xs]


        

    def add_shadow_measurements(self,state_as_vector,measurements,use_QASM_qulacs_converter=True, mapper=None):
        # print("classical_shadow.add_shadow_measurements Doesn't work since Numpy update for reshaping array of objects with array like features.")
        state_as_vector_list=list(state_as_vector)
        state_as_vector_reordered=[0.]*len(state_as_vector_list)
        for n in range(len(state_as_vector_list)):
            bins="0"*(self.num_qubits-len(str(bin(n))[2:]))+str(bin(n))[2:]
            state_as_vector_reordered[int(bins[::-1],2)]=state_as_vector_list[n]
        if self.channel == "Clifford":
            print("Begin shadow collection: ", "N samples: ",self.total_samples)
            # for p in range(no_estimators_K):
            for shadow_collection_counter in range(measurements):
                
                # elif simulator == "MPS" or "mps":

                if use_QASM_qulacs_converter:
                    # rand_clifford=qiskit.quantum_info.random_clifford(self.num_qubits)
                    random_tableau = stim.Tableau.random(self.num_qubits)
                    symp_mat = [str(x).replace("_","I") for x in[random_tableau.x_output(k) for k in range(len(random_tableau))]] + [str(x).replace("_","I") for x in random_tableau.to_stabilizers()]
                    rand_clifford = qiskit.quantum_info.Clifford(symp_mat)
                    rand_cliff_circ = rand_clifford.to_circuit()
                    # qasm = rand_cliff_circ.qasm() 
                    qasm = qasm2.dumps(rand_cliff_circ)
                    qasm = textwrap.dedent(qasm).strip()
                    circuit = convert_QASM_to_qulacs_circuit(qasm.splitlines())
                    
                    state = qulacs.QuantumState(self.num_qubits)
                    state.load(state=state_as_vector_list)
                    #gate_initial.update_quantum_state(state)
                    circuit.update_quantum_state(state)
                    data = state.sampling(1)
                    data=str([format(value, "b").zfill(self.num_qubits) for value in data][0])
                    

                else:
                    # rand_clifford=qiskit.quantum_info.random_clifford(self.num_qubits)
                    random_tableau = stim.Tableau.random(self.num_qubits)
                    symp_mat = [str(x).replace("_","I") for x in[random_tableau.x_output(k) for k in range(len(random_tableau))]] + [str(x).replace("_","I") for x in random_tableau.to_stabilizers()]
                    rand_clifford = qiskit.quantum_info.Clifford(symp_mat)
                    gate_cliff=qulacs.gate.DenseMatrix([int(list_el) for list_el in list(np.arange(self.num_qubits-1,-0.1,-1))],rand_clifford.to_matrix())
                    state = qulacs.QuantumState(self.num_qubits)
                    state.load(state=state_as_vector_reordered)
                    #gate_initial.update_quantum_state(state)
                    gate_cliff.update_quantum_state(state)
                    data = state.sampling(1)
                    data=str([format(value, "b").zfill(self.num_qubits) for value in data][0])[::-1]



                #print("data: ",data)
                self.shadow_measurements_central.append(data)
                self.shadow_Us_central.append(rand_clifford)
                
               
            self.shadow_measurements=np.reshape(self.shadow_measurements_central[:self.no_estimators_K*len(self.shadow_measurements_central)//self.no_estimators_K],    [self.no_estimators_K,len(self.shadow_measurements_central)//self.no_estimators_K])
            self.shadow_Us=to_matrix(self.shadow_Us_central,len(self.shadow_Us_central)//self.no_estimators_K,self.no_estimators_K)#np.reshape(self.shadow_Us_central[:self.no_estimators_K*len(self.shadow_Us_central)//self.no_estimators_K],    [self.no_estimators_K,len(self.shadow_Us_central)//self.no_estimators_K])

        elif self.channel == "Pauli":
            print("Begin shadow collection: ", "N samples: ",self.total_samples)
            #for shadow_collection_counter in range(no_estimators_K*N_per_estimator):
            
            for shadow_collection_counter in range(measurements):
                pauli_meas=["X","Y","Z"]
                U_string1=""
                U_string2=""
                for qubit_no in range(self.num_qubits):
                    meas_basis_int=random.randint(0,2)
                    if meas_basis_int == 0:
                        U_string1 += "H"
                        U_string2 += "I"

                    elif meas_basis_int == 1:
                        U_string1 += "H"
                        U_string2 += "S"    
                    elif meas_basis_int == 2:
                        U_string1 += "I"
                        U_string2 += "I"
                    else:
                        sys.exit()       


                
                rand_clifford1=qiskit.quantum_info.Clifford.from_label(U_string1)
                rand_clifford2=qiskit.quantum_info.Clifford.from_label(U_string2)
                rand_clifford= rand_clifford2.adjoint().compose(rand_clifford1)#np.matmul(rand_clifford1.to_matrix(),rand_clifford2.to_matrix().conjugate().transpose())

                #rand_clifford=rand_clifford2.to_circuit().inverse().compose(rand_clifford1.to_circuit())

                gate_cliff=qulacs.gate.DenseMatrix([int(list_el) for list_el in list(np.arange(self.num_qubits-1,-0.1,-1))],rand_clifford.to_matrix())
                #gate_initial=qulacs.gate.DenseMatrix([int(list_el) for list_el in list(np.arange(self.num_qubits-1,-0.1,-1))],qiskit.quantum_info.Operator(state_as_unitary).data)

                state = qulacs.QuantumState(self.num_qubits)
                #state.set_zero_state()
                #gate_initial.update_quantum_state(state)
                state.load(state=state_as_vector_reordered)
                gate_cliff.update_quantum_state(state)
                data = state.sampling(1)
                data=str([format(value, "b").zfill(self.num_qubits) for value in data][0])[::-1]

                self.shadow_measurements_central.append(data)
                self.shadow_Us_central.append([U_string1,U_string2])

            self.shadow_measurements=np.reshape(self.shadow_measurements_central[:self.no_estimators_K*len(self.shadow_measurements_central)//self.no_estimators_K],    [self.no_estimators_K,len(self.shadow_measurements_central)//self.no_estimators_K])
            # self.shadow_Us=np.reshape(self.shadow_Us_central[:self.no_estimators_K*len(self.shadow_Us_central)//self.no_estimators_K],    [self.no_estimators_K,len(self.shadow_Us_central)//self.no_estimators_K])
            self.shadow_Us=to_matrix(self.shadow_Us_central,len(self.shadow_Us_central)//self.no_estimators_K,self.no_estimators_K)
        elif self.channel == "Alternating":
            ### https://arxiv.org/pdf/2307.12912 ###
            print("Begin shadow collection: ", "N samples: ",self.total_samples)
            #for shadow_collection_counter in range(no_estimators_K*N_per_estimator):
            for shadow_collection_counter in range(measurements):
                P = np.eye(2*self.num_qubits)
                np.random.shuffle(P) 
                while np.linalg.det(P) == -1: #want det == 1
                    np.random.shuffle(P)
                    
                rotations=decompose_to_givens_rotation(P)
                #print("rotations: ",rotations)    
                f_op_tot=[]#FermionicOp({"+_0 -_0": 0})
                for  i, j, theta in rotations[:]: #create mapping for Majorana operators using the definition in terms of C/A operators and the qiskit mapper
                    if i%2==0 and j%2 == 0:
                        #print(f"G({i}, {j}, {theta})")
                        my_str1=f"-_{math.floor(i/2.)} -_{math.floor(j/2.)}"
                        my_str2=f"+_{math.floor(i/2.)} -_{math.floor(j/2.)}"
                        my_str3=f"-_{math.floor(i/2.)} +_{math.floor(j/2.)}"
                        my_str4=f"+_{math.floor(i/2.)} +_{math.floor(j/2.)}"
                        f_op = FermionicOp({my_str1: +1., my_str2: +1., my_str3: +1., my_str4: +1.},int(len(P)/2))
                        f_op_tot.append([(theta/2.)*f_op,int(i/2.),int(j/2.)])
                    elif i%2==1 and j%2 == 1:
                        #print(f"G({i}, {j}, {theta})")
                        my_str1=f"-_{math.floor(i/2.)} -_{math.floor(j/2.)}"
                        my_str2=f"+_{math.floor(i/2.)} -_{math.floor(j/2.)}"
                        my_str3=f"-_{math.floor(i/2.)} +_{math.floor(j/2.)}"
                        my_str4=f"+_{math.floor(i/2.)} +_{math.floor(j/2.)}"
                        f_op = FermionicOp({my_str1: -1., my_str2: +1., my_str3: +1., my_str4: -1.},int(len(P)/2))
                        f_op_tot.append([(theta/2.)*f_op,int(i/2.),int(j/2.)])
                    elif i%2==0 and j%2 == 1:
                        #print(f"G({i}, {j}, {theta})")
                        my_str1=f"-_{math.floor(i/2.)} -_{math.floor(j/2.)}"
                        my_str2=f"+_{math.floor(i/2.)} -_{math.floor(j/2.)}"
                        my_str3=f"-_{math.floor(i/2.)} +_{math.floor(j/2.)}"
                        my_str4=f"+_{math.floor(i/2.)} +_{math.floor(j/2.)}"
                        f_op = FermionicOp({my_str1: -complex(0,1), my_str2: -complex(0,1), my_str3: +complex(0,1), my_str4: +complex(0,1)},int(len(P)/2))
                        # print(complex(0,1)*f_op)  
                        f_op_tot.append([(theta/2.)*f_op,int(i/2.),int(j/2.)])
                    elif i%2==1 and j%2 == 0:
                        #print(f"G({i}, {j}, {theta})")
                        my_str1=f"-_{math.floor(i/2.)} -_{math.floor(j/2.)}"
                        my_str2=f"+_{math.floor(i/2.)} -_{math.floor(j/2.)}"
                        my_str3=f"-_{math.floor(i/2.)} +_{math.floor(j/2.)}"
                        my_str4=f"+_{math.floor(i/2.)} +_{math.floor(j/2.)}"
                        f_op = FermionicOp({my_str1: -complex(0,1), my_str2: +complex(0,1), my_str3: -complex(0,1), my_str4: +complex(0,1)},int(len(P)/2))
                        # print(complex(0,1)*f_op)
                        f_op_tot.append([(theta/2.)*f_op,int(i/2.),int(j/2.)])
                #print("f_op_tot: ",f_op_tot)


                # U=np.eye(2**int(len(P)/2.))
                # for p_op in f_op_tot:
                #     jwm_p_op=mapper.map(p_op[0])#
                #     if len(jwm_p_op.paulis)>1:
                #         sys.exit("more than one pauli in mapped Given's rotation")
                #     V= HamiltonianGate(jwm_p_op.paulis[0], -jwm_p_op.coeffs[0].imag).to_matrix()#np.cosh(jwm_p_op.coeffs[0].imag)*np.eye(2**int(len(P)/2.)) + np.sinh(jwm_p_op.coeffs[0].imag) * Pauli(jwm_p_op.paulis[0]).to_matrix()
                #     U= U @ V


                def qiskit_pauli_str_to_qulacs_target(pauli_str):
                    pauli = list(pauli_str)
                    target_list=[]
                    pauli_index=[]
                    for n in range(len(pauli)):
                        if pauli[n] == "X":
                            target_list.append(n)
                            pauli_index.append(1)
                        elif pauli[n] == "Y":
                            target_list.append(n)
                            pauli_index.append(2)   
                        elif pauli[n] == "Z":
                            target_list.append(n)
                            pauli_index.append(3)    
                    return pauli_index,target_list
                                                    
                qlcs_qc=qulacs.QuantumCircuit(self.num_qubits)
                for p_op in f_op_tot[::-1]:
                    jwm_p_op=mapper.map(p_op[0])#
                    if len(jwm_p_op.paulis)>1:
                        sys.exit("more than one pauli in mapped Given's rotation")
                    #print(jwm_p_op.paulis[0],jwm_p_op.coeffs[0], p_op[1],p_op[2])
                    # print(jwm_p_op.paulis[0], jwm_p_op.coeffs[0].imag)
                    # V= HamiltonianGate(jwm_p_op.paulis[0], -jwm_p_op.coeffs[0].imag)#np.cosh(jwm_p_op.coeffs[0].imag)*np.eye(2**int(len(P)/2.)) + np.sinh(jwm_p_op.coeffs[0].imag) * Pauli(jwm_p_op.paulis[0]).to_matrix()
                    pauli_index,target_list = qiskit_pauli_str_to_qulacs_target(jwm_p_op.paulis[0].to_label())
                    rot_gate = PauliRotation(target_list,pauli_index,angle=2.*jwm_p_op.coeffs[0].imag)
                    qlcs_qc.add_gate(rot_gate)



                gate_cliff=  qlcs_qc #qulacs.gate.DenseMatrix([int(list_el) for list_el in list(np.arange(self.num_qubits-1,-0.1,-1))],U)

                state = qulacs.QuantumState(self.num_qubits)
                state.load(state=state_as_vector_reordered)
                gate_cliff.update_quantum_state(state)
                data = state.sampling(1)


                data=str([format(value, "b").zfill(self.num_qubits) for value in data][0])[::-1]

                
                self.shadow_Us_central.append(P)  
                self.shadow_measurements_central.append(data)

            self.shadow_measurements=np.reshape(self.shadow_measurements_central[:self.no_estimators_K*len(self.shadow_measurements_central)//self.no_estimators_K],    [self.no_estimators_K,len(self.shadow_measurements_central)//self.no_estimators_K])
            # self.shadow_Us=np.reshape(self.shadow_Us_central[:self.no_estimators_K*len(self.shadow_Us_central)//self.no_estimators_K],    [self.no_estimators_K,len(self.shadow_Us_central)//self.no_estimators_K])   
            self.shadow_Us=to_matrix(self.shadow_Us_central,len(self.shadow_Us_central)//self.no_estimators_K,self.no_estimators_K)  

        else:
            sys.exit("invalid measurement channel for Classical shadows")
        print("Shadow collection complete")  



    def evaluate_operators(self,op_list,samples_per_estimator=0,num_estimators=None):
        ### evaluate an operator expectation value using matrix multiplication with the shadow DM, the operators in op_list must be a matrix. Only implemented for cliffords and paulis channels.

        self.identity=np.identity(2**int(self.num_qubits))
        if samples_per_estimator==0:
            samples_per_estimator=self.N_per_estimator

        if num_estimators == None:
            num_estimators = self.no_estimators_K
            shadow_measurements = self.shadow_measurements
            shadow_Us = self.shadow_Us
        else:
            shadow_measurements = np.reshape(self.shadow_measurements_central[:num_estimators*samples_per_estimator],[num_estimators,samples_per_estimator])
            # shadow_Us = np.reshape(self.shadow_Us_central[:num_estimators*samples_per_estimator],[num_estimators,samples_per_estimator])
            shadow_Us=to_matrix(self.shadow_Us_central,samples_per_estimator,num_estimators)
            print("shadow_measurements.shape: ",shadow_measurements.shape)
            print("shadow_Us.shape: ",len(shadow_Us),len(shadow_Us[0]))#,len(shadow_Us[1]))
        result_list=[]
        err_list=[]
        #energy=0.
        #total_err=0.
        
        list_for_median=[]
        #print(ucc.qubit_ham[n],np.trace(ucc.qubit_ham[n].to_matrix()))
        for x in range(self.no_estimators_K): ### for each of the K estimators 
            ##list_for_mean=[]
            ##for y in self.shadow_list[x*self.N_per_estimator:(x+1)*self.N_per_estimator]:
            ##    list_for_mean.append(y)
            
            #        density_estimator=self.shadow_list[x]  ##deepcopy(list_for_mean[0]) 
            density_estimator=   np.zeros((2**int(self.num_qubits),2**int(self.num_qubits)),dtype=complex)
            if self.channel == "Clifford":
                for meas_count in range(samples_per_estimator):
                    data=self.shadow_measurements[x][meas_count]
                    measured_SV=qiskit.quantum_info.Statevector.from_label(data)
                    rand_clifford = self.shadow_Us[x][meas_count]
                    measurement_DM=qiskit.quantum_info.DensityMatrix(measured_SV.evolve(rand_clifford.adjoint().to_matrix()))####.to_circuit().inverse()))#.inverse()))
                    M_m1_measurement_DM=(((2**measured_SV.num_qubits)  +  1)*np.array(measurement_DM)) - self.identity
                    density_estimator+=(M_m1_measurement_DM)

            elif self.channel == "Pauli":
                for meas_count in range(samples_per_estimator):
                    data=self.shadow_measurements[x][meas_count]
                    #measured_SV=qiskit.quantum_info.Statevector.from_label(data)
                    rand_clifford = self.shadow_Us[x][meas_count]
                    SV_qubit_list=[]
                    DM_qubit_list=[]
                    for qubit_no in range(len(data)):
                        SV=qiskit.quantum_info.Statevector.from_label(data[qubit_no])
                        SV_qubit_list.append(SV)
                        SV=SV.evolve(qiskit.quantum_info.Clifford.from_label(rand_clifford[0][qubit_no]))
                        SV=SV.evolve(qiskit.quantum_info.Clifford.from_label(rand_clifford[1][qubit_no]))
                        DM=qiskit.quantum_info.DensityMatrix(SV)
                        DM_qubit_list.append(3.*DM - self.identity_1qb)    
                    DM_full=deepcopy(DM_qubit_list[0])
                    for qubit_no in np.arange(1,len(DM_qubit_list)):
                        DM_full=DM_full.tensor(DM_qubit_list[qubit_no])
                    density_estimator+=(np.array(DM_full))
                    #measurement_DM=qiskit.quantum_info.DensityMatrix(measured_SV.evolve(rand_clifford.adjoint().to_matrix()))####.to_circuit().inverse()))#.inverse()))
                    #M_m1_measurement_DM=(((2**measured_SV.num_qubits)  +  1)*np.array(measurement_DM)) - self.identity
                    #density_estimator+=(M_m1_measurement_DM)        
            ##for k in list_for_mean[1:]:
            ##    density_estimator+=k
            #print(density_estimator)    
            density_estimator=density_estimator*(1/float(samples_per_estimator)) ### this is the classical estimate of the non-physical DM constructed using N_per_estimator samples

            list_for_median.append(density_estimator) ### using the classical DM construct estimate of tr(O_i \rho) 
        for n in range(len(op_list)):
            n_median=[np.trace(np.matmul(op_list[n],DM_i)) for DM_i in list_for_median]
            result_list.append(np.median(n_median))
            n_median.sort()
            if (self.no_estimators_K % 2) == 0:
                term_err=abs(n_median[int(np.ceil(self.no_estimators_K/2.)+1)] - n_median[int(np.floor(self.no_estimators_K/2.))])
            else:
                #print(np.ceil(no_estimators_K/2.))
                term_err=abs(n_median[int(np.ceil(self.no_estimators_K/2.))] - n_median[int(np.ceil(self.no_estimators_K/2.)-2)])/2.
            #total_err=np.sqrt(total_err**2 + term_err**2)
            err_list.append(term_err)
            #print(list_for_median,"   ",np.median(list_for_median))
        return result_list, err_list
    
    def evaluate_operators_pauli(self,op_list,samples_per_estimator=0,num_estimators=None): #or anything else with a to_matrix function
        ### evaluates operators with a to_matrix function exactly with matrix multiplication from the shadow DM
        if samples_per_estimator==0:
            samples_per_estimator=self.N_per_estimator

        if num_estimators == None:
            num_estimators = self.no_estimators_K
            shadow_measurements = self.shadow_measurements
            shadow_Us = self.shadow_Us
        else:
            shadow_measurements = np.reshape(self.shadow_measurements_central[:num_estimators*samples_per_estimator],[num_estimators,samples_per_estimator])
            # shadow_Us = np.reshape(self.shadow_Us_central[:num_estimators*samples_per_estimator],[num_estimators,samples_per_estimator])
            shadow_Us=to_matrix(self.shadow_Us_central,samples_per_estimator,num_estimators)
            print("shadow_measurements.shape: ",shadow_measurements.shape)
            print("shadow_Us.shape: ",len(shadow_Us),len(shadow_Us[0]))#,len(shadow_Us[1]))

        result_list=[]
        err_list=[]
        #energy=0.
        #total_err=0.
        
        list_for_median=[]
        #print(ucc.qubit_ham[n],np.trace(ucc.qubit_ham[n].to_matrix()))
        for x in range(self.no_estimators_K): ### for each of the K estimators 
            ##list_for_mean=[]
            ##for y in self.shadow_list[x*self.N_per_estimator:(x+1)*self.N_per_estimator]:
            ##    list_for_mean.append(y)
            
            #        density_estimator=self.shadow_list[x]  ##deepcopy(list_for_mean[0]) 
            density_estimator=   np.zeros((2**int(self.num_qubits),2**int(self.num_qubits)),dtype=complex)
            if self.channel == "Clifford":
                self.identity=np.identity(2**int(self.num_qubits))
                for meas_count in range(samples_per_estimator):
                    data=self.shadow_measurements[x][meas_count]
                    measured_SV=qiskit.quantum_info.Statevector.from_label(data)
                    rand_clifford = self.shadow_Us[x][meas_count]
                    measurement_DM=qiskit.quantum_info.DensityMatrix(measured_SV.evolve(rand_clifford.adjoint().to_matrix()))####.to_circuit().inverse()))#.inverse()))
                    M_m1_measurement_DM=(((2**measured_SV.num_qubits)  +  1)*np.array(measurement_DM)) - self.identity
                    density_estimator+=(M_m1_measurement_DM)

            elif self.channel == "Pauli":
                for meas_count in range(samples_per_estimator):
                    data=self.shadow_measurements[x][meas_count]
                    #measured_SV=qiskit.quantum_info.Statevector.from_label(data)
                    rand_clifford = self.shadow_Us[x][meas_count]
                    SV_qubit_list=[]
                    DM_qubit_list=[]
                    for qubit_no in range(len(data)):
                        SV=qiskit.quantum_info.Statevector.from_label(data[qubit_no])
                        SV_qubit_list.append(SV)
                        SV=SV.evolve(qiskit.quantum_info.Clifford.from_label(rand_clifford[0][qubit_no]))
                        SV=SV.evolve(qiskit.quantum_info.Clifford.from_label(rand_clifford[1][qubit_no]))
                        DM=qiskit.quantum_info.DensityMatrix(SV)
                        DM_qubit_list.append(3.*DM - self.identity_1qb)    
                    DM_full=deepcopy(DM_qubit_list[0])
                    for qubit_no in np.arange(1,len(DM_qubit_list)):
                        DM_full=DM_full.tensor(DM_qubit_list[qubit_no])
                    density_estimator+=(np.array(DM_full))
                    #measurement_DM=qiskit.quantum_info.DensityMatrix(measured_SV.evolve(rand_clifford.adjoint().to_matrix()))####.to_circuit().inverse()))#.inverse()))
                    #M_m1_measurement_DM=(((2**measured_SV.num_qubits)  +  1)*np.array(measurement_DM)) - self.identity
                    #density_estimator+=(M_m1_measurement_DM)        
            ##for k in list_for_mean[1:]:
            ##    density_estimator+=k
            #print(density_estimator)    
            density_estimator=density_estimator*(1/float(samples_per_estimator)) ### this is the classical estimate of the non-physical DM constructed using N_per_estimator samples

            list_for_median.append(density_estimator) ### using the classical DM construct estimate of tr(O_i \rho) 
        for n in range(len(op_list)):
            n_median=[np.trace(np.matmul(op_list[n].to_matrix(),DM_i)) for DM_i in list_for_median]
            result_list.append(np.median(n_median))
            n_median.sort()
            if (self.no_estimators_K % 2) == 0:
                term_err=abs(n_median[int(np.ceil(self.no_estimators_K/2.)+1)] - n_median[int(np.floor(self.no_estimators_K/2.))])
            else:
                #print(np.ceil(no_estimators_K/2.))
                term_err=abs(n_median[int(np.ceil(self.no_estimators_K/2.))] - n_median[int(np.ceil(self.no_estimators_K/2.)-2)])/2.
            #total_err=np.sqrt(total_err**2 + term_err**2)
            err_list.append(term_err)
            #print(list_for_median,"   ",np.median(list_for_median))
        return result_list, err_list
    
    



    
    def evaluate_RDM(self,op_list,samples_per_estimator=0,num_estimators=None):
        ### evaluate rdms with alternating shadows. See paper below. Operators must be passed as a majorana label e.g.
        #### turn 1/2rdm labelled by creation-annihilation operator labels p,q / p,q,r,s  into majorana labels
        # def gen_2rdm_maj_list(p,q,r,s):
        #     maj_list=[]
        #     for a in range(2*p,2*p+2):
        #         for b in range(2*q,2*q+2):
        #             for c in range(2*r,2*r+2):
        #                 for d in range(2*s,2*s+2):
        #                     maj_list.append([a,b,c,d])
        #     return maj_list   
        # def gen_1rdm_maj_list(p,q):
        #     maj_list=[]
        #     for a in range(2*p,2*p+2):
        #         for b in range(2*q,2*q+2):
        #             maj_list.append([a,b])
        #     return maj_list 
        ### https://arxiv.org/pdf/2010.16094.pdf ###
        if samples_per_estimator==0:
            samples_per_estimator=self.N_per_estimator

        if num_estimators == None:
            num_estimators = self.no_estimators_K
            shadow_measurements = self.shadow_measurements
            shadow_Us = self.shadow_Us
        else:
            shadow_measurements = np.reshape(self.shadow_measurements_central[:num_estimators*samples_per_estimator],[num_estimators,samples_per_estimator])
            # shadow_Us = np.reshape(self.shadow_Us_central[:num_estimators*samples_per_estimator],[num_estimators,samples_per_estimator])
            shadow_Us=to_matrix(self.shadow_Us_central,samples_per_estimator,num_estimators)
            print("shadow_measurements.shape: ",shadow_measurements.shape)
            print("shadow_Us.shape: ",len(shadow_Us),len(shadow_Us[0]))#,len(shadow_Us[1]))
        result_list=[]
        err_list=[]
        if self.channel == "Alternating":
            for n in range(len(op_list)):
                list_for_median=[]
                if len(op_list[n])==4 and not allUnique(op_list[n]):
                    col1=op_list[n][0]
                    col2=op_list[n][1]
                    col3=op_list[n][2]
                    col4=op_list[n][3]
                    if allEqual([col1,col2,col3,col4]): 
                        estimate = 1.
                    elif col1 == col2:
                        estimate = self.evaluate_RDM([[col3,col4]],samples_per_estimator)[0][0]
                    elif col2 == col3:
                        estimate = self.evaluate_RDM([[col1,col4]],samples_per_estimator)[0][0]        
                    elif col3 == col4:
                        estimate = self.evaluate_RDM([[col1,col2]],samples_per_estimator)[0][0]            
                    elif col1 == col3:
                        estimate = - self.evaluate_RDM([[col2,col4]],samples_per_estimator)[0][0]
                    elif col2 == col4:
                        estimate = - self.evaluate_RDM([[col1,col3]],samples_per_estimator)[0][0]
                    elif col1 == col4:           
                        estimate = self.evaluate_RDM([[col2,col3]],samples_per_estimator)[0][0]
                    else:
                        sys.exit("unhandled case for column matching check Classical_shadows.py evaluate_RDM()")
                    list_for_median.append(estimate)    
                else:    
                    for x in range(self.no_estimators_K): ### for each of the K estimators 
                        estimate=0
                        zeros=0
                        pluses=0
                        minuses=0
                        det_plus=0
                        det_minus=0
                        for meas_count in range(samples_per_estimator):
                            
                            if len(op_list[n])==2:
                                col1=op_list[n][0]
                                col2=op_list[n][1]
                                if col1 == col2:
                                    estimate = samples_per_estimator
                                else:    
                                    P_k=self.shadow_Us[x][meas_count]
                                    # P_k_t=P_k.transpose()
                                    b_k=self.shadow_measurements[x][meas_count]
                                    # print(b_k)
                                    col_vec1=np.zeros(2*self.num_qubits)
                                    col_vec1[col1]=1.
                                    col_vec2=np.zeros(2*self.num_qubits)
                                    col_vec2[col2]=1.
                                    row1=list(P_k @ col_vec1).index(1.) #which rows are non-zero in the permutatino matrix given the columns col1 & 2 
                                    row2=list(P_k @ col_vec2).index(1.)
                                    #print("P_k: ")
                                    #print(P_k)
                                    # print(b_k)
                                    # print(col1,row1)
                                    # print(col2,row2)
                                    ixgrid=np.ix_([np.min([row1,row2]),np.max([row1,row2])],[col1,col2]) #will give non-zero submatrix for rows and cols
                                    
                                    if np.max([row1,row2]) == np.min([row1,row2])+1 and np.min([row1,row2])%2==0: #if C/A operators act on different qubits then 1rdm on a basis state is zero
                                        if b_k[-1-int(np.min([row1,row2])/2)]=="0": #-1 - because of little endian
                                            meas_ex_val=complex(0,1)
                                            pluses+=1
                                        else:
                                            meas_ex_val=complex(0,-1)
                                            minuses+=1
                                                
                                    elif row1==row2: #should never happen if input operators are different
                                        if row1%2 == 0 and row2%2 == 0:
                                            meas_ex_val=1
                                        elif row1%2 == 1 and row2%2 == 1:   
                                            meas_ex_val=1

                                    else:
                                        zeros+=1
                                        meas_ex_val=0  

                                    if np.linalg.det(P_k[ixgrid])==1:
                                        det_plus+=1
                                    elif np.linalg.det(P_k[ixgrid])==-1:
                                        det_minus+=1
                                    else:
                                        sys.exit("determinant is wrong: "+str(np.linalg.det(P_k[ixgrid])))
                                    eigenval=math.comb(2*self.num_qubits,len(op_list[n]))/math.comb(self.num_qubits,int(len(op_list[n])/2.))
                                    estimate+=eigenval*meas_ex_val*np.linalg.det(P_k[ixgrid])
                            elif len(op_list[n]) == 4:
                                col1=op_list[n][0]
                                col2=op_list[n][1]
                                col3=op_list[n][2]
                                col4=op_list[n][3]
                                
                                P_k=self.shadow_Us[x][meas_count]
                                # P_k_t=P_k.transpose()
                                b_k=self.shadow_measurements[x][meas_count]
                                # print(b_k)
                                col_vec1=np.zeros(2*self.num_qubits)
                                col_vec1[col1]=1.
                                col_vec2=np.zeros(2*self.num_qubits)
                                col_vec2[col2]=1.

                                col_vec3=np.zeros(2*self.num_qubits)
                                col_vec3[col3]=1.
                                col_vec4=np.zeros(2*self.num_qubits)
                                col_vec4[col4]=1.

                                row1=list(P_k @ col_vec1).index(1.) #which rows are non-zero in the permutatino matrix given the columns col1 & 2 
                                row2=list(P_k @ col_vec2).index(1.)
                                row3=list(P_k @ col_vec3).index(1.)
                                row4=list(P_k @ col_vec4).index(1.)
                                if allUnique([row1,row2,row3,row4]):
                                    sorted_rows=np.sort([row1,row2,row3,row4])
                                    ixgrid=np.ix_(sorted_rows,[col1,col2,col3,col4]) #will give non-zero submatrix for rows and cols
                                    row_sort1=sorted_rows[0]
                                    row_sort2=sorted_rows[1]
                                    row_sort3=sorted_rows[2]
                                    row_sort4=sorted_rows[3]
                                    if row_sort2 == row_sort1+1 and row_sort4 == row_sort3+1 and row_sort1%2==0 and row_sort3%2==0: #if C/A operators act on different qubits then rdm on a basis state is zero
                                        if b_k[-1-int(row_sort1/2)]=="0" and b_k[-1-int(row_sort3/2)]=="0": #-1 - because of little endian
                                            meas_ex_val=-1
                                        elif b_k[-1-int(row_sort1/2)]=="1" and b_k[-1-int(row_sort3/2)]=="0":
                                            meas_ex_val=1
                                        elif b_k[-1-int(row_sort1/2)]=="0" and b_k[-1-int(row_sort3/2)]=="1":
                                            meas_ex_val=1
                                        elif b_k[-1-int(row_sort1/2)]=="1" and b_k[-1-int(row_sort3/2)]=="1":
                                            meas_ex_val=-1        
                                        else:
                                            sys.exit("error in vaues of b_k, not 0 or 1")

                                    else:
                                        zeros+=1
                                        meas_ex_val=0  

                                    if np.linalg.det(P_k[ixgrid])==1:
                                        det_plus+=1
                                    elif np.linalg.det(P_k[ixgrid])==-1:
                                        det_minus+=1
                                    else:
                                        sys.exit("determinant is wrong: "+str(np.linalg.det(P_k[ixgrid])))
                                    eigenval=math.comb(2*self.num_qubits,len(op_list[n]))/math.comb(self.num_qubits,int(len(op_list[n])/2.))
                                    estimate+=eigenval*meas_ex_val*np.linalg.det(P_k[ixgrid])      
                                else:
                                    sys.exit("Not all rows in permuted matrix are unique")    
                                

                            else:
                                sys.exit("RDMs of size "+str(len(op_list[n]))+" are not implemented")    
                        list_for_median.append(estimate/samples_per_estimator) 
                        # print("zeros:",zeros,det_minus,det_plus) 
                #print(list_for_median,np.median(list_for_median))
                result_list.append(np.median(list_for_median)) 
        else:    
            sys.exit("not implemented, use other functions for clifford or pauli shadows")              

        return result_list, err_list        


    

    def evaluate_operator_overlap_stabilizer_perp(self,op_list,samples_per_estimator=0,num_estimators = None):
        '''
        can provide multiple operators in the form [[coefficient_1, index_a_1,index_b_1],[coefficient_2, index_a_2,index_b_2],...]
        for each pair of cbs labelled by index_a/b compute the overlap with each stabilizer state made by U_i^{\dagger}|b_i> then do a median of means
        '''
        
        ### https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-021-04351-z/MediaObjects/41586_2021_4351_MOESM1_ESM.pdf ###
        if samples_per_estimator==0:
            samples_per_estimator=self.N_per_estimator

        if num_estimators == None:
            num_estimators = self.no_estimators_K
            shadow_measurements = self.shadow_measurements
            shadow_Us = self.shadow_Us
        else:
            shadow_measurements = np.reshape(self.shadow_measurements_central[:num_estimators*samples_per_estimator],[num_estimators,samples_per_estimator])
            # shadow_Us = np.reshape(self.shadow_Us_central[:num_estimators*samples_per_estimator],[num_estimators,samples_per_estimator])
            shadow_Us=to_matrix(self.shadow_Us_central,samples_per_estimator,num_estimators)
            print("shadow_measurements.shape: ",shadow_measurements.shape)
            print("shadow_Us.shape: ",len(shadow_Us),len(shadow_Us[0]))#,len(shadow_Us[1]))
        # if samples_per_estimator > self.N_per_estimator:
        #     sys.exit("asked for more samples than exist in shadow: "+str(samples_per_estimator)+">"+str(self.N_per_estimator))
        result_list=[]
        err_list=[]
        #energy=0.
        #total_err=0.
        
        list_for_median=[]
        for n in range(len(op_list)):
            list_for_median.append([0]*num_estimators)
        #print(ucc.qubit_ham[n],np.trace(ucc.qubit_ham[n].to_matrix()))
        for x in range(num_estimators): ### for each of the K estimators 
            ##list_for_mean=[]
            ##for y in self.shadow_list[x*self.N_per_estimator:(x+1)*self.N_per_estimator]:
            ##    list_for_mean.append(y)
            
            #        density_estimator=self.shadow_list[x]  ##deepcopy(list_for_mean[0]) 
            #density_estimator=   np.zeros((2**int(self.num_qubits),2**int(self.num_qubits)),dtype=complex)
            if self.channel == "Clifford":
                
                for meas_count in range(samples_per_estimator):
                    data=shadow_measurements[x][meas_count]
                    rand_clifford = shadow_Us[x][meas_count]
                    data_dummy=data.replace("1","X")
                    data_pauli=data_dummy.replace("0","I")
                    #print(data_pauli)

                    stab_data_pauli=qiskit.quantum_info.Pauli(data_pauli)
                    stab_data=qiskit.quantum_info.StabilizerState(stab_data_pauli)
                    stab_data=stab_data.evolve(rand_clifford.adjoint())
                    stab_matrix=[]
                    #phase_vec=[]
                    for n in stab_data.clifford.to_labels(mode="S"):
                        #phase_vec.append(n[0])
                        row=[]
                        for m in n[:]:
                            row.append(m)
                        stab_matrix.append(row[1:]+[1 if row[0]=="+" else -1 ])
                    reduced_stab_matrix=stabilizer.reduce_to_row_echelon(stab_matrix)
                    # for row in reduced_stab_matrix:
                    #     print(row)      
                    ovlp_magnitude= 1/(np.sqrt(2)**stabilizer.x_rank_of_reduced(reduced_stab_matrix))
                    # ss=qiskit.quantum_info.StabilizerState(stab_data.clifford.symplectic_matrix)
                    meas_out=stab_data.measure()[0]
                    for n in range(len(op_list)):
                        ovlp_state1="0"*(self.num_qubits-len(str(bin(op_list[n][1]))[2:]))+str(bin(op_list[n][1]))[2:]
                        ovlp_state2="0"*(self.num_qubits-len(str(bin(op_list[n][2]))[2:]))+str(bin(op_list[n][2]))[2:]
                        #print(ovlp_state1,ovlp_state2,get_ovlp_phase_basis(meas_out,ovlp_state1,reduced_stab_matrix),np.conj(get_ovlp_phase_basis(meas_out,ovlp_state2,reduced_stab_matrix)),ovlp_magnitude)
                        if ovlp_state1 != ovlp_state2:
                            list_for_median[n][x]+= 2.*ovlp_magnitude*ovlp_magnitude*((2**self.num_qubits)  +  1) * (stabilizer.get_ovlp_phase_basis(meas_out,ovlp_state1,reduced_stab_matrix)*np.conj(stabilizer.get_ovlp_phase_basis(meas_out,ovlp_state2,reduced_stab_matrix)))
                        # elif ovlp_state1 == ovlp_state2:
                        #     list_for_median[n][x]+= (-1/op_list[n][0]) + ovlp_magnitude*ovlp_magnitude*((2**self.num_qubits)  +  1) * (get_ovlp_phase_basis(meas_out,ovlp_state1,reduced_stab_matrix)*np.conj(get_ovlp_phase_basis(meas_out,ovlp_state2,reduced_stab_matrix)) + np.conj(get_ovlp_phase_basis(meas_out,ovlp_state1,reduced_stab_matrix))*get_ovlp_phase_basis(meas_out,ovlp_state2,reduced_stab_matrix))
                        

            else: 
                sys.exit("Not implemented for this Channel: "+str(self.channel))   
            #density_estimator=density_estimator*(1/float(samples_per_estimator)) ### this is the classical estimate of the non-physical DM constructed using N_per_estimator samples
        # print("list for median: ",list_for_median)
        for n in range(len(op_list)):
            n_median=list_for_median[n]
            result_list.append(np.median(n_median)*(1/float(samples_per_estimator))*op_list[n][0])
            n_median.sort()
            if num_estimators==1:
                term_err=0.
            else:    
                if (num_estimators % 2) == 0:
                    term_err=abs(n_median[int(np.ceil(num_estimators/2.)+1)] - n_median[int(np.floor(num_estimators/2.))])
                else:
                    #print(np.ceil(no_estimators_K/2.))
                    term_err=abs(n_median[int(np.ceil(num_estimators/2.))] - n_median[int(np.ceil(num_estimators/2.)-2)])/2.
            #total_err=np.sqrt(total_err**2 + term_err**2)
            err_list.append(term_err*(1/float(samples_per_estimator))*op_list[n][0])
            #print(list_for_median,"   ",np.median(list_for_median))
        return result_list, err_list



    

    def evaluate_operator_pauli_stabilizer(self,op_list,samples_per_estimator=0,num_estimators=None):
        '''
        - efficient (in terms of classical evaluatino time) evaluation of pauli expectation valules for Clifford channel (not measurement efficient), could be adapted for efficient evaluation of Pauli channel by converting qiskit Pauli U operators to Clifford operators
        - elements of op_list are qiskit Pauli operators
        '''
        if samples_per_estimator==0:
            samples_per_estimator=self.N_per_estimator

        if num_estimators == None:
            num_estimators = self.no_estimators_K
            shadow_measurements = self.shadow_measurements
            shadow_Us = self.shadow_Us
        else:
            shadow_measurements = np.reshape(self.shadow_measurements_central[:num_estimators*samples_per_estimator],[num_estimators,samples_per_estimator])
            # shadow_Us = np.reshape(self.shadow_Us_central[:num_estimators*samples_per_estimator],[num_estimators,samples_per_estimator])
            shadow_Us=to_matrix(self.shadow_Us_central,samples_per_estimator,num_estimators)

        result_list=[]
        err_list=[]
        #energy=0.
        #total_err=0.
        
        list_for_median=[]
        for n in range(len(op_list)):
            list_for_median.append([0]*self.no_estimators_K)
        #print(ucc.qubit_ham[n],np.trace(ucc.qubit_ham[n].to_matrix()))
        for x in range(self.no_estimators_K): ### for each of the K estimators 
            ##list_for_mean=[]
            ##for y in self.shadow_list[x*self.N_per_estimator:(x+1)*self.N_per_estimator]:
            ##    list_for_mean.append(y)
            
            #        density_estimator=self.shadow_list[x]  ##deepcopy(list_for_mean[0]) 
            #density_estimator=   np.zeros((2**int(self.num_qubits),2**int(self.num_qubits)),dtype=complex)
            if self.channel == "Clifford":
                
                for meas_count in range(samples_per_estimator):
                    data=self.shadow_measurements[x][meas_count]
                    rand_clifford = self.shadow_Us[x][meas_count]
                    data_dummy=data.replace("1","X")
                    data_pauli=data_dummy.replace("0","I")
                    #print(data_pauli)

                    stab_data_pauli=qiskit.quantum_info.Pauli(data_pauli)
                    stab_data=qiskit.quantum_info.StabilizerState(stab_data_pauli)
                    stab_data=stab_data.evolve(rand_clifford.adjoint())
                    for n in range(len(op_list)):
                        if op_list[n].to_label() != "I"*self.num_qubits:
                            list_for_median[n][x]+=((2**self.num_qubits)  +  1) * stab_data.expectation_value(op_list[n])
                        else:
                            list_for_median[n][x]+=1    
                        

            else: 
                sys.exit("Not implemented for this Channel: "+str(self.channel))   
            #density_estimator=density_estimator*(1/float(samples_per_estimator)) ### this is the classical estimate of the non-physical DM constructed using N_per_estimator samples
        #print("list for median: ",list_for_median)
        for n in range(len(op_list)):
            n_median=list_for_median[n]
            result_list.append(np.median(n_median)*(1/float(samples_per_estimator)))
            n_median.sort()
            if (self.no_estimators_K % 2) == 0:
                term_err=abs(n_median[int(np.ceil(self.no_estimators_K/2.)+1)] - n_median[int(np.floor(self.no_estimators_K/2.))])
            else:
                #print(np.ceil(no_estimators_K/2.))
                term_err=abs(n_median[int(np.ceil(self.no_estimators_K/2.))] - n_median[int(np.ceil(self.no_estimators_K/2.)-2)])/2.
            #total_err=np.sqrt(total_err**2 + term_err**2)
            err_list.append(term_err)
            #print(list_for_median,"   ",np.median(list_for_median))
        return result_list, err_list



    



