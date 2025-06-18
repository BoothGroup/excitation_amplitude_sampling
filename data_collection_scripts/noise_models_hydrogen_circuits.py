### Data collection for the classical simulation part of the noise interpolation plot


from qiskit_nature.second_q.operators import ElectronicIntegrals, FermionicOp ,SparseLabelOp, PolynomialTensor, tensor_ordering
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
#from qiskit_nature.second_q.circuit.library import UCCSD

########### UCC modules #######
#from qiskit_nature.second_q.circuit import QubitConverter
from qiskit_nature.second_q.circuit.library.ansatzes import UCC
from qiskit_nature.second_q.circuit.library.initial_states import HartreeFock


########### VQE Modules #######
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit.quantum_info import Statevector


########### OLD VQE Modules #######
from qiskit_algorithms import VQE
from qiskit_aer.primitives import Estimator as AerEstimator


######### Optimiser Modules #########
from qiskit_algorithms.optimizers import COBYLA , SLSQP , L_BFGS_B


from qiskit.quantum_info import SparsePauliOp

from qiskit.circuit.library import NLocal,EfficientSU2
######### PySCF + others #########

import sys
sys.argv.append("--quiet")  # For Vayesta
import numpy as np
from pyscf import gto, scf, lib, ao2mo
# from vayesta.lattmod import Hubbard1D, LatticeRHF
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
from pyscf.ci import cisd

# import julia
# julia.install()
# from julia.api import Julia
# jl = Julia(compiled_modules=False)
# from julia import Main
# from julia import ITensors

# Main.include("../Quantum_cluster/src/julia_functions.jl")


# from src import Lan_coeffs

import ffsim
import numpy as np
from pyscf import cc

from operator import itemgetter

import scipy.optimize
from qiskit_aer import AerSimulator
from qiskit import transpile
from qiskit_ibm_runtime.fake_provider import FakeBrisbane
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
import pickle 

from qiskit_nature.second_q.circuit.library.ansatzes.utils import generate_fermionic_excitations 

import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
from qiskit_aer.noise import NoiseModel
from qiskit.providers.fake_provider import Fake127QPulseV1
from qiskit.providers.models import backendproperties
from qiskit_ibm_runtime.fake_provider import FakeBrisbane
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator


from ebcc import REBCC, Space, GEBCC
from ebcc.ext.fci import _coefficients_to_amplitudes_restricted, _ci_vector_to_coefficients_restricted
from pyscf.fci import cistring

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


def civec_from_shadow(Shadow, fermi_vac):
    ###index for HF state
    index_a=0
    index_b=int(fermi_vac,2)
    op_list=[]
    op_list.append([1.,index_a,index_b])
    results_l,errors_l = Shadow.evaluate_operator_overlap_stabilizer_perp(op_list,samples_per_estimator=int(nshots//3),num_estimators = 3)
    # print(results_l)
    c0_cs=results_l[0]

    a_exc = generate_fermionic_excitations(num_excitations=1, num_spatial_orbitals=norb, num_particles=nelec, alpha_spin=True, beta_spin=True, max_spin_excitation=None, generalized=False, preserve_spin=True)
    a_exc_strings=excitations_tuples_to_strings(a_exc,fermi_vac)
    op_list=[]
    for a in a_exc_strings:
        index_a=0#"0"*SV_basis_0.num_qubits#int(list(SV_basis_0.to_dict().keys())[0],2)
        index_b=int(a,2)
        op_list.append([1.,index_a,index_b])

    results_l,errors_l = Shadow.evaluate_operator_overlap_stabilizer_perp(op_list,samples_per_estimator=int(nshots//3),num_estimators = 3)
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

    results_l,errors_l = Shadow.evaluate_operator_overlap_stabilizer_perp(op_list,samples_per_estimator=int(nshots//3),num_estimators = 3)
    ab_int_norm_results = [x for x in results_l]
    print(ab_int_norm_results)
 



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

    return my_civec_cs


# nshots = int(1e3)
# n_for_variance=16
# alpha=.02
# nsite=4
# d=1.0

nshots = int(float(sys.argv[3]))
n_for_variance=int(sys.argv[4])
alpha1=float(sys.argv[5])
alpha2=float(sys.argv[6])
alpha_grid=float(sys.argv[7])
nsite=int(sys.argv[1])
d=float(sys.argv[2])

fermi_vac= "0"*(nsite//2)+"1"*(nsite//2)+"0"*(nsite//2)+"1"*(nsite//2) #'00110011'

print("Fermi vac state: ",fermi_vac)

# Build noise model from backend properties
brisbane_backend = FakeBrisbane()
brisbane_noise_model = NoiseModel.from_backend(brisbane_backend)
# Get basis gates from noise model
eagle_basis_gates = brisbane_noise_model.basis_gates

# Get coupling map from backend
coupling_map = brisbane_backend.configuration().coupling_map


import qiskit_aer.noise as noise

# Build noise model from backend properties
brisbane_backend = FakeBrisbane()
brisbane_noise_model = NoiseModel.from_backend(brisbane_backend)
# Get basis gates from noise model
eagle_basis_gates = brisbane_noise_model.basis_gates


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





my_atom=[("H 0. 0. %f" % xyz) for xyz in [d*x for x in list(range(nsite))]]

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
assert cc.converged
print('CCSD total energy: {}'.format(cc.e_tot))
CCSD_energy = cc.e_tot





one_body = myhf.mo_coeff.T @ myhf.get_hcore() @ myhf.mo_coeff
eri = ao2mo.kernel(myhf._eri, (myhf.mo_coeff, myhf.mo_coeff,myhf.mo_coeff,myhf.mo_coeff), compact=False)
two_body = eri.reshape((myhf.mo_coeff.shape[-1],) * 4)

# Constructing the electronic hamiltonian in second quantised representation
integrals = ElectronicIntegrals.from_raw_integrals(h1_a=one_body, h1_b = one_body, h2_aa=two_body, h2_bb= two_body, h2_ba= two_body , auto_index_order=True) 

# Defining the many body electronic hamiltionian in second quantised representation

h_elec = ElectronicEnergy(integrals, constants = {'nuclear_repulsion_energy':mol.energy_nuc()}).second_q_op()       
mapper = JordanWignerMapper()
qubit_ham = mapper.map(h_elec) # this performs the JW transformation to qubit representation

my_fci = pyscf.fci.FCI(myhf)
my_fci.kernel()
print('FCI total energy: {}'.format(my_fci.e_tot))



mol_data = ffsim.MolecularData.from_scf(myhf)
norb = mol_data.norb
nelec = mol_data.nelec
nocc = nelec[0]
nvir=norb-nocc
mol_hamiltonian = mol_data.hamiltonian

# Construct UCJ operator
n_reps = 1

if nsite == 4:
    optimal_params = np.array([5.20238071e-01,-1.46440164e+00,6.32596863e-02,2.91782243e-01,-6.90402316e-02,8.06905575e-01,9.24387942e-01,-4.91323970e-01,-6.34107959e-01,-1.38679445e+00,8.44947515e-01,4.94526211e-03,9.42290059e-01,-5.37566280e-01,-5.66186135e-01,3.39029312e-01,3.24740432e-01,-1.55763138e-01,-3.05035972e-01,-6.58872439e-01,-5.08110967e-01,-5.57867419e-04,-4.23376768e-01])
elif nsite ==2:
    optimal_params = np.array([-1.96964782,-1.20855392,-0.81585484,0.42315576,0.08903292,-0.35243734,-0.35243734])
else:
    sys.exit("only handles 2 or 4 sites")
# with open('LUCJ_nn_params_n_reps_1_Hydrogen_4_atom.pkl', 'wb') as f:
#     pickle.dump(optimal_params_1, f)

pairs_aa = [(p, p + 1) for p in range(norb - 1)]
pairs_ab = [(p, p) for p in range(norb)]
interaction_pairs = (pairs_aa, pairs_ab)
operator = ffsim.UCJOpSpinBalanced.from_parameters(
        optimal_params, norb=norb, n_reps=n_reps, interaction_pairs=interaction_pairs
    )

reference_state = ffsim.hartree_fock_state(norb, nelec)

# Apply the operator to the reference state
ansatz_state = ffsim.apply_unitary(reference_state, operator, norb=norb, nelec=nelec)

# Compute the energy ⟨ψ|H|ψ⟩ of the ansatz state
hamiltonian = ffsim.linear_operator(mol_hamiltonian, norb=norb, nelec=nelec)
energy = np.real(np.vdot(ansatz_state, hamiltonian @ ansatz_state))
print(f"Energy: {energy}")



print(nelec)
# qubits = 2*norb
qubits = qiskit.QuantumRegister(2 * norb, name="q")
circuit = qiskit.QuantumCircuit(qubits)
circuit.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec),qubits)
circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(operator),qubits)
circuit.measure_all()
# get backend
# provider = IBMProvider()
backend = FakeBrisbane()

qc_compiled = transpile(circuit, backend,optimization_level=3)

print("depth:",qc_compiled.depth(),"\ntotal gates: ",qc_compiled.size(),"\ngate breakdown: ",qc_compiled.count_ops())



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

# backend = FakeBrisbane()
# backend = AerSimulator()


  



for alpha in np.arange(alpha1,alpha2, alpha_grid):

    custom_noise_model = get_noise_model(alpha)

    backend = AerSimulator(
                            noise_model=custom_noise_model,
                            coupling_map=brisbane_backend.coupling_map,
                            basis_gates=eagle_basis_gates)
    # transpiled_circuit = transpile(circ, brisbane_backend)


    operator = ffsim.UCJOpSpinBalanced.from_parameters(
            optimal_params, norb=norb, n_reps=n_reps, interaction_pairs=interaction_pairs
        )

    # with open('cliff_list_actual.pkl', 'rb') as f:
    #     clifford_list_2 = pickle.load(f)
    # clifford_list_2


    
    qubits = qiskit.QuantumRegister(2 * norb, name="q")
    circuit = qiskit.QuantumCircuit(qubits)
    circuit.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec),qubits)
    circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(operator),qubits)

    pm = generate_preset_pass_manager(backend=brisbane_backend, optimization_level=3)
    isa_psi = pm.run(circuit)
    isa_observables = qubit_ham.apply_layout(isa_psi.layout)



    Nshots=nshots


    # pauli_groups=qubit_ham.group_commuting()
    estimator_options = dict(default_shots=Nshots)
    estimator = Estimator(mode=backend,options=estimator_options)

    energies=[]
    for n in range(n_for_variance):
        job = estimator.run([(isa_psi, isa_observables)])
        pub_result = job.result()[0]
        # print(f"Expectation values: {pub_result.data.evs}")
        print(f"Expectation values: {pub_result.data.evs + mol.energy_nuc()}")
        energies.append(pub_result.data.evs + mol.energy_nuc())
    print(np.mean(energies),np.std(energies))  

    proj_energies=[]
    ccsd_rdm_energies=[]
    ccsd_rdm_energies_no_lambda = []
    for n in range(n_for_variance):
        clifford_list_2=[]
        for n in range(nshots):
            rand_clifford=qiskit.quantum_info.random_clifford(2*norb)
            clifford_list_2.append(rand_clifford.to_dict())
        meaurement_result = []
        for cliff_count in range(nshots):
            rand_cliff = clifford_list_2[cliff_count]
            circuit = qiskit.QuantumCircuit(qubits)
            # circuit.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec),qubits)
            sp_circ,a1 = cbssp_unitary(fermi_vac,"0"*(2*norb))
            circuit.append(sp_circ,qubits)
            circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(operator),qubits)
            circuit.append(qiskit.quantum_info.Clifford.from_dict(rand_cliff).to_instruction(),qubits)
            circuit.measure_all()
            # qiskit.qasm2.dump(circuit, "qasm_circuits/classical_shadow_circuit_"+str(cliff_count)+".qasm")
            # circuit2=qiskit.QuantumCircuit.from_qasm_file("qasm_circuits/classical_shadow_circuit_"+str(cliff_count)+".qasm")
            qc_compiled = transpile(circuit, brisbane_backend)

            # # Execute the circuit on the qasm simulator.
            # # We've set the number of repeats of the circuit
            # # to be 1024, which is the default.
            # job_sim = backend.run(qc_compiled, shots=1)

            # # Grab the results from the job.
            # result_sim = job_sim.result()
            # meaurement_result.append(list(result_sim.get_counts().keys())[0])
            # # print(result_sim.get_counts())
            result = backend.run(qc_compiled,shots=1).result()

            counts = result.get_counts(0)
            meaurement_result.append(list(counts.keys())[0])

            # plot_histogram(dict(sorted(counts.items(), key=itemgetter(1), reverse=True)[:15]))



        # print(meaurement_result)
        cliffords_for_Us = [qiskit.quantum_info.Clifford.from_dict(rand_cliff) for rand_cliff in clifford_list_2]
        shadow = classical_shadow.Classical_shadow("Clifford",2*norb,meaurement_result,cliffords_for_Us)
        # shadow.N_per_estimator = int(Nshots/3)
        t1addr, t1sign = cisd.tn_addrs_signs(norb, nocc, 1)

        op_list=[]
        index_a=0#"0"*SV_basis_0.num_qubits#int(list(SV_basis_0.to_dict().keys())[0],2)
        ###index for HF state
        index_b=int(fermi_vac,2)
        op_list.append([1.,index_b,index_a])
        result,error = shadow.evaluate_operator_overlap_stabilizer_perp(op_list,samples_per_estimator=int(nshots//3),num_estimators = 3)
        c0_cs=result[0]


        ab_exc = generate_fermionic_excitations(num_excitations=2,max_spin_excitation=1, num_spatial_orbitals=norb, num_particles=nelec, alpha_spin=True, beta_spin=True, generalized=False, preserve_spin=True)
        ab_exc_strings = excitations_tuples_to_strings(ab_exc,fermi_vac)
        ### construct the labels for operators of the form |index_a><index_b| + |index_b><index_a| (for the real part) + imaginary part
        ab_int_norm_results=[]
        # print("ab_exc_strings: ",ab_exc_strings)
        op_list=[]
        for ab in ab_exc_strings:
            index_a=0#"0"*SV_basis_0.num_qubits#int(list(SV_basis_0.to_dict().keys())[0],2)
            index_b=int(ab,2)
            # print(index_b)
            op_list.append([1.,index_b,index_a])
        result,error = shadow.evaluate_operator_overlap_stabilizer_perp(op_list,samples_per_estimator=int(nshots//3),num_estimators = 3)
        c2_cs_ab = np.einsum("i,j,ij->ij", t1sign, t1sign, np.reshape(result,(len(t1addr),len(t1addr))))
        c2_cs_ab = np.reshape(c2_cs_ab,(nocc, nvir, nocc, nvir)).transpose(0, 2, 1, 3)

        c2_cs_rs = c2_cs_ab/c0_cs
        g_ovvo = two_body[:nocc,nocc:,nocc:,:nocc]
        e_singles=0.
        e_doubles = 2 * np.einsum("ijab,iabj->", c2_cs_rs, g_ovvo) - np.einsum("ijab,ibaj->", c2_cs_rs, g_ovvo)
        e_proj = myhf.e_tot + e_singles + e_doubles
        print(e_proj)
        proj_energies.append(e_proj)
      



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

        

        # Notes: 
        # - There are public fci_to_amplitudes_{spin} functions, but I use the private functions for
        #   finer control.
        # - If you're using custom C amplitudes there may be conventional differences. This can be worked
        #   out if needed.
        # - The `coefficients` object can be initialised from NumPy arrays as
        #   `coefficients = ebcc.util.Namespace(c1=c1, c2=c2, c3=c3)`, scaled by `c0` I guess?

        # Extract the C amplitudes from the FCI calculation
        civec = civec_from_shadow(shadow,fermi_vac)
        coefficients = _ci_vector_to_coefficients_restricted(civec, space, max_order=2)

        # Get the T amplitudes
        amplitudes = _coefficients_to_amplitudes_restricted(coefficients, max_order=2)

        # Initialise a ebcc_CCSD calculation and get the 1RDM
        ci_ebcc_ccsd = REBCC(myhf, ansatz="CCSD", space=space)
        ci_ebcc_ccsd.amplitudes = amplitudes
        ci_ebcc_ccsd.solve_lambda()
        ci_rdm1_sd = ci_ebcc_ccsd.make_rdm1_f()
        ci_rdm2_sd = ci_ebcc_ccsd.make_rdm2_f()

        two_e_energy = np.einsum('ijkl,ijkl->', two_body, ci_rdm2_sd) * 0.5
        one_e_energy = np.einsum('ij,ij->', one_body, ci_rdm1_sd)
        ci_sd_rdm_energy = two_e_energy + one_e_energy + mol.energy_nuc()

        ccsd_rdm_energies.append(ci_sd_rdm_energy)


        ci_ebcc_ccsd_no_lambda = REBCC(myhf, ansatz="CCSD", space=space)
        ci_ebcc_ccsd_no_lambda.amplitudes = amplitudes
        
        ci_rdm1_sd_no_lambda = ci_ebcc_ccsd_no_lambda.make_rdm1_f()
        ci_rdm2_sd_no_lambda = ci_ebcc_ccsd_no_lambda.make_rdm2_f()

        two_e_energy = np.einsum('ijkl,ijkl->', two_body, ci_rdm2_sd_no_lambda) * 0.5
        one_e_energy = np.einsum('ij,ij->', one_body, ci_rdm1_sd_no_lambda)
        ci_sd_rdm_energy_no_lambda = two_e_energy + one_e_energy + mol.energy_nuc()

        ccsd_rdm_energies_no_lambda.append(ci_sd_rdm_energy_no_lambda)

    # print(np.mean(proj_energies),np.std(proj_energies))          
    



    print('FCI total energy: {}'.format(my_fci.e_tot))
    print("Expectation value: ",np.mean(energies),np.std(energies))    
    print("Mixed Estimator: ",np.mean(proj_energies),np.std(proj_energies))  
    print('Energy from RDMs: ', np.mean(ccsd_rdm_energies),np.std(ccsd_rdm_energies))
    with open('energies_fr_exp_val_for_noise_plot_'+str(nshots)+"_alpha_"+str(alpha)+'_Hydrogen_chain_'+str(nsite)+'_atom_'+str(d)+'_distance_new2.pkl', 'wb') as f:
        pickle.dump(energies, f)

    with open('energies_fr_mixed_for_noise_plot_'+str(nshots)+"_alpha_"+str(alpha)+'_Hydrogen_chain_'+str(nsite)+'_atom_'+str(d)+'_distance_new2.pkl', 'wb') as f:
        pickle.dump(proj_energies, f)    

    with open('energies_fr_csd_rdm_for_noise_plot_'+str(nshots)+"_alpha_"+str(alpha)+'_Hydrogen_chain_'+str(nsite)+'_atom_'+str(d)+'_distance_new2.pkl', 'wb') as f:
        pickle.dump(ccsd_rdm_energies, f)   
    
    with open('energies_fr_csd_rdm_no_lam_for_noise_plot_'+str(nshots)+"_alpha_"+str(alpha)+'_Hydrogen_chain_'+str(nsite)+'_atom_'+str(d)+'_distance_new2.pkl', 'wb') as f:
        pickle.dump(ccsd_rdm_energies_no_lambda, f)            
