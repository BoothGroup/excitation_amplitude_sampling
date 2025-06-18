### calculate energy from expectation values using IBM runtime

from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator
import ffsim
import numpy as np
from pyscf import cc
from pyscf import gto, scf, ao2mo, fci, ci
from pyscf import cc as cupclus

import scipy.optimize

import qiskit

from qiskit import transpile
from qiskit_ibm_runtime.fake_provider import FakeBrisbane
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# from qiskit_aer import AerSimulator

from qiskit_nature.second_q.operators import ElectronicIntegrals
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy

import pickle 

backend = FakeBrisbane()
# backend = AerSimulator()


nsite=4
d=1.0

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


mol_data = ffsim.MolecularData.from_scf(myhf)
norb = mol_data.norb
nelec = mol_data.nelec
nocc = nelec[0]
nvir=norb-nocc


n_reps = 1

optimal_params_1 = np.array([5.20238071e-01,-1.46440164e+00,6.32596863e-02,2.91782243e-01,-6.90402316e-02,8.06905575e-01,9.24387942e-01,-4.91323970e-01,-6.34107959e-01,-1.38679445e+00,8.44947515e-01,4.94526211e-03,9.42290059e-01,-5.37566280e-01,-5.66186135e-01,3.39029312e-01,3.24740432e-01,-1.55763138e-01,-3.05035972e-01,-6.58872439e-01,-5.08110967e-01,-5.57867419e-04,-4.23376768e-01])

pairs_aa = [(p, p + 1) for p in range(norb - 1)]
pairs_ab = [(p, p) for p in range(norb)]
interaction_pairs = (pairs_aa, pairs_ab)

operator = ffsim.UCJOpSpinBalanced.from_parameters(
        optimal_params_1, norb=norb, n_reps=n_reps, interaction_pairs=interaction_pairs
    )
qubits = qiskit.QuantumRegister(2 * norb, name="q")
circuit = qiskit.QuantumCircuit(qubits)
circuit.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec),qubits)
circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(operator),qubits)

pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
isa_psi = pm.run(circuit)
isa_observables = qubit_ham.apply_layout(isa_psi.layout)



Nshots=1000
n_for_variance=8

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

f_name="energy_exp_vals.dat"
f = open(f_name, "a")
for m in energies:
    f.write(str(m)+"\n")
f.close()  
