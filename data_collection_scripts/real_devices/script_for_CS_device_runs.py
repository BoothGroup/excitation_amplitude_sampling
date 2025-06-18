### script for submission of classical shadows to IBM runtime. Replace FakeBrisbane with desired device

import qiskit
from qiskit import transpile
from qiskit_ibm_runtime.fake_provider import FakeBrisbane
import pickle

nshots = 8000

backend = FakeBrisbane()
measurement_result = []
for cliff_count in range(nshots):
    circuit2=qiskit.QuantumCircuit.from_qasm_file("qasm_circuits/classical_shadow_circuit_"+str(cliff_count)+".qasm")
    qc_compiled = transpile(circuit2, backend)
    job_sim = backend.run(qc_compiled, shots=1)
    result_sim = job_sim.result()
    measurement_result.append(list(result_sim.get_counts().keys())[0])

with open('measurement_results_shadows.pkl', 'wb') as f:
    pickle.dump(measurement_result, f)

f_name="measurement_results_shadows.txt"
f = open(f_name, "a")
for m in measurement_result:
    f.write(str(m)+"\n")
f.close()   
