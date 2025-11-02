from qiskit import QuantumCircuit

# Create a simple quantum circuit
qc = QuantumCircuit(2)

qc.h(0)  # Apply Hadamard gate for superposition
qc.cx(0, 1)  # Apply CNOT gate for entanglement

qc.measure_all()

# Simulate the quantum circuit
from qiskit import Aer, execute

simulator = Aer.get_backend('qasm_simulator')

result = execute(qc, simulator, shots=1000).result()
counts = result.get_counts(qc)

print(counts)