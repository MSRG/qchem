"""
Functions to make particular ansatzes.
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_nature.drivers.second_quantization import PySCFDriver
from qiskit_nature.drivers import UnitsType
from qiskit_nature.transformers.second_quantization.electronic import ActiveSpaceTransformer
from qiskit_nature.problems.second_quantization.electronic import ElectronicStructureProblem
from qiskit_nature.mappers.second_quantization import *
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.circuit.library import *
from qiskit.circuit.library import *
from qiskit.algorithms.optimizers import *
from qiskit.algorithms import VQE
from qiskit.opflow.expectations import *
from qiskit.test.mock import *
from qiskit.providers.aer import AerSimulator
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.utils.mitigation import CompleteMeasFitter, TensoredMeasFitter
from qiskit.compiler import transpile
from qiskit.opflow import (
    CircuitStateFn,
    ExpectationBase,
    ExpectationFactory,
    ListOp,
    OperatorBase,
    PauliSumOp,
    StateFn,
)
from qiskit.opflow.converters import CircuitSampler
from mitiq.zne import execute_with_zne

from .Molecule import Molecule


def make_uccsd_ansatz(molecule: Molecule) -> QuantumCircuit:
    converter = molecule.converter
    problem = molecule.problem
    num_particles = problem.num_particles
    num_spin_orbitals = problem.num_spin_orbitals
    initial_state = molecule.init_state
    ansatz = UCCSD(
        qubit_converter = converter,
        num_particles = num_particles,
        num_spin_orbitals = num_spin_orbitals,
        initial_state = initial_state,
    )

    # Assign a run name
    ansatz._run_name = 'uccsd'

    return ansatz


def make_hwe_ansatz(molecule: Molecule, depth: int) -> QuantumCircuit:
    """
    Implements https://www.nature.com/articles/nature23879.

    Code taken from https://github.com/lockwo/Paper-Review/blob/main/HEA-VQE/hea_vqe.ipynb.
    """
    num_q = molecule.qubit_op.num_qubits
    circuit = QuantumCircuit(num_q)
    params = ParameterVector("theta", length=num_q * (3 * depth + 2))
    counter = 0
    for q in range(num_q):
        circuit.rx(params[counter], q)
        counter += 1 
        circuit.rz(params[counter], q)
        counter += 1
    for d in range(depth):
        for q in range(num_q - 1):
            circuit.cx(q, q + 1)
        for q in range(num_q):
            circuit.rz(params[counter], q)
            counter += 1
            circuit.rx(params[counter], q)
            counter += 1 
            circuit.rz(params[counter], q)
            counter += 1

    # Assign a run name
    circuit._run_name = f'hwe-{depth}'

    return circuit


def make_esu2_ansatz(molecule: Molecule, su2_gates=['ry', 'rz']) -> QuantumCircuit:
    """
    Ansatz alternates between su2_gates layer and entanglement layer.
    
    Notes:
        - Recommended to toggle between ['ry'] and ['ry', 'rz'] for su2_gates.
    """
    # Should keep entanglement as linear and reps as 1 to minimize depth
    ansatz = EfficientSU2(molecule.qubit_op.num_qubits, su2_gates=su2_gates, entanglement='linear', reps=1)
    return ansatz


def make_excitation_preserving_ansatz(molecule: Molecule) -> QuantumCircuit:
    ansatz = ExcitationPreserving(
        molecule.qubit_op.num_qubits,
        entanglement = 'linear',
        initial_state = molecule.init_state,
        reps = 1,
    )
    ansatz._preferred_init_points = np.zeros(ansatz.num_parameters)
    return ansatz


def make_uccgsd_ansatz(molecule: Molecule) -> QuantumCircuit:
    converter = molecule.converter
    problem = molecule.problem
    num_particles = problem.num_particles
    num_spin_orbitals = problem.num_spin_orbitals
    initial_state = molecule.init_state
    ansatz = UCCSD(
        qubit_converter = converter,
        num_particles = num_particles,
        num_spin_orbitals = num_spin_orbitals,
        initial_state = initial_state,
        generalized=True,
    )
    return ansatz

def make_upccgsd_ansatz(molecule: Molecule) -> QuantumCircuit:
    converter = molecule.converter
    problem = molecule.problem
    num_particles = problem.num_particles
    num_spin_orbitals = problem.num_spin_orbitals
    initial_state = molecule.init_state
    ansatz = PUCCD(
        qubit_converter = converter,
        num_particles = num_particles,
        num_spin_orbitals = num_spin_orbitals,
        initial_state = initial_state,
        include_singles = (True, True),
        generalized=True,
    )
    return ansatz

def make_k_upccgsd_ansatz(molecule: Molecule, k = 1):
    # Follows https://arxiv.org/abs/1810.02327
    ansatz = QuantumCircuit(upccgsd.num_qubits)
    upccgsd = make_upccgsd_ansatz(molecule)
    n = upccgsd.num_parameters
    theta = ParameterVector('theta', n * k)
    for i in range(k):
        ansatz = ansatz.compose(upccgsd.assign_parameters(theta[i: i+n]))
    return ansatz
