"""
Intuitive Molecule class.
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

from .utils import parse_xyz_file

class Molecule:
    """
    Notes:
        - When passing in molecule information to ansatz, use problem.num_particles, etc. instead of instance attributes.
        This is because of possible active space restrictions.
    Important instance attributes:
        - driver: PySCFDriver instance
        - problem: ElectronicStructureProblem instance
        - second_q_op
        - num_particles
        - num_spin_orbitals
        - converter: QubitConverter instance wrapping qubit_mapper instance
        - qubit_op
        - init_state: HartreeFock initial state for VQE
        - active_space_energy_shift: Energy shift from restricting active space (0 if no restriction)
        - nuclear_repulsion_energy
        - ee_property: very useful, see https://qiskit.org/documentation/nature/stubs/qiskit_nature.properties.second_quantization.electronic.ElectronicEnergy.html#qiskit_nature.properties.second_quantization.electronic.ElectronicEnergy
    """
    def __init__(self, xyz_file, active_num_electrons = None, active_num_molecular_orbitals = None, active_orbitals = None, qubit_mapper = ParityMapper(), two_qubit_reduction = True, z2symmetry_reduction = 'auto', custom_second_q_op = None, spin = 0, charge = 0, basis='sto3g'):
        self.xyz_file = xyz_file
        self.active_num_electrons = active_num_electrons
        self.active_num_molecular_orbitals = active_num_molecular_orbitals
        self.active_orbitals = active_orbitals
        self.qubit_mapper = qubit_mapper
        self.two_qubit_reduction = two_qubit_reduction
        self.z2symmetry_reduction = z2symmetry_reduction
        self.custom_second_q_op = custom_second_q_op
        self.spin = spin
        self.charge = charge
        self.basis = basis
        self.make_molecule()

    def make_molecule(self):
        self.atom_str = parse_xyz_file(self.xyz_file)
        self.set_basic_properties()

    def set_basic_properties(self):
        # Open to extension if anyome figures out Daubechies Wavelets https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.3.020360
        self.driver = PySCFDriver(
            atom=self.atom_str,
            unit=UnitsType.ANGSTROM,
            basis=self.basis,
            spin=self.spin,
            charge=self.charge,
        )
        if self.active_num_electrons is not None:
            self.problem = ElectronicStructureProblem(self.driver, [ActiveSpaceTransformer(
                self.active_num_electrons,
                self.active_num_molecular_orbitals,
                self.active_orbitals,
            )])
        else:
            self.problem = ElectronicStructureProblem(self.driver)
        self.second_q_op = self.problem.second_q_ops()[0].simplify()
        if self.custom_second_q_op is not None:
            self.second_q_op = self.custom_second_q_op
        self.num_particles = self.problem.num_particles
        self.num_spin_orbitals = self.problem.num_spin_orbitals
        # TODO: fix z2symmetry_reduction
        self.converter = QubitConverter(mapper=self.qubit_mapper, two_qubit_reduction=self.two_qubit_reduction, z2symmetry_reduction=None)
        self.qubit_op = self.converter.convert(self.second_q_op, num_particles=self.num_particles)
        # Construct Hartree Fock initial state
        self.init_state = HartreeFock(self.num_spin_orbitals, self.num_particles, self.converter)
        # If need any other similar properties, either manually get it from self.problem etc. or extend code
        # Get energy shift from ActiveSpaceTransformer
        self.active_space_energy_shift = 0
        self.ee_property = self.problem.grouped_property_transformed.get_property("ElectronicEnergy")
        if hasattr(self.ee_property, '_shift') and 'ActiveSpaceTransformer' in self.ee_property._shift:
            self.active_space_energy_shift = np.real(self.ee_property._shift['ActiveSpaceTransformer'])
        self.nuclear_repulsion_energy = self.ee_property.nuclear_repulsion_energy

