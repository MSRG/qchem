"""
Intuitive VQE class.
"""
# Patch
from .lib.quantum_instance import QuantumInstance
import qiskit.utils
qiskit.utils.QuantumInstance = QuantumInstance
from .lib.circuit_sampler import CircuitSampler
import qiskit.opflow.converters
qiskit.opflow.converters.CircuitSampler = CircuitSampler
# Import
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
from qiskit.opflow.gradients import GradientBase
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

from .Molecule import Molecule


class VQE:
    def __init__(self, molecule: Molecule, ansatz: QuantumCircuit, optimizer: Optimizer, shots, backend, mitigation_settings, gradient: GradientBase = None, debug = 0, seed_algorithm_globals = None, seed_simulator = None, seed_transpiler = None, optimization_level = 3, initial_layout = None):
        # Debug level
        # TODO: implement good logging to ensure all settings are correctly parsed and set
        self.debug = debug

        self.molecule = molecule
        self.ansatz = ansatz
        self.optimizer = optimizer
        self.shots = shots
        self.backend = backend
        self.mitigation_settings = mitigation_settings
        self.gradient = gradient

        self.seed_algorithm_globals = seed_algorithm_globals
        self.seed_simulator = seed_simulator
        self.seed_transpiler = seed_transpiler
        self.optimization_level = optimization_level
        self.initial_layout = initial_layout

        # Set seeds
        if seed_algorithm_globals is not None:
            algorithm_globals.random_seed = seed_algorithm_globals

        # Construct and set quantum_instance
        self.quantum_instance = self.get_quantum_instance()

    def get_quantum_instance(self, shots = None, backend = None, mitigation_settings = None, seed_simulator = None, seed_transpiler = None, optimization_level = None, initial_layout = None) -> QuantumInstance:
        if shots is None:
            shots = self.shots
        if backend is None:
            backend = self.backend
        if mitigation_settings is None:
            mitigation_settings = self.mitigation_settings
        if seed_simulator is None:
            seed_simulator = self.seed_simulator
        if seed_transpiler is None:
            seed_transpiler = self.seed_transpiler
        if optimization_level is None:
            optimization_level = self.optimization_level
        if initial_layout is None:
            initial_layout = self.initial_layout
        
        kwargs = {
            'backend': backend,
            'shots': shots,
            'seed_simulator': seed_simulator,
            'seed_transpiler': seed_transpiler,
            'optimization_level': optimization_level, # Do not change this: we want circuits to be as optimized as possible
            'initial_layout': initial_layout,
        }

        # Parse mitigation_settings
        if 'readout' in mitigation_settings:
            # Which readout
            readout = mitigation_settings['readout']
            if readout == 'matrix-full':
                kwargs['measurement_error_mitigation_cls'] = CompleteMeasFitter
            if readout == 'matrix-tensored':
                kwargs['measurement_error_mitigation_cls'] = TensoredMeasFitter
            if readout == 'matrix-free':
                kwargs['mthree'] = True
            # Refresh rate
            if 'readout-refresh' in mitigation_settings:
                kwargs['cals_matrix_refresh_period'] = mitigation_settings['readout-refresh']
        
        quantum_instance = QuantumInstance(**kwargs)
        
        if self.debug > 0:
            print('quantum_instance:', {
                'backend': quantum_instance.backend,
                'measurement_error_mitigation_cls': quantum_instance.measurement_error_mitigation_cls,
                'measurement_error_mitigation_shots': quantum_instance.measurement_error_mitigation_shots,
                'cals_matrix_refresh_period': quantum_instance.cals_matrix_refresh_period,
                'mthree': quantum_instance.mthree,
            })

        return quantum_instance

    def construct_expectation(
        self,
        parameters, 
        molecule: Molecule = None,
        ansatz: QuantumCircuit = None,
        custom_observable = None,
    ):
        """Helper for AdaptVQE."""
        if molecule is None:
            molecule = self.molecule
        if ansatz is None:
            ansatz = self.ansatz
        if custom_observable is None:
            observable = molecule.qubit_op
        
        circuit = ansatz.assign_parameters(parameters)
        expectation = PauliExpectation()

        observable_meas = expectation.convert(StateFn(observable, is_measurement=True))
        ansatz_circuit_op = CircuitStateFn(circuit)
        expect_op = observable_meas.compose(ansatz_circuit_op).reduce()

        return expect_op


    def get_energy_evaluation(self, parameters, molecule: Molecule = None, ansatz: QuantumCircuit = None,  quantum_instance: QuantumInstance = None, mitigation_settings = None, custom_observable = None, callback = None, repeat = 1):
        """
        Evaluate ansatz at a set of parameters.

        Notes:
            - If want custom quantum_instance, please use get_quantum_instance and pass in one.
            - mitigation_settings should be same mitigation_settings used to create quantum_instance if provided.
        """
        if molecule is None:
            molecule = self.molecule
        if ansatz is None:
            ansatz = self.ansatz
        if quantum_instance is None:
            quantum_instance = self.quantum_instance
        if mitigation_settings is None:
            mitigation_settings = self.mitigation_settings
        zne = False
        if 'zne' in mitigation_settings and mitigation_settings['zne']:
            zne = True
        stc = False
        if 'stc' in mitigation_settings and mitigation_settings['stc']:
            stc = True
        
        cs = CircuitSampler(quantum_instance)
        circuit = ansatz.assign_parameters(parameters)
        expectation = PauliExpectation()

        if custom_observable is None:
            observable = molecule.qubit_op
        else:
            observable = custom_observable

        def execute(circuit):
            scaled_energies = None

            observable_meas = expectation.convert(StateFn(observable, is_measurement=True))
            ansatz_circuit_op = CircuitStateFn(circuit)
            expect_op = observable_meas.compose(ansatz_circuit_op).reduce()

            if stc and zne:
                raise NotImplementedError('Cannot combine stc and zne.')
            elif stc:
                mean = self.state_tomography_correction(parameters)
            elif zne:
                from mitiq.zne.inference import RichardsonFactory, LinearFactory

                scale_factors = [1.0, 3.0, 5.0]
                scaled_energies = []
                for scale_factor in scale_factors:
                    mean = np.real(cs.convert(expect_op, scale_factor=scale_factor).eval())
                    scaled_energies.append(mean)
                if 'zne_factory' not in mitigation_settings:
                    factory = RichardsonFactory(scale_factors)
                elif mitigation_settings['zne_factory'] == 'linear':
                    factory = LinearFactory(scale_factors)
                elif mitigation_settings['zne_factory'] == 'richardson':
                    factory = RichardsonFactory(scale_factors)
                else:
                    raise NotImplementedError(f'ZNE factory {mitigation_settings["zne_factory"]} not implemented.')
                
                mean = factory.extrapolate(scale_factors, scaled_energies)
            else:
                mean = np.real(cs.convert(expect_op).eval())

            return mean, scaled_energies
        
        if zne:
            energy, scaled_energies = execute(circuit)
        else:
            means = []
            for _ in range(repeat):
                mean, _ = execute(circuit)
                means.append(mean)
            energy = np.mean(means)
        
        if callback:
            import time

            information = {
                'parameters': list(parameters),
                'energy': energy,
                'time': time.time(),
            }
            if zne:
                information['scaled_energies'] = scaled_energies
                
            callback(information)

        return energy

    def get_energy(self, molecule: Molecule = None, ansatz: QuantumCircuit = None,  optimizer: Optimizer = None, quantum_instance: QuantumInstance = None, mitigation_settings = None, gradient: GradientBase = None, callback = None, repeat = 1):
        """
        Get electronic energy using VQE.

        Notes:
            - Energy shifts with active space restriction and nuclear repulsion energy not added.
            - If want custom quantum_instance, please use get_quantum_instance and pass in one.
        """
        if molecule is None:
            molecule = self.molecule
        if ansatz is None:
            ansatz = self.ansatz
        if optimizer is None:
            optimizer = self.optimizer
        if quantum_instance is None:
            quantum_instance = self.quantum_instance
        if mitigation_settings is None:
            mitigation_settings = self.mitigation_settings
        if gradient is None:
            gradient = self.gradient
        
        def energy(parameters):
            return self.get_energy_evaluation(
                parameters,
                molecule,
                ansatz,
                quantum_instance,
                mitigation_settings,
                callback=callback,
                repeat=repeat,
            )
        
        # Following code snippet taken from https://qiskit.org/documentation/_modules/qiskit/algorithms/minimum_eigen_solvers/vqe.html#VQE
        expected_size = ansatz.num_parameters
        point = None
        if hasattr(ansatz, 'preferred_init_points'):
            point = ansatz.preferred_init_points
        if hasattr(ansatz, '_preferred_init_points'):
            point = ansatz._preferred_init_points
        if point is None:
            bounds = getattr(ansatz, 'parameter_bounds', None)
            if bounds is None:
                bounds = [(-2 * np.pi, 2 * np.pi)] * expected_size
            lower_bounds = []
            upper_bounds = []
            for lower, upper in bounds:
                lower_bounds.append(lower if lower is not None else -2 * np.pi)
                upper_bounds.append(upper if upper is not None else 2 * np.pi)
            point = algorithm_globals.random.uniform(lower_bounds, upper_bounds)
        x0 = point

        if gradient is not None:
            jac = gradient.gradient_wrapper(
                ~StateFn(molecule.qubit_op) @ StateFn(ansatz),
                bind_params=list(ansatz.parameters),
                backend=quantum_instance,
            )
        else:
            jac = None

        if self.debug > 0:
            print('jac:', jac)

        return optimizer.minimize(
            fun=energy, 
            x0=list(x0), 
            jac=jac
        )
    
    
    def state_tomography(self, parameters, ansatz: QuantumCircuit = None, quantum_instance = None, shots = None, optimization_level = None):
        from qiskit.ignis.verification.tomography import state_tomography_circuits, StateTomographyFitter

        if ansatz is None:
            ansatz = self.ansatz
        if quantum_instance is None:
            quantum_instance = self.quantum_instance
        if shots is None:
            shots = self.quantum_instance.run_config.shots
        if optimization_level is None:
            optimization_level = self.optimization_level
        
        circuit = ansatz.assign_parameters(parameters)
        tomography_circuits = state_tomography_circuits(circuit, circuit.qubits)
        result = quantum_instance.execute(tomography_circuits)
        fitter = StateTomographyFitter(result, tomography_circuits)
        density_matrix = fitter.fit()
        
        return density_matrix
        

    def state_tomography_correction(self, parameters, observable = None, ansatz: QuantumCircuit = None, quantum_instance = None, shots = None):
        if ansatz is None:
            ansatz = self.ansatz
        if quantum_instance is None:
            quantum_instance = self.quantum_instance
        if shots is None:
            shots = self.quantum_instance.run_config.shots
        if observable is None:
            observable = self.molecule.qubit_op

        density_matrix = self.state_tomography(
            parameters=parameters,
            ansatz=ansatz,
            quantum_instance=quantum_instance,
            shots=shots
        )
        eigenvalues, eigenvectors = np.linalg.eig(density_matrix)
        # The imaginary parts are very small due to precision errors
        eigenvalues = [np.real(ev) for ev in eigenvalues]
        psi = eigenvectors[:, np.argmax(eigenvalues)]
        psi.reshape((len(psi),))
        H = observable.to_matrix()
        # np.inner doesn't do complex conjugate
        corrected_energy = np.inner(psi.conjugate(), H.dot(psi))

        return np.real(corrected_energy)
