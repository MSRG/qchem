import datetime
import json
import importlib

import click
import numpy as np
from qiskit_ibm_runtime import QiskitRuntimeService, Estimator, Sampler, Session, Options
from qiskit.primitives import BackendEstimator, BackendSampler
from qiskit.providers.fake_provider import FakeLagos
from qiskit_nature.drivers.second_quantization import PySCFDriver
from qiskit_nature.drivers import UnitsType
from qiskit_nature.problems.second_quantization.electronic import ElectronicStructureProblem
from qiskit_nature.transformers.second_quantization.electronic import ActiveSpaceTransformer
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import ParityMapper, JordanWignerMapper, BravyiKitaevMapper
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_nature.circuit.library import *
from qiskit.circuit.library import *
from qiskit.algorithms.optimizers import SPSA, COBYLA
from qiskit.ignis.verification.tomography import state_tomography_circuits, StateTomographyFitter
from qiskit.compiler import transpile
import mthree
from zne import zne, ZNEStrategy
from zne.extrapolation import *

# TODO: turn start/end logging into a decorator

# Global variables
RUNTIME = False

XYZ_FILE = None
CHARGE = None
SPIN = None
BASIS = None
NUM_ACTIVE_ELECTRONS = None
NUM_ACTIVE_ORBITALS = None
ACTIVE_ORBITALS = None
MAPPER = None
ANSATZ = None
OPTIMIZER = None
BACKEND = None
SHOTS = None
OPTIMIZATION_LEVEL = None
RESILIENCE_LEVEL = None

LAST_MTHREE_CALIBRATION = None


############################################
# Utility functions
############################################


def get_counts(result, num_qbits):
    shots = result.metadata[0]['shots']
    quasi_dists = result.quasi_dists
    counts = []
    for qd in quasi_dists:
        d = {}
        for k, v in qd.items():
            k = bin(k)[2:].zfill(num_qbits)
            v = int(v * shots)
            d[k] = v
        counts.append(d)
    return counts


def get_qd(counts):
    qd = []
    for count in counts:
        d = {}
        for k, v in count.items():
            k = int(k, base=2)
            d[k] = v
        qd.append(d)
    return qd


def parse_xyz_file(xyz_file):
    xyz = []
    with open(xyz_file) as fp:
        n = 1
        for line in fp:
            line = line.strip()
            if line.isdigit():
                n = int(line) + 1
                continue
            xyz.append(line)
            n -= 1
            if n == 0:
                break
    atom_str = xyz[1:]
    return atom_str


def parse_settings(settings_file):
    print(f'Parsing settings at {datetime.datetime.now().isoformat()}', flush=True)

    with open(settings_file, 'r') as fp:
        settings = json.load(fp)
    
    global XYZ_FILE
    XYZ_FILE = settings['XYZ_FILE']

    global CHARGE
    CHARGE = settings['CHARGE']

    global SPIN
    SPIN = settings['SPIN']

    global BASIS
    BASIS = settings['BASIS']

    global NUM_ACTIVE_ELECTRONS
    NUM_ACTIVE_ELECTRONS = settings['NUM_ACTIVE_ELECTRONS']

    global NUM_ACTIVE_ORBITALS
    NUM_ACTIVE_ORBITALS = settings['NUM_ACTIVE_ORBITALS']

    global ACTIVE_ORBITALS
    ACTIVE_ORBITALS = settings['ACTIVE_ORBITALS']

    global MAPPER
    module = importlib.import_module('qiskit_nature.mappers.second_quantization')
    MAPPER = getattr(module, settings['MAPPER'])()

    global ANSATZ
    ANSATZ = settings['ANSATZ']

    global OPTIMIZER
    module = importlib.import_module('qiskit.algorithms.optimizers')
    if 'MAXITER' not in settings:
        OPTIMIZER = getattr(module, settings['OPTIMIZER'])()
    else:
        OPTIMIZER = getattr(module, settings['OPTIMIZER'])(maxiter=settings['MAXITER'])

    global BACKEND
    if RUNTIME == True:
        BACKEND = settings['BACKEND']
    else:
        module = importlib.import_module('qiskit.providers.fake_provider')
        BACKEND = getattr(module, settings['BACKEND'])()

    global SHOTS
    SHOTS = settings['SHOTS']

    global OPTIMIZATION_LEVEL
    OPTIMIZATION_LEVEL = settings['OPTIMIZATION_LEVEL']

    global RESILIENCE_LEVEL
    RESILIENCE_LEVEL = settings['RESILIENCE_LEVEL']

    print(f'Settings parsed at {datetime.datetime.now().isoformat()}', flush=True)


############################################
# Ansatz
############################################


def make_hwe_ansatz(num_qubits, depth):
    """
    Implements https://www.nature.com/articles/nature23879.
    Code taken from https://github.com/lockwo/Paper-Review/blob/main/HEA-VQE/hea_vqe.ipynb.

    :param num_qubits: Number of qubits (qubit_op.num_qubits)
    """
    num_q = num_qubits
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

    return circuit

def make_esu2_ansatz(num_qubits, su2_gates=['ry', 'rz']):
    """
    Ansatz alternates between su2_gates layer and entanglement layer.
    
    Notes:
        - Recommended to toggle between ['ry'] and ['ry', 'rz'] for su2_gates.

    :param num_qubits: Number of qubits (qubit_op.num_qubits)
    """
    # Should keep entanglement as linear and reps as 1 to minimize depth
    ansatz = EfficientSU2(num_qubits, su2_gates=su2_gates, entanglement='linear', reps=1)
    return ansatz


def make_excitation_preserving_ansatz(num_qubits, initial_state = None):
    """
    Excitation preserving ansatz.

    :param num_qubits: Number of qubits (qubit_op.num_qubits)
    :param initial_state: Initial state of ansatz (ie. Hartree-Fock initial state)
    """
    ansatz = ExcitationPreserving(
        num_qubits,
        entanglement = 'linear',
        initial_state = initial_state,
        reps = 1,
    )
    ansatz._preferred_init_points = np.zeros(ansatz.num_parameters)
    return ansatz


############################################
# main
############################################


@click.command()
@click.option('--runtime', type=bool, default=False, help='Set True to run with Qiskit runtime.')
@click.option('--channel', help='IBM Quantum credentials: see https://qiskit.org/documentation/partners/qiskit_ibm_runtime/getting_started.html')
@click.option('--token', help='IBM Quantum credentials: see https://qiskit.org/documentation/partners/qiskit_ibm_runtime/getting_started.html')
@click.option('--settings', required=True, help='Settings file for the VQE algorithm.')
def main(runtime, channel, token, settings):
    start = datetime.datetime.now()
    print(f'Start at {start.isoformat()}', flush=True)
    
    set_credentials(runtime, channel, token)
    run(settings)

    end = datetime.datetime.now()
    print(f'End at {end.isoformat()}', flush=True)
    print(f'Runtime is {(end - start).seconds}s', flush=True)
    print('DONE', flush=True)


def set_credentials(runtime, channel, token):
    """
    Set the credentials for the IBM Quantum account.

    :param runtime: Set True to run with Qiskit runtime.
    :param channel: IBM Quantum credentials: see https://qiskit.org/documentation/partners/qiskit_ibm_runtime/getting_started.html
    :param token: IBM Quantum credentials: see https://qiskit.org/documentation/partners/qiskit_ibm_runtime/getting_started.html
    """
    print(f'Setting credentials at {datetime.datetime.now().isoformat()}', flush=True)

    if runtime == True:
        print('Running with Qiskit runtime')
        QiskitRuntimeService.save_account(channel=channel, token=token, overwrite=True)
        global RUNTIME
        RUNTIME = True
    else:
        print('Running locally')

    print(f'Credentials set at {datetime.datetime.now().isoformat()}', flush=True)


def run(settings_file):
    """
    Run the VQE algorithm with the given settings.

    :param settings_file: Settings file for the VQE algorithm.
    """
    parse_settings(settings_file)

    print(f'PYSCF calculation start at {datetime.datetime.now().isoformat()}', flush=True)

    driver = PySCFDriver(atom=parse_xyz_file(XYZ_FILE), unit=UnitsType.ANGSTROM, charge=CHARGE, spin=SPIN, basis=BASIS)
    problem = ElectronicStructureProblem(driver, [ActiveSpaceTransformer(
        NUM_ACTIVE_ELECTRONS,
        NUM_ACTIVE_ORBITALS,
        ACTIVE_ORBITALS,
    )])
    second_q_ops = problem.second_q_ops()
    converter = QubitConverter(mapper=MAPPER, two_qubit_reduction=True)
    qubit_op = converter.convert(
        second_q_op=second_q_ops['ElectronicEnergy'],
        num_particles=problem.num_particles,
    )
    ee_property = problem.grouped_property_transformed.get_property('ElectronicEnergy')
    nuclear_repulsion_energy = ee_property.nuclear_repulsion_energy
    active_space_energy_shift = 0
    if hasattr(ee_property, '_shift') and 'ActiveSpaceTransformer' in ee_property._shift:
        active_space_energy_shift = np.real(ee_property._shift['ActiveSpaceTransformer'])

    print(f'nuclear_repulsion_energy={nuclear_repulsion_energy}')
    print(f'active_space_energy_shift={active_space_energy_shift}')

    print(f'PYSCF calculation end at {datetime.datetime.now().isoformat()}', flush=True)

    global ANSATZ
    if ANSATZ.startswith('hwe-'):
        depth = int(ANSATZ.split('-')[1])
        ANSATZ = make_hwe_ansatz(qubit_op.num_qubits, depth)
    elif ANSATZ.startswith('esu2-'):
        su2_gates = ANSATZ.split('-')[1].split(',')
        ANSATZ = make_esu2_ansatz(qubit_op.num_qubits, su2_gates)
    elif ANSATZ == 'excitation_preserving':
        initial_state = HartreeFock(
            problem.num_spin_orbitals,
            problem.num_particles,
            converter,
        )
        ANSATZ = make_excitation_preserving_ansatz(qubit_op.num_qubits, initial_state)
    else:
        raise NotImplementedError(f'Ansatz {ANSATZ} not implemented')

    initial_point = np.random.uniform(-np.pi, np.pi, len(ANSATZ.parameters))
    # if hasattr(ANSATZ, '_preferred_init_points'):
    #     initial_point = ANSATZ._preferred_init_points
    
    def vqe(estimator, sampler, zne_strategy = None):
        with open('output.txt', 'w') as fp:
            fp.write('iteration\tenergy\ttime\tparameters\n')
        
        def get_energy():

            iteration = 1
            if RESILIENCE_LEVEL == 6:
                mit = mthree.M3Mitigation(sampler)

            def energy(theta):
                nonlocal iteration
                nonlocal mit
                nonlocal estimator
                nonlocal zne_strategy

                if RUNTIME == False and RESILIENCE_LEVEL == 2:
                    result = estimator.run(circuits=ANSATZ, observables=qubit_op, parameter_values=[theta], shots=SHOTS, zne_strategy=zne_strategy).result().values[0]
                elif RESILIENCE_LEVEL < 4:
                    result = estimator.run(circuits=ANSATZ, observables=qubit_op, parameter_values=[theta], shots=SHOTS).result().values[0]
                elif RESILIENCE_LEVEL == 5 or RESILIENCE_LEVEL == 6:
                    num_qubits = sampler.backend.configuration().num_qubits
                    circuit = ANSATZ.assign_parameters(theta)
                    tomography_circuits = state_tomography_circuits(circuit, circuit.qubits)
                    job = sampler.run(circuits=tomography_circuits, shots=SHOTS)
                    result = job.result()

                    if RESILIENCE_LEVEL == 6:
                        global LAST_MTHREE_CALIBRATION
                        # mthree recalibrate every 300s
                        if LAST_MTHREE_CALIBRATION is None or (datetime.datetime.now() - LAST_MTHREE_CALIBRATION).seconds > 300:
                            LAST_MTHREE_CALIBRATION = datetime.datetime.now()
                            mit.cals_from_system()

                        counts = get_counts(result, num_qubits)
                        qd = get_qd(mit.apply_correction(counts, range(num_qubits)))
                        for i in range(len(result.quasi_dists)):
                            result.quasi_dists[i] = qd[i]
                    fitter = StateTomographyFitter(result, tomography_circuits)
                    density_matrix = fitter.fit()
                    eigenvalues, eigenvectors = np.linalg.eig(density_matrix)
                    # The imaginary parts are very small due to precision errors
                    eigenvalues = [np.real(ev) for ev in eigenvalues]
                    psi = eigenvectors[:, np.argmax(eigenvalues)]
                    psi.reshape((len(psi),))
                    H = qubit_op.to_matrix()
                    # np.inner doesn't do complex conjugate
                    result = np.real(np.inner(psi.conjugate(), H.dot(psi)))
                else:
                    pass

                # logging
                with open('output.txt', 'a') as fp:
                    parameters = str(theta).replace('\n', '')
                    fp.write(f'{iteration}\t{result}\t{datetime.datetime.now().isoformat()}\t{parameters}\n')
                    iteration += 1

                return result

            return energy
        
        energy = get_energy()

        OPTIMIZER.minimize(energy, initial_point)

    if RUNTIME == True:
        service = QiskitRuntimeService()
        with Session(service=service, backend=BACKEND):
            if RESILIENCE_LEVEL == 2:
                options = Options(optimization_level=OPTIMIZATION_LEVEL, resilience_level=RESILIENCE_LEVEL)
                options.resilience.noise_factors = [1, 3, 5]
                options.resilience.extrapolator = "LinearExtrapolator"
                estimator = Estimator(options=options)
            elif RESILIENCE_LEVEL < 4:
                estimator = Estimator(options=Options(
                    optimization_level=OPTIMIZATION_LEVEL,
                    resilience_level=RESILIENCE_LEVEL,
                ))
                sampler = None
            elif RESILIENCE_LEVEL == 4:
                raise NotImplementedError('Matrix-free not implemented.')
            elif RESILIENCE_LEVEL == 5:
                estimator = None
                sampler = Sampler(options=Options(
                    optimization_level=OPTIMIZATION_LEVEL,
                    resilience_level=0,
                ))
            elif RESILIENCE_LEVEL == 6:
                estimator = None
                sampler = Sampler(options=Options(
                    optimization_level=OPTIMIZATION_LEVEL,
                    resilience_level=1,
                ))
            else:
                raise NotImplementedError(f'Resilience level {RESILIENCE_LEVEL} does not exist.')
            
            vqe(estimator, sampler)
    else:
        if RESILIENCE_LEVEL == 2:
            zne_strategy = ZNEStrategy(noise_factors=[1, 3, 5], extrapolator=PolynomialExtrapolator(degree=1))
            zne_estimator = zne(BackendEstimator)
            estimator = zne_estimator(backend=BACKEND)
            sampler = None
        elif RESILIENCE_LEVEL < 4:
            estimator = BackendEstimator(backend=BACKEND, options={
                'optimization_level': OPTIMIZATION_LEVEL,
                'resilience_level': RESILIENCE_LEVEL,
            })
            sampler = None
        elif RESILIENCE_LEVEL == 4:
            raise NotImplementedError('Matrix-free not implemented.')
        elif RESILIENCE_LEVEL == 5:
            estimator = None
            sampler = BackendSampler(backend=BACKEND, options={
                'optimization_level': OPTIMIZATION_LEVEL,
                'resilience_level': 0,
            })
        elif RESILIENCE_LEVEL == 6:
            estimator = None
            sampler = BackendSampler(backend=BACKEND, options={
                'optimization_level': OPTIMIZATION_LEVEL,
                'resilience_level': 1,
            })
        else:
            raise NotImplementedError(f'Resilience level {RESILIENCE_LEVEL} does not exist.')

        vqe(estimator, sampler, zne_strategy)
        

if __name__ == '__main__':
    main()
