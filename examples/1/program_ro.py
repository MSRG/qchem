import sys
# IMPORTANT: path subjective to change
sys.path.insert(0, '/home/freeman/github/qchem')

from typing import List, Union
import os
import shutil
import json
import msrg_vqe
import msrg_vqe.ansatz
from msrg_vqe import Molecule, VQE
from qiskit_nature.mappers.second_quantization import QubitMapper, ParityMapper
from qiskit.algorithms.optimizers import Optimizer, COBYLA, SPSA
from qiskit.opflow.gradients import GradientBase, Gradient
from qiskit.providers.aer import AerSimulator
from qiskit.providers.fake_provider import FakeLagos, FakeGuadalupe

# SET THIS
xyz_namespace = '/home/freeman/github/qchem/examples/1/xyz'

argv = sys.argv
configuration_number, folder = int(argv[1]), argv[2]

# Utility
def generate_permutations(lengths):
    if len(lengths) <= 0:
        return [[]]
    else:
        left = lengths[0]
        right = lengths[1:]
        p = generate_permutations(right)
        sol = []
        for i in range(left):
            for x in p:
                sol.append([i] + x.copy())
        return sol

def setup(logging_path, xyz_file, kwargs):
    """
    Creates subdirectory, setup.json, and returns callback function for VQE.
    """
    shutil.copyfile(xyz_file, os.path.join(logging_path, 'molecule.xyz'))
    setup = {}
    for key in kwargs:
        if hasattr(kwargs[key], '_run_name'):
            setup[key] = kwargs[key]._run_name
        else:
            setup[key] = kwargs[key]
    print(setup)
    with open(os.path.join(logging_path, 'setup.json'), 'w') as fp:
        json.dump(setup, fp)
    def callback(d):
        for key in d:
            with open(os.path.join(logging_path, f'{key}.log'), 'a') as fp:
                fp.write(f'{str(d[key])}\n')
    return callback


# Get xyz files
xyz_files = next(os.walk(xyz_namespace))[2]
xyz_files = [os.path.join(xyz_namespace, xyz_file) for xyz_file in xyz_files]
print('xyz_files:', xyz_files, flush=True)

# Settings setup
active_space_kwargs = [
    {
        'active_num_electrons': 2,
        'active_num_molecular_orbitals': 2,
        'active_orbitals': [37, 38],
    },
]

parity_mapper = ParityMapper()
parity_mapper._run_name = 'parity_mapper'

qubit_mapper = [
    parity_mapper,
]

two_qubit_reduction = [
    True,
]

ansatz = [
    'hwe-1',
]

cobyla = COBYLA(maxiter=251)
cobyla._run_name = 'cobyla'

optimizer = [
    cobyla,
]

fin_diff = Gradient('fin_diff')
fin_diff._run_name = 'fin_diff'
gradient = [
    fin_diff,
]

shots = [
    8192,
]

fake_lagos = AerSimulator.from_backend(FakeLagos())
fake_lagos._run_name = 'fake_lagos'
fake_guadalupe = AerSimulator.from_backend(FakeGuadalupe())
fake_guadalupe._run_name = 'fake_guadalupe'

mitigation_settings = [
    {
        'readout': 'matrix-full',
        'readout-refresh': 5,
        'zne': False,
        'stc': False,
    },
]

molecule_settings = [
    {
        'charge': 0,
        'spin': 0,
        'basis': 'sto-3g',
    },
]

def run(
    xyz_file: List[str],
    active_space_kwargs: List[dict],
    qubit_mapper: List[QubitMapper],
    two_qubit_reduction: List[bool],
    ansatz: List[str],
    optimizer: List[Optimizer],
    gradient: List[Union[GradientBase, None]],
    shots: List[int],
    mitigation_settings: List[dict],
    molecule_settings: List[dict]
):
    all_kwargs = locals()
    keys = list(all_kwargs.keys())
    lengths = [len(all_kwargs[key]) for key in keys]
    permutations = generate_permutations(lengths)
    print(len(permutations), flush=True)
    print(permutations, flush=True)
    p = permutations[configuration_number]
    print(f'RUNNING CONFIGURATION {configuration_number}:', p)
    
    kwargs = {}
    for i, key in enumerate(keys):
        kwargs[key] = all_kwargs[key][p[i]]
    run_once(**kwargs)

def run_once(
    xyz_file: str,
    active_space_kwargs: dict,
    qubit_mapper: QubitMapper,
    two_qubit_reduction: bool,
    ansatz: str,
    optimizer: Optimizer,
    gradient: Union[GradientBase, None],
    shots: int,
    mitigation_settings: dict,
    molecule_settings: List[dict],
):
    molecule = Molecule(
        xyz_file=xyz_file,
        qubit_mapper=qubit_mapper,
        two_qubit_reduction=two_qubit_reduction,
        **active_space_kwargs,
        **molecule_settings,
    )

    # Make ansatz
    if ansatz == 'uccsd':
        ansatz = msrg_vqe.ansatz.make_uccsd_ansatz(molecule)
        ansatz._run_name = 'uccsd'
    if ansatz == 'hwe-1':
        ansatz = msrg_vqe.ansatz.make_hwe_ansatz(molecule, 1)
        ansatz._run_name = 'hwe-1'
    if ansatz == 'hwe-2':
        ansatz = msrg_vqe.ansatz.make_hwe_ansatz(molecule, 2)
        ansatz._run_name = 'hwe-2'
    if ansatz == 'hwe-3':
        ansatz = msrg_vqe.ansatz.make_hwe_ansatz(molecule, 3)
        ansatz._run_name = 'hwe-3'
    if ansatz == 'esu2_ry':
        ansatz = msrg_vqe.ansatz.make_esu2_ansatz(molecule, su2_gates=['ry'])
        ansatz._run_name = 'esu2_ry'
    if ansatz == 'esu2_ryrz':
        ansatz = msrg_vqe.ansatz.make_esu2_ansatz(molecule, su2_gates=['ry', 'rz'])
        ansatz._run_name = 'esu2_ryrz'
    if ansatz == 'excitation_preserving':
        ansatz = msrg_vqe.ansatz.make_excitation_preserving_ansatz(molecule)
        ansatz._run_name = 'excitation_preserving'

    # Make backend
    backend = fake_lagos
    initial_layout = None

    backend.set_options(method='density_matrix')
    vqe = VQE(
        molecule=molecule,
        ansatz=ansatz,
        optimizer=optimizer,
        shots=shots,
        backend=backend,
        mitigation_settings=mitigation_settings,
        gradient=gradient,
        initial_layout=initial_layout,
    )

    kwargs = locals()
    del kwargs['molecule']
    del kwargs['vqe']

    callback = setup(folder, xyz_file, kwargs)
    vqe.get_energy(callback=callback)

run(
    xyz_files,
    active_space_kwargs,
    qubit_mapper,
    two_qubit_reduction,
    ansatz,
    optimizer,
    gradient,
    shots,
    mitigation_settings,
    molecule_settings,
)
