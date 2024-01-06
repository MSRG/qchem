import os
from qiskit_nature.mappers.second_quantization import ParityMapper
from msrg_vqe import Molecule, VQE
from msrg_vqe.ansatz import make_uccsd_ansatz
from qiskit.algorithms.optimizers import COBYLA
from qiskit.providers.aer import AerSimulator
from qiskit.providers.fake_provider import FakeManila


class TestCoords:
    """
    Context manager to setup a test .xyz file.
    """
    def __init__(self, xyz_file):
        self.xyz_file = xyz_file
    def __enter__(self):
        # Test .xyz file
        xyz = '2\n\nH 0 0 0\nH 0 0 0.743'
        with open(self.xyz_file, 'w') as fp:
            fp.write(xyz)
    
    def __exit__(self, *args):
        # Cleanup
        if os.path.exists(self.xyz_file):
            os.remove(self.xyz_file)

with TestCoords('test.xyz'):
    molecule = Molecule(
        xyz_file='test.xyz',
        active_num_electrons=2,
        active_num_molecular_orbitals=2,
        active_orbitals=[0, 1],
        qubit_mapper=ParityMapper(),
        two_qubit_reduction=True,
        spin=0,
        charge=0,
        basis='sto3g',
    )

ansatz = make_uccsd_ansatz(molecule)
optimizer=COBYLA()
mitigation_settings = {
    'readout': 'matrix-full', # matrix-full | matrix-tensored | matrix-free,
    'readout-refresh': 5, # recalibrate every 5 min
    'zne': False, # use zero noise extrapolation
    'stc': False,
}
backend = AerSimulator.from_backend(FakeManila())

vqe = VQE(
    molecule=molecule,
    ansatz=ansatz,
    optimizer=optimizer,
    shots=8192,
    backend=backend,
    mitigation_settings=mitigation_settings,
    gradient=None,
    debug=1,
)

def callback(*args):
    print(args)

vqe.get_energy(callback=callback)




