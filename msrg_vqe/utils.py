"""
Useful utility functions.
"""
from typing import List, Tuple

import pyscf
from qiskit_nature.drivers.second_quantization import PySCFDriver
from qiskit_nature.drivers import UnitsType
from qiskit_nature.operators.second_quantization import FermionicOp
from numpy import pi
from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import TransformationPass
from .lib.folding import fold_gates_from_left

def parse_xyz_file(xyz_file: str) -> List[str]:
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


def get_orbital_info(xyz_file: str, spin = 0, charge = 0, basis = 'sto3g') -> List[float]:
    """
    Return list of orbital energies and list of orbital occupancies.
    """
    mol = pyscf.M(
        atom=xyz_file,
        basis=basis,
        spin=spin,
        charge=charge,
    )
    hf = mol.HF()
    hf.kernel()
    return hf.mo_energy, hf.mo_occ



def openfermion_to_qiskit(terms: dict, num_particles: Tuple[int, int]) -> FermionicOp:
    """
    Convert OpenFermion FermionOperator terms dictionary to Qiskit FermionicOp.
    
    Notes: 
    - OpenFermion and Qiskit use different orderings for their operators, see 
    https://qiskit.slack.com/archives/C7SJ0PJ5A/p1661449277983889.
    """
    if num_particles[0] != num_particles[1]:
        # TODO: What happens if number of alpha and beta particles are not equal in the case of OpenFermion?
        raise NotImplementedError('Unhandled case where number of alpha and beta are not equal.')
    
    n_MOs = sum(num_particles)
    n_SOs = n_MOs * 2

    op = FermionicOp('') * 0

    for k, v in terms.items():
        op_str = ''
        for op_tuple in k:
            if op_tuple[1] == 1:
                op_type = '+'
            else:
                op_type = '-'
            index = op_tuple[0] // 2
            if op_tuple[0] % 2 == 1:
                index += n_MOs
            op_str += f' {op_type}_{index}'
        op += FermionicOp(op_str, register_length=n_SOs) * v
    return op.simplify()


class QiskitToMitiqTranslator(TransformationPass):
            """
            Convert gates Mitiq does not support to supported ones.
            See https://github.com/unitaryfund/mitiq/issues/558.
            """
            def run(self, dag):
                for node in dag.op_nodes():
                    # sx to u3
                    if node.op.name == 'sx':
                        replacement = QuantumCircuit(1)
                        replacement.u3(pi/2, -pi/2, pi/2, 0)
                        dag.substitute_node_with_dag(node, circuit_to_dag(replacement))
                return dag


class MitiqToQiskitTranslator(TransformationPass):
    """
    Reverses operation of QiskitToMitiqTranslator.
    """
    def run(self, dag):
        for node in dag.op_nodes():
            # revert sx
            if node.op.name == 'u3':
                # TODO: verify parameters if more than one gate is converted to u3
                replacement = QuantumCircuit(1)
                replacement.sx(0)
                dag.substitute_node_with_dag(node, circuit_to_dag(replacement))
        return dag


def fold_gates(circuit, scale_factor):
    translated = QiskitToMitiqTranslator()(circuit)
    folded = fold_gates_from_left(translated, scale_factor)
    result = MitiqToQiskitTranslator()(folded)

    return result


def get_used_qubits(circuit):
    """Get a list of quantum registers of qubits which have a gate using them."""
    from qiskit.converters import circuit_to_dag

    dag = circuit_to_dag(circuit)
    s = set()
    for node in dag.op_nodes():
        s = s.union(set(node.qargs))
    return list(s)
    