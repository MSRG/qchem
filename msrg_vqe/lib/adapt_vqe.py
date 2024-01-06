# Modified from https://github.com/Qiskit/qiskit-nature/blob/0.4.0/qiskit_nature/algorithms/ground_state_solvers/adapt_vqe.py.
# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A ground state calculation employing the AdaptVQE algorithm."""

from typing import Optional, List, Tuple, Union

import copy
import re
import logging

import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.opflow import OperatorBase, PauliSumOp
from qiskit.opflow.gradients import GradientBase, Gradient
from qiskit.utils.validation import validate_min
from qiskit_nature.exceptions import QiskitNatureError

from .circuit_sampler import CircuitSampler
from ..VQE import VQE

logger = logging.getLogger(__name__)


class AdaptVQE:
    """A ground state calculation employing the AdaptVQE algorithm.

    The performance of AdaptVQE significantly depends on the choice of `gradient` (see also
    `qiskit.opflow.gradients`) and its parameters such as `grad_method`, `qfi_method` (if
    applicable) and `epilson`.

    To reproduce the default behavior of AdaptVQE prior to Qiskit Nature 0.4 you should supply
    `delta=1` explicitly. This will use a finite difference scheme for the gradient evaluation
    whereas after version 0.4 a parameter shift gradient will be used.
    For more information refer to the gradient framework of Qiskit Terra:
    https://qiskit.org/documentation/tutorials/operators/02_gradients_framework.html
    """

    # pylint: disable=unused-argument
    def __init__(
        self,
        vqe: VQE,
        threshold: float = 1e-5,
        max_iterations: Optional[int] = None,
    ) -> None:

        validate_min("threshold", threshold, 1e-15)

        # This is the ending ansatz, ie. the ansatz right before result is returned; different from self._ansatz
        self.ansatz = None
        self.vqe = vqe
        self.molecule = self.vqe.molecule

        self._qubit_converter = self.molecule.converter

        self._threshold = threshold
        self._max_iterations = max_iterations
        self.gradient = self.vqe.gradient

        self._excitation_pool: List[OperatorBase] = []
        self._excitation_list: List[OperatorBase] = []

        self._ansatz: QuantumCircuit = None
        self._sampler: CircuitSampler = None

    @property
    def gradient(self) -> GradientBase:
        """Returns the gradient."""
        return self._gradient

    @gradient.setter
    def gradient(self, grad: Optional[GradientBase] = None) -> None:
        """Sets the gradient."""
        if grad is None:
            grad = Gradient(grad_method="param_shift")
        self._gradient = grad

    def returns_groundstate(self) -> bool:
        """Whether this class returns only the ground state energy or also the ground state itself."""
        return True

    def _compute_gradients(
        self,
        theta: List[float],
        vqe: VQE,
    ) -> List[Tuple[float, PauliSumOp]]:
        """
        Computes the gradients for all available excitation operators.

        Args:
            theta: list of (up to now) optimal parameters
            vqe: the variational quantum eigensolver instance used for solving

        Returns:
            List of pairs consisting of gradient and excitation operator.
        """
        res = []
        # compute gradients for all excitation in operator pool
        for exc in self._excitation_pool:
            # add next excitation to ansatz
            self._ansatz.operators = self._excitation_list + [exc]
            # Due to an outstanding issue in Terra, the ansatz needs to be decomposed for all
            # gradient to work correctly. Once this issue is resolved, this workaround can be
            # removed.
            vqe.ansatz = self._ansatz.decompose()
            # We need to explicitly convert this to a list in order to make the object hash-able
            param_sets = list(vqe.ansatz.parameters)
            # zip will only iterate the length of the shorter list
            theta1 = dict(zip(vqe.ansatz.parameters, theta))
            op = vqe.construct_expectation(theta1)
            # compute gradient
            state_grad = self.gradient.convert(operator=op, params=param_sets)
            # Assign the parameters and evaluate the gradient
            value_dict = {param_sets[-1]: 0.0}
            state_grad_result = self._sampler.convert(state_grad, params=value_dict).eval()
            logger.info("Gradient computed : %s", str(state_grad_result))
            res.append((np.abs(state_grad_result[-1]), exc))
        return res

    @staticmethod
    def _check_cyclicity(indices: List[int]) -> bool:
        """
        Auxiliary function to check for cycles in the indices of the selected excitations.

        Args:
            indices: the list of chosen gradient indices.
        Returns:
            Whether repeating sequences of indices have been detected.
        """
        cycle_regex = re.compile(r"(\b.+ .+\b)( \b\1\b)+")
        # reg-ex explanation:
        # 1. (\b.+ .+\b) will match at least two numbers and try to match as many as possible. The
        #    word boundaries in the beginning and end ensure that now numbers are split into digits.
        # 2. the match of this part is placed into capture group 1
        # 3. ( \b\1\b)+ will match a space followed by the contents of capture group 1 (again
        #    delimited by word boundaries to avoid separation into digits).
        # -> this results in any sequence of at least two numbers being detected
        match = cycle_regex.search(" ".join(map(str, indices)))
        logger.debug("Cycle detected: %s", match)
        # Additionally we also need to check whether the last two numbers are identical, because the
        # reg-ex above will only find cycles of at least two consecutive numbers.
        # It is sufficient to assert that the last two numbers are different due to the iterative
        # nature of the algorithm.
        return match is not None or (len(indices) > 1 and indices[-2] == indices[-1])

    def solve(self, callback):
        # setup circuit sampler
        self._sampler = CircuitSampler(self.vqe.quantum_instance)

        # We construct the ansatz once to be able to extract the full set of excitation operators.
        self._ansatz = copy.deepcopy(self.vqe.ansatz)
        self._ansatz._build()
        self._excitation_pool = copy.deepcopy(self._ansatz.operators)

        threshold_satisfied = False
        alternating_sequence = False
        max_iterations_exceeded = False
        prev_op_indices: List[int] = []
        theta: List[float] = []
        max_grad: Tuple[float, Optional[PauliSumOp]] = (0.0, None)
        iteration = 0
        while self._max_iterations is None or iteration < self._max_iterations:
            iteration += 1
            logger.info("--- Iteration #%s ---", str(iteration))
            # compute gradients

            cur_grads = self._compute_gradients(theta, self.vqe)
            # pick maximum gradient
            max_grad_index, max_grad = max(
                enumerate(cur_grads), key=lambda item: np.abs(item[1][0])
            )
            # store maximum gradient's index for cycle detection
            prev_op_indices.append(max_grad_index)
            # log gradients
            if logger.isEnabledFor(logging.INFO):
                gradlog = f"\nGradients in iteration #{str(iteration)}"
                gradlog += "\nID: Excitation Operator: Gradient  <(*) maximum>"
                for i, grad in enumerate(cur_grads):
                    gradlog += f"\n{str(i)}: {str(grad[1])}: {str(grad[0])}"
                    if grad[1] == max_grad[1]:
                        gradlog += "\t(*)"
                logger.info(gradlog)
            if np.abs(max_grad[0]) < self._threshold:
                if iteration == 1:
                    raise QiskitNatureError(
                        "Gradient choice is not suited as it leads to all zero gradients gradients. "
                        "Try a different gradient method."
                    )
                logger.info(
                    "Adaptive VQE terminated successfully with a final maximum gradient: %s",
                    str(np.abs(max_grad[0])),
                )
                threshold_satisfied = True
                break
            # check indices of picked gradients for cycles
            if self._check_cyclicity(prev_op_indices):
                logger.info("Alternating sequence found. Finishing.")
                logger.info("Final maximum gradient: %s", str(np.abs(max_grad[0])))
                alternating_sequence = True
                break
            # add new excitation to self._ansatz
            self._excitation_list.append(max_grad[1])
            theta.append(0.0)
            # run VQE on current Ansatz
            self._ansatz.operators = self._excitation_list
            self.vqe.ansatz = self._ansatz
            self.vqe.initial_point = theta
            result = self.vqe.get_energy(callback=callback)
            theta = result.x.tolist()

            # Store ending ansatz
            self.ansatz = self._ansatz.copy()
        else:
            # reached maximum number of iterations
            max_iterations_exceeded = True
            logger.info("Maximum number of iterations reached. Finishing.")
            logger.info("Final maximum gradient: %s", str(np.abs(max_grad[0])))

        if threshold_satisfied:
            finishing_criterion = "Threshold converged"
        elif alternating_sequence:
            finishing_criterion = "Aborted due to cyclicity"
        elif max_iterations_exceeded:
            finishing_criterion = "Maximum number of iterations reached"
        else:
            raise QiskitNatureError("The algorithm finished due to an unforeseen reason!")

        result.finishing_criterion = finishing_criterion

        return result
