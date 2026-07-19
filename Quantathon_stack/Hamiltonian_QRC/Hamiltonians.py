import numpy as np

#################################################
# Transverse Field Ising Model
##################################################
import re
from qutip import Qobj, basis, qeye, sigmax, sigmay, sigmaz, tensor


PAULI_OPERATORS = {
    "I": qeye(2),
    "X": sigmax(),
    "Y": sigmay(),
    "Z": sigmaz(),
}


def pauli_word_to_matrix(word: str) -> Qobj:
    """
    Convert a Pauli word such as 'XXII' into the QuTiP operator
        X ⊗ X ⊗ I ⊗ I
    Lowercase letters and whitespace are accepted.
    """
    word = word.replace(" ", "").upper()

    if not word:
        raise ValueError("The Pauli word cannot be empty.")

    invalid = set(word) - set(PAULI_OPERATORS)
    if invalid:
        raise ValueError(
            f"Invalid Pauli characters: {sorted(invalid)}. "
            "Only I, X, Y, and Z are allowed."
        )

    return tensor([PAULI_OPERATORS[character] for character in word])


def pauli_expression_to_matrix(expression: str) -> Qobj:
    """
    Convert a sum of Pauli words into a QuTiP operator.

    Examples
    --------
    'XXII + zzII'
    'XX - YY + ZZ'
    '2*XXII - 0.5*ZZII'
    """
    expression = expression.replace(" ", "")

    if not expression:
        raise ValueError("The expression cannot be empty.")

    # Ensure the first term has an explicit sign.
    if expression[0] not in "+-":
        expression = "+" + expression

    raw_terms = re.findall(r"([+-])([^+-]+)", expression)

    parsed_terms = []
    number_of_qubits = None

    for sign, term in raw_terms:
        match = re.fullmatch(
            r"(?:(\d+(?:\.\d*)?|\.\d+)\*)?([IXYZixyz]+)",
            term,
        )

        if match is None:
            raise ValueError(f"Could not parse term: {sign}{term}")

        coefficient_text, pauli_word = match.groups()
        coefficient = float(coefficient_text) if coefficient_text else 1.0

        if sign == "-":
            coefficient *= -1

        pauli_word = pauli_word.upper()

        if number_of_qubits is None:
            number_of_qubits = len(pauli_word)
        elif len(pauli_word) != number_of_qubits:
            raise ValueError(
                "All Pauli words must have the same length. "
                f"Expected {number_of_qubits}, got {len(pauli_word)} "
                f"for '{pauli_word}'."
            )

        parsed_terms.append(
            coefficient * pauli_word_to_matrix(pauli_word)
        )

    if not parsed_terms:
        raise ValueError("No valid Pauli terms were found.")

    result = parsed_terms[0]

    for term in parsed_terms[1:]:
        result = result + term

    return result


def TFIM_to_matrix(num_spin: int, params: dict) -> Qobj:
    """
    params:
        Dictionary containing:
            params["J"] : nearest-neighbour interaction strength
            params["h"] : transverse-field strength
    """
    if num_spin < 1:
        raise ValueError("num_spin must be at least 1.")

    J = params.get("J", 1.0)
    h = params.get("h", 1.0)
    periodic = params.get("periodic", False)

    # Start from the zero operator with the correct dimensions.
    H = 0 * pauli_word_to_matrix("I" * num_spin)

    # Nearest-neighbour ZZ terms.
    for i in range(num_spin - 1):
        pauli_zzterm = ["I"] * num_spin
        pauli_zzterm[i] = "Z"
        pauli_zzterm[i + 1] = "Z"

        pauli_word = "".join(pauli_zzterm)
        H += -J * pauli_word_to_matrix(pauli_word)

    # Optional interaction between the last and first spins.
    if periodic and num_spin > 2:
        pauli_zzterm = ["I"] * num_spin
        pauli_zzterm[-1] = "Z"
        pauli_zzterm[0] = "Z"

        H += -J * pauli_word_to_matrix("".join(pauli_zzterm))

    # Transverse-field X terms.
    for i in range(num_spin):
        pauli_xterm = ["I"] * num_spin
        pauli_xterm[i] = "X"

        pauli_word = "".join(pauli_xterm)
        H += -h * pauli_word_to_matrix(pauli_word)

    return H



def XXZ_hx(num_spin: int, params: dict) -> Qobj:
    """
    params:
        Dictionary containing:
            params["Jxy"]      : in-plane (XX + YY) exchange coupling
            params["Delta"]    : ZZ anisotropy (ZZ coupling = Jxy * Delta).
                                 Delta == 1 -> isotropic Heisenberg,
                                 Delta == 0 -> XX model.
            params["hx"]       : uniform transverse field along x
            params["periodic"] : close the ring (needs num_spin > 2)
    """
    if num_spin < 1:
        raise ValueError("num_spin must be at least 1.")

    Jxy = params.get("Jxy", 1.0)
    delta = params.get("Delta", 1.0)
    hx = params.get("hx", 0.5)
    periodic = params.get("periodic", False)

    H = 0 * pauli_word_to_matrix("I" * num_spin)

    def _bond(i, j):
        term = 0 * pauli_word_to_matrix("I" * num_spin)
        for pauli, coeff in (("X", Jxy), ("Y", Jxy), ("Z", Jxy * delta)):
            letters = ["I"] * num_spin
            letters[i] = pauli
            letters[j] = pauli
            term += coeff * pauli_word_to_matrix("".join(letters))
        return term

    # Nearest-neighbour XXZ bonds.
    for i in range(num_spin - 1):
        H += _bond(i, i + 1)

    # Optional bond closing the ring.
    if periodic and num_spin > 2:
        H += _bond(num_spin - 1, 0)

    # Transverse-field X terms.
    for i in range(num_spin):
        pauli_xterm = ["I"] * num_spin
        pauli_xterm[i] = "X"
        H += hx * pauli_word_to_matrix("".join(pauli_xterm))

    return H


#################################################
# Quantum-algebra utilities
#################################################
def trace(state):
    """
    Trace of a density matrix / operator.

    Accepts a QuTiP Qobj or a numpy array. For a ket (or bra) this returns the
    squared norm <psi|psi>, i.e. the trace of the equivalent projector.
    """
    if isinstance(state, Qobj):
        if state.isket or state.isbra:
            return complex(state.norm() ** 2)   # tr(|psi><psi|) = <psi|psi>
        return complex(state.tr())

    arr = np.asarray(state)
    if arr.ndim == 1:
        return complex(np.vdot(arr, arr))
    return complex(np.trace(arr))


def _partial_trace_matrix(rho: np.ndarray, keep, dims) -> np.ndarray:
    """
    Numpy core of the partial trace: reduce ``rho`` onto the subsystems in
    ``keep`` given the per-subsystem dimensions ``dims`` (e.g. [2, 2, 2, 2]).

    Reshape into a rank-2N tensor and contract (via einsum) the bra/ket index
    pair of every subsystem that is not kept.
    """
    keep = sorted(int(k) for k in keep)
    dims = [int(d) for d in dims]
    num_sub = len(dims)

    row = list(range(num_sub))                 # bra index labels
    col = list(range(num_sub, 2 * num_sub))    # ket index labels
    for i in range(num_sub):
        if i not in keep:
            col[i] = row[i]                    # repeated label -> traced out

    tensor_rho = rho.reshape(dims + dims)
    out_labels = [row[i] for i in keep] + [col[i] for i in keep]
    reduced = np.einsum(tensor_rho, row + col, out_labels)

    kept_dim = int(np.prod([dims[i] for i in keep])) if keep else 1
    return reduced.reshape(kept_dim, kept_dim)


def partial_trace(state, keep, dims=None):
    """
    Reduced density matrix on the subsystems listed in ``keep``.

    Parameters
    ----------
    state : Qobj or ndarray
        A ket or a density matrix. Kets are promoted to |psi><psi| first.
    keep : int or sequence of int
        Subsystem index/indices to keep; everything else is traced out.
    dims : sequence of int, optional
        Per-subsystem dimensions. Inferred from a Qobj; required for a raw
        numpy array.

    Returns
    -------
    Qobj or ndarray
        Reduced density matrix, matching the input type.
    """
    if np.isscalar(keep):
        keep = [keep]

    if isinstance(state, Qobj):
        rho = state if state.isoper else state.proj()
        sub_dims = state.dims[0]
        reduced = _partial_trace_matrix(rho.full(), keep, sub_dims)
        kept_dims = [sub_dims[i] for i in sorted(keep)]
        return Qobj(reduced, dims=[kept_dims, kept_dims])

    rho = np.asarray(state)
    if rho.ndim == 1:
        rho = np.outer(rho, rho.conj())
    if dims is None:
        raise ValueError("`dims` is required when `state` is a numpy array.")
    return _partial_trace_matrix(rho, keep, dims)


def expectation(operator, state):
    """
    Expectation value <O> of ``operator`` in ``state``.

    For a ket this is <psi|O|psi>; for a density matrix it is tr(O rho).
    Accepts Qobj or ndarray for either argument. Returns a Python float when
    the result is real to numerical precision (Hermitian observables always
    land here), otherwise a complex.
    """
    O = operator.full() if isinstance(operator, Qobj) else np.asarray(operator)

    if isinstance(state, Qobj):
        if state.isket:
            psi = state.full()
            val = (psi.conj().T @ O @ psi).ravel()[0]
        elif state.isbra:
            psi = state.dag().full()
            val = (psi.conj().T @ O @ psi).ravel()[0]
        else:
            val = np.trace(O @ state.full())
    else:
        s = np.asarray(state)
        if s.ndim == 1:                              # flat state vector
            val = s.conj() @ O @ s
        elif s.ndim == 2 and 1 in s.shape:           # (N,1) or (1,N) ket/bra
            v = s.reshape(-1)
            val = v.conj() @ O @ v
        elif s.ndim == 2 and s.shape[0] == s.shape[1]:   # square density matrix
            val = np.trace(O @ s)
        else:
            raise ValueError(
                f"Cannot interpret state array of shape {s.shape} as a "
                "ket, bra, or density matrix."
            )

    val = complex(val)
    return val.real if abs(val.imag) < 1e-12 else val


def evolution_decomposed(H, t: float, sign: float = -1.0):
    r"""
    Eigendecompose the propagator exp(sign * i * H * t) without multiplying the
    factors back together.

    For a Hermitian H = V diag(lambda) V^dagger the exponential factors as

        exp(sign i H t) = V . diag(exp(sign i lambda t)) . V^dagger,

    so the returned middle matrix is diagonal and *already exponentiated*.

    Parameters
    ----------
    H : Qobj or ndarray
        Hermitian Hamiltonian.
    t : float
        Evolution time.
    sign : float
        Sign in the exponent. sign = -1 (default) gives the physical propagator
        e^{-iHt}; sign = +1 gives e^{+iHt}.

    Returns
    -------
    list of ndarray
        [V, expD, V_dag] where
            V      : unitary matrix of eigenvectors (columns),
            expD   : diagonal matrix diag(exp(sign i lambda t)),
            V_dag  : conjugate transpose of V.

    Reconstruct the full operator with ``recompose([V, expD, V_dag])`` or apply
    it to a vector x cheaply as ``V @ (expD @ (V_dag @ x))``.
    """
    M = H.full() if isinstance(H, Qobj) else np.asarray(H)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("H must be a square matrix.")

    eigvals, V = np.linalg.eigh(M)                  # H = V diag(eigvals) V^dag
    expD = np.diag(np.exp(sign * 1j * eigvals * t))
    V_dag = V.conj().T
    return [V, expD, V_dag]


def recompose(decomp) -> np.ndarray:
    """Multiply a [V, expD, V_dag] decomposition back into a single matrix."""
    V, expD, V_dag = decomp
    return V @ expD @ V_dag


def main():

    params = {
        "J": 1.0,
        "h": 0.5,
        "periodic": False,
    }

    H = TFIM_to_matrix(num_spin=7, params=params)

    print(H)
    print(H.shape)

    # --- XXZ + utilities smoke test -------------------------------------
    xxz_params = {"Jxy": 1.0, "Delta": 1.0, "hx": 0.5, "periodic": False}
    H_xxz = XXZ_hx(num_spin=4, params=xxz_params)
    print("XXZ Hermitian:", H_xxz.isherm, "shape:", H_xxz.shape)

    psi = tensor([basis(2, 0), basis(2, 1), basis(2, 0), basis(2, 1)])

    print("trace(|psi><psi|):", trace(psi.proj()))
    print("<Z0>:", expectation(pauli_word_to_matrix("ZIII"), psi))

    rho_pair = partial_trace(psi, keep=[0, 1])
    print("reduced (0,1) dims:", rho_pair.dims, "trace:", trace(rho_pair))

    V, expD, V_dag = evolution_decomposed(H_xxz, t=0.7)
    U_decomp = recompose([V, expD, V_dag])
    U_direct = (-1j * H_xxz * 0.7).expm().full()
    print("||decomposed - expm||:",
          float(np.linalg.norm(U_decomp - U_direct)))

    return 0


if __name__ == "__main__":
    main()