# Instructions to run

1. Modify `program_[suffix].py`. 
    - Modify the line `sys.path.insert(0, '/home/freeman/github/qchem')`.
    - Modify the line `xyz_namespace = '/home/freeman/github/qchem/examples/1/xyz'`.
    - CTRL+F for the comment "Make backend" and configure which backend is to be run on.
    - Modify the settings (see Tutorial notebook for reference).
    Note that we set `initial_layout` ie. the qubits that are going to be used. 
    Ideally, this should be `None` ie. we let transpiler decide, but Qiskit sometimes throws an error.
    Thus, try to set to `None` and see what happens.
2. `./run_[suffix].sh`.
