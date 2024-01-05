# Description
VQE with STC is implemented entirely in `main.py`.
Experiments can be performed with statevector, noisy simulation backends, and real devices.
For completeness, older versions of the codebase (not using Qiskit runtime) can be found in the "main" branch.

# Install
Ensure Python version is 3.10 (recommended to use pyenv).
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install qiskit-terra==0.22.4
git clone https://github.com/peelstnac/qiskit-ignis
cd qiskit-ignis
pip install -r requirements.txt
python setup.py install
cd ..
rm -rf qiskit-ignis
git clone https://github.com/peelstnac/mthree
cd mthree
pip install -r requirements.txt
python setup.py install
cd ..
rm -rf mthree
```

# Running

## With Qiskit Runtime
```
cd path/to/folder/to/store/logs
python -W ignore::DeprecationWarning main.py --runtime=True --token=your_token --channel=your_channel --settings=path/to/settings > log.txt 2>&1
```

## Locally
```
cd path/to/folder/to/store/logs
python -W ignore::DeprecationWarning main.py --settings=path/to/settings > log.txt 2&>1
```
