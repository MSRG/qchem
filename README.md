# Install

```
python -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

For environment to work with Jupyter notebooks, we need to use `ipykernel`.
Make sure environment is activated first.
```
python -m ipykernel install --user --name=venv
```
To activate environment in Jupyter notebook, do `kernel > change kernel > env` after opening notebook.
