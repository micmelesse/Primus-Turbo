# Primus-Turbo


## Docker Image
```
rocm/megatron-lm:v25.5_py310
```

## Install
```
git clone https://github.com/AMD-AIG-AIMA/Primus-Turbo.git
git submodule update --init --recursive
```

### Develop
```
pip3 install -r requirements.txt
pip3 install -e .
pytest -s ./tests/
```

### Deploy
```
pip3 install -r requirements.txt
python3 -m build --wheel --no-isolation
pip3 install --extra-index-url https://test.pypi.org/simple ./primus_turbo-XXX.whl
```
