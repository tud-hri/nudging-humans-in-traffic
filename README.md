# Cognitive-AV

Arkady Zgonnikov & Niek Beckers


Making automated vehicles more compatible with humans by endowing AVs with human-based cogntive decision-making algorithms.

Arkady's evidence accumulation model: [paper](https://psyarxiv.com/p8dxn/) and [model code](https://osf.io/x3ns6/).

## Run

As a first step we created an example scenario with an intersection, an ego vehicle turning left, and an oncoming AV (hardcoded actions for now).

Run: 
```python
python run.py
```

## Installation

You need a couple of python packages to run the code. To install them, we advise you to create a python environment first, for this specific project. If you don't want to, skip these instructions:

```python
python -m venv venv
``` 
You probably need to activate the environment (sometimes this happens automatically, check whether the command line in a terminal starts with `(venv)`):
```python
venv\Scripts\activate.bat  # windows
source venv/bin/activate   # unix/macos
```
Check if the command line starts with `(venv)`, which means that the virtual environment is activated.

Install the python packages:

```
pip install -r requirements.txt
```

### Optimization algorithms

We need a optimization algorithm for path planning of our agents (AV and simulated human), with the following requirements:

- open-source
- reasonable fast solver (ideally sub 100ms depending on problem complexity)
- usable for offline simulation and human-in-the-loop experiments (connected to solver speed)
- relatively simple implementation for fast prototyping (including built-in tools for differentiation of the objective function) 
- python interface (preferred, if not, CPython can be used)

We start with ACADOS, a new iteration of ACADO, which also uses CASADI for differentiation. It promises to be fast, specifically made for NMPC, has a python interface, actively maintained at the moment. Drawbacks are that it is still in development, documentation is limited compared to ACADO. 

__References__

- See [this thread](https://groups.google.com/g/casadi-users/c/Z_zu8hqTR3A?pli=1) for a good comparison between CASADI and ACADO/ACADOS.
- [__ACADOS__](https://github.com/acados/acados)
    + Fast embedded solver for NMPC
    + Interface with Python
    + [github](https://github.com/acados/acados), [paper](https://arxiv.org/abs/1910.13753)
- [__CASADI__](https://web.casadi.org/)
    + Possible to use IPOPT
    + More general optimization framework, therefore probably a bit slower. 
    + Used by ACADOS too for differentiation, among others. 
- [__ACADO__](https://acado.github.io/)
    + Previous version of ACADOS.
    + Reasonably fast.
    + C++ (we can use CPython to interface it with Python; see [link](http://grauonline.de/wordpress/?page_id=3244)). 
    + (Nonlinear) MPC. 
    
### CARLO
We are using [CARLO](https://github.com/Stanford-ILIAD/CARLO) as a simple 2D driving simulator. 

CARLO is added as a `git subtree`; to update it, run the following command:

```python
git subtree pull --prefix carlo https://github.com/Stanford-ILIAD/CARLO.git master --squash
```
