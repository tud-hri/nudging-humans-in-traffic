# Cognitive-AV

Arkady Zgonnikov & Niek Beckers


Making automated vehicles more compatible with humans by endowing AVs with human-based cogntive decision-making algorithms


## Installation

You need a couple of python packages to run the code. To install them, we advise you to create a python environment first, for this specific project. If you don't want to, skip these instructions:

```python
python -m venv venv
``` 
You probably need to activate the environment (sometimes this happens automatically):
```python
venv\Scripts\activate.bat  # windows
source venv/bin/activate   # unix/macos
```

Install the python packages:

```
pip install -r requirements.txt
```


### CARLO
We are using [CARLO](https://github.com/Stanford-ILIAD/CARLO) as a simple 2D driving simulator. 

CARLO is added as a `git subtree`; to update it, run the following command:

```python
git subtree pull --prefix carlo https://github.com/Stanford-ILIAD/CARLO.git master --squash
```