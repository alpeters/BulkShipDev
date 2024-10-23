# BulkShipDev

Processes raw AIS tracking data for bulk ships to obtain annual statistics, including input file for ML

### Python version
This code runs with python/3.8.10.

!!! This code may need to be updated in order to run distributed processing. Any updates should closely follow the code in ContainerShipDev. !!!

Use pyenv for managing python versions.
To set up python virtual environment ShipDist (SHIPping DISTributed processing) for version compatibility:

```console
username@machine:~$ pyenv virtualenv 3.8.10 ShipDist
username@machine:~$ pyenv activate ShipDist
username@machine:~$ pip install -r python_requirements.txt
username@machine:~$ pyenv deactivate
```

Each time you run code, make sure you are in the virtual environment:
```console
username@machine:~$ pyenv activate ShipDist
```
Be careful when running code from VS Code, as there are multiple ways to do so, and they do not necessarily use the same environment:

1. (Preferred) Run Python File in Integrated Terminal
    - Right-click on the Python file in the editor and select "Run Python File in Terminal", OR open the terminal (View > Terminal) and run script using `python your_script.py`
    - This method uses the Python interpreter selected in VS Code. Ensure the correct interpreter is selected by using Python: Select Interpreter from the Command Palette. The activated environment is displayed in parentheses before the prompt.
2. (May work) Jupyter Notebooks
    - Run cells using the 'Run Cell' and associated buttons at the top of each cell.
    - Select the kernel by clicking on the kernel name in the top right corner of the notebook.
3. (Don't use!) Interactive Window
    - Highlight code in the editor and press Shift+Enter or right-click and select "Run Selection/Line in Python Interactive Window".
    - As far as I can tell, you cannot control the environment and therefore this method should not be used!
