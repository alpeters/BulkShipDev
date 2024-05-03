# ShippingEmissions

Repo for collaborative work with shipping emissions data

### Python version
This code runs with python/3.8.10.
To set up python virtual environment for version compatibility:

```console
username@machine:~$ virtualenv --no-download env
username@machine:~$ source env/bin/activate
username@machine:~$ pip install --no-index --upgrade pip
username@machine:~$ pip install -r python_requirements.txt
username@machine:~$ deactivate
```

Each time you run code, make sure you are in the virtual environment:
```console
username@machine:~$ source env/bin/activate
```

Note that I've only included python packages for the first data processing parts.
We will need to add more packages as we go.