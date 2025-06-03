# Tests

This repository contains a local CI testing environment that can be run with

    ./run_tests.sh <flag>

The possible flags are

```
-h, --help
    Prints usage information
-c, --clean
    Cleans the python virtual environment by uninstalling all packages
-i, --install
    Installs triage and other packages needed for testing
-t, --test
    Runs the tests
-f, --format
    Runs an auto-formatter, format checker and static typing checker
```

When the `-i` flag is passed, the script creates its own python environment in `~/.local/python_envs/triage_testing_venv`, installs mypy, flake8, autopep8 and triage in editable mode.
