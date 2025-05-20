#!/usr/bin/env bash

# Create virtual environment for the test
testing_env=${HOME}/.local/python_envs/triage_testing_venv

# Check if venv already exists
if [[ ! -d ${testing_env} ]]; then
    python3 -m venv ${testing_env}
fi

# Activate the environment
source ${testing_env}/bin/activate
if [[ ${?} -ne 0 ]]; then
    echo "Failed to activate environment"
    exit 1
fi

# If given the -c option, uninstall packages
if [[ ${1} ]]; then
    case ${1} in
    -c)
        # Clean every package but triage
        clean=true
        ;;
    -i)
        install=true
        ;;
    esac
fi

if [[ $clean ]]; then
    python3 -m pip uninstall -yr <(python3 -m pip freeze | sed -nr '/triage/!p')
fi

# Install triage
if [[ $install ]]; then
    python3 -m pip install mypy
    python3 -m pip install flake8
    python3 -m pip install -e .
fi
