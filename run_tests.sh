#!/usr/bin/env bash

# Get options
usage="Usage: ${0} [-h | --help] [-c | --clean] [-i | --install] [-t | --test] \
                   [-f | --format]"

# Check if at least one character was passed
if [[ ${1} ]]; then
    :
else
    echo $usage
    exit 1
fi

# Get options
OPTS=$(getopt -o 'hcitf' --long 'help,clean,install,test,format' -n ${0} -- ${@})

# If any error in getopt, exit
if [ ${?} -ne 0 ]; then
    echo $usage
    exit 1
fi

# Pass options to $@
eval set -- ${OPTS}
unset OPTS

pass_args=""
while [[ ${1} ]]; do
    case ${1} in
    -h | --help)
        echo $usage
        exit 0
        ;;
    -t | --test)
        do_test=true
        shift
        ;;
    -i | --install)
        install=true
        pass_args+=" -i"
        shift
        ;;
    -c | --clean)
        pass_args+=" -c"
        clean=true
        shift
        ;;
    -f | --format)
        format=true
        shift
        ;;
    --)
        shift
        break
        ;;
    *)
        echo $usage
        exit 1
        ;;
    esac
done

# Create and activate the venv for the tests
# Pass option if given to the first script
source tests/create_activate_venv.sh ${pass_args}

if [[ $format ]]; then
    # Run the format checker
    flake8 --ignore=F401,E226,W503,W504 triage

    # Run the type checker
    mypy triage
fi

# Make sure previous script was successful
if [[ ${?} -ne 0 ]]; then
    echo "Problem with create_activate_venv, aborting"
    exit 1
fi

# Run the tests
if [[ $do_test ]]; then
    python3 tests/unit_tests.py
    python3 tests/integrated_tests.py
fi
