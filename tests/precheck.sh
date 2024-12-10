echo "Format check with flake8"
flake8 triage

echo "Type check with mypy"
mypy triage
