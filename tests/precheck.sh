echo "Format check with flake8"
flake8 triage/*.py

echo "Type check with mypy"
mypy triage/*.py
