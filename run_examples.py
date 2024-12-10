"""
Simple script to run the drivers

"""
from triage.generate_timeseries import driver_LinearSSM
from triage.generate_timeseries import driver_HiddenSemiMarkovModel
from triage.examples import initial_linearSSM

driver_LinearSSM()
driver_HiddenSemiMarkovModel()
initial_linearSSM.run_example()
