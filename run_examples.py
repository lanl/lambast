"""
Simple script to run the drivers

"""
from triage.examples import copulas, initial_linearSSM
from triage.generate_timeseries import driver_HSMM, driver_LinearSSM

driver_LinearSSM()
driver_HSMM()
initial_linearSSM.run_example()
copulas.run_example()
