"""Target-specific conditioning of endothelial monolayers.

A monolayer-state model (orientation domains, junction destabilisation, junction damage, cell
retention) calibrated on published results from the same parallel-plate chamber, and a protocol
designer that turns the wall shear stress a device will see into a conditioning path.

Modules
-------
observations  published values (digitised or quoted) and the protocols that produced them
model         the monolayer-state model, integrated for many protocols at once
calibrate     least-squares calibration, the ensemble of accepted parameter sets, scenario
              ensembles and the profile of the switch shear
design        protocol design for a target shear, robust over the ensemble, reported as the
              simplest near-optimal path
device        one pump flow conditioning a device whose regions see different shears
closed_loop   receding-horizon design with imaging feedback against monolayers it does not know
_kernel       the model's time loop compiled with numba (checked against model.simulate)
run_all       reproduces every result used in the paper
"""
