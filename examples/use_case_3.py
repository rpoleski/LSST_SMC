import numpy as np

from ulens_lsst_smc.simulatedevents import SimulatedEvents


isochrone_old_file = "../data/iso_1.0e10_0.004.dat"
isochrone_young_file = "../data/iso_2.0e8_0.004.dat"

model_old_file = "../data/model_old.pkl"
model_young_file = "../data/model_yng.pkl"

microlensing_file = "../data/model_fast.txt"

out_file = "../data/simulated_events_001.json"

n_samples = 10000

# settings end here

# What we need:
#  RA, Dec
#  source flux
#  lens flux
#  other blending fluxes
#  t_E
#  rho
#  t_0, u_0
#  add planet: s, q, alpha

simulated = SimulatedEvents(
    microlensing_file,
    isochrone_old_file, isochrone_young_file,
    model_old_file, model_young_file)

simulated.n_samples = n_samples

simulated.generate_coords()
simulated.generate_fluxes()
simulated.generate_microlensing_parameters()
simulated.add_planets()

simulated.save(out_file)
