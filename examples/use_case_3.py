import numpy as np

from ulens_lsst_smc.simulatedevents import SimulatedEvents


isochrone_old_file = "../data/iso_1.0e10_0.004.dat"
isochrone_young_file = "../data/iso_2.0e8_0.004.dat"

model_old_file = "../data/model_old.pkl"
model_young_file = "../data/model_yng.pkl"

microlensing_file = "../data/model_fast.txt"

mu_rel_file = "../data/mu_rel.txt"

n_samples = 10000

out_file = "../data/simulated_events_001.json"

# settings end here

simulated = SimulatedEvents(
    microlensing_file,
    isochrone_old_file, isochrone_young_file,
    model_old_file, model_young_file, mu_rel_file)

simulated.n_samples = n_samples

simulated.generate_coords()
simulated.generate_fluxes()
simulated.generate_microlensing_parameters()
simulated.add_planets()

simulated.save(out_file)

