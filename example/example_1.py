import sys
sys.path.append("/home/ebnahaib/ThermoBO/thermo_bo")

from saasbo import SAASBO, SAASBOParameters

saasbo_parameters = SAASBOParameters(
    search_space={"Mg": [0.001, 10, 10],
                  "Zn": [0.001, 1.5, 10],
                  "Cu": [0.001, 2.5, 10],
                  "Mn": [0.001, 2.5, 10],
                  "Temp": [50.0, 100.0, 10]},
    seed_points=10,
    target="density",
    gp = "mc",
    target_mask=True,
    device="cpu",
    csv_path="density.csv")

saasbo = SAASBO(saasbo_parameters=saasbo_parameters)
print(saasbo.df)
#saasbo.save_df()
saasbo.run_optimization()

