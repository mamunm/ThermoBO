# Author: Osman Mamun

"""
Data Compilation for Pfeiffer.
"""
import re
from pathlib import Path

import pandas as pd

files = sorted(Path("computational").glob("results_*.txt"))
data = []
PROPS = ["Density (g/cm3)", "Electric conductivity (S/m)", 
         "Heat capacity (J/(mol K))", "Thermal conductivity (W/(mK))",
         "Thermal diffusivity (m2/s)", "Linear thermal expansion (1/K)"]

for f in files:
    contents = f.read_text()
    temp = [i.split() for i in re.findall(r'[a-zA-z]{1,2}[\s]+[\d.]+', contents)]
    temp = {k: float(v) for k, v in temp}
    temp["temperature"] = float(re.search(r"At Temperature = ([\d]+)°C", contents).groups()[0])
    temp['origin'] = f.stem
    temp.update(dict(zip(PROPS, 
                         list(map(float, contents.split('\n')[-2].split())))))
    data.append(temp)

df = pd.DataFrame(data)
df.to_csv("computational_data.csv", index=False)