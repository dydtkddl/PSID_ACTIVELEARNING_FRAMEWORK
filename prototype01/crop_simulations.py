import pandas as pd
import os
def process_for_path(path):
    print(path)
    sim_name = path.split("/Output/System_0")[0]
    sim_splited = sim_name.split("_")
    cutoff = sim_splited[-1]
    pressure = float(sim_splited[-2].split("bar")[0])
    temp = float(sim_splited[-3].split("K")[0])
    gas =  sim_splited[-4]
    mof = "_".join(sim_splited[:-4])
    dic = {
        "name" : mof,
        "gas" : gas,
        "pressure" : pressure,
        "temp" : temp,
        "cutoff" : cutoff,
    }
    return dic

ls  = []
for path in [ x for x in os.listdir() if os.path.isdir(x)]:
    try:
        path += "/Output/System_0"
        with open(path + f"/{os.listdir(path)[0]}" , "r") as f:
            data = f.read()
            uptake_absolute = float(data.split("Average loading absolute [mol/kg framework]")[1].split(" +/-")[0].split()[0])
#        ref_data = choiced_df[choiced_df["Adsorbent"] == path.split("[CoreMOF]")[1].split("O2_298.15")[0]]["O2_20C_5bar"].values[0]
            dic = process_for_path(path)
            dic["path"] = path
            dic["Average loading absolute [mol/kg framework]"] = uptake_absolute
        # dic["ref_data"] = ref_data
            uptake_excess = float(data.split("Average loading excess [mol/kg framework]")[1].split(" +/-")[0].split()[0])
            dic["Average loading excess [mol/kg framework]"] = uptake_excess
            uptake_absolute_per_unitcell = float(data.split("Average loading absolute [molecules/unit cell]")[1].split(" +/-")[0].split()[0])
            dic["Average loading absolute [molecules/unit cell]"] = uptake_absolute_per_unitcell
            uptake_excess_per_unitcell = float(data.split("Average loading excess [molecules/unit cell]")[1].split(" +/-")[0].split()[0])
            dic["Average loading excess [molecules/unit cell]"] = uptake_excess_per_unitcell
        ls.append(dic)
    except:
        print(f"error : {path}")
pd.DataFrame(ls).to_csv("00_result.csv")


