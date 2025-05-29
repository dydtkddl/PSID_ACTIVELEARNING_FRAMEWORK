import shutil 
import argparse  
import os 
from datetime import datetime
import json 
import pandas as pd 
# positional arguments
def main():
    parser = argparse.ArgumentParser(description = "Positionnal arguments")
    parser.add_argument("target_gas", type = str , help = "target gas name")
    parser.add_argument("target_temp", type= float , help = "target temperature")
    parser.add_argument("config" ,  type = str , default = "config.json" , help = "configfile")
    args = parser.parse_args() 

    print("target gas : %s"%(args.target_gas))
    print("target temp : %s"%args.target_temp)
    target_gas = args.target_gas
    target_temp = args.target_temp
    config      = args.config
    
    python_file_path = os.path.abspath(__file__)
    python_file_dir_path = os.path.dirname(python_file_path)
    print(python_file_dir_path)
    with open(python_file_dir_path + "/" + config , 'r') as f:
        config = json.load(f)
    mof_database_relative_path = config["mof_database_path"]
    mof_database_abs_path = os.path.join(python_file_dir_path, mof_database_relative_path)
    mof_database_df = pd.read_csv(mof_database_abs_path)
    kinetic_diameter = None
    KINETIC_DIAMETERS = config["kinetic_diameters"]
    for record in KINETIC_DIAMETERS:
        if record["formula"] == target_gas or record["name"] == target_gas:
            kinetic_diameter = record["kinetic_diameter_A"]
            print("target gas kinetic diameter is %s"%kinetic_diameter)
    if kinetic_diameter == None :
        raise ValueError(f"target_gas '{target_gas}' not found in kinetic_diameters list.")
    screened_mofs = mof_database_df[ mof_database_df["PLD"] > kinetic_diameter].reset_index(drop = True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    DIR_NAME = f"Isotherm_{target_gas}_{target_temp}K_{timestamp}"
    print(os.path.abspath(DIR_NAME))
    os.makedirs(DIR_NAME, exist_ok = True )
    print("Folder generated : %s"%DIR_NAME)
    
    screened_mofs.to_csv(DIR_NAME + "/target_mofs.csv")

    main_config = config["main_config"]
    main_config["main"]["gas"] = target_gas 
    main_config["main"]["temperature"] = target_temp
    main_config["main"]["target_mofs_csv"] = os.path.abspath(DIR_NAME + "/target_mofs.csv")
    
    main_config["GCMC"]["GAS"] = target_gas 
    main_config["GCMC"]["ExternalTemperature"] = target_temp

    with open(DIR_NAME+"/config.json" , 'w') as f:
        json.dump(main_config,f,indent=2)

if __name__ == "__main__" : 
    main()
