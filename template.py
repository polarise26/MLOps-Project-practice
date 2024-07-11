import logging
import os

from pathlib import Path

logging.basicConfig(level = logging.INFO, format = "[%(asctime)s]: %(message)s:")

proj_name = "MLOps-Proj"

list_of_files = [".github/workflows/.gitkeep",
                 f"src/{proj_name}/init.py",
                 f"src/{proj_name}/components/init.py",
                 f"src/{proj_name}/utils/init.py",
                 f"src/{proj_name}/utils/common.py",
                 f"src/{proj_name}/config/init.py",
                 f"src/{proj_name}/config/configuration.py",
                 f"src/{proj_name}/pipeline/init.py",
                 f"src/{proj_name}/entity/init.py",
                 f"src/{proj_name}/entity/entity_config.py",
                 f"src/{proj_name}/constants/init.py",
                 "config/config.yaml",
                 "params.yaml",
                 "main.py",
                 "app.py",
                 "Dockerfile",
                 "requirements.txt",
                 "setup.py",
                 "trials/trial.ipynb",
                 "templates/index.html"]

for fpath in list_of_files:
    fpath = Path(fpath)
    fdir, fname = os.path.split(fpath)
    
    if fdir != "":
        os.makedirs(fdir, exist_ok = True)
        logging.info(f"Creating directory: {fdir} for file {fname}.")
        
    if (not os.path.exists(fpath)) or (os.path.getsize(fpath) == 0):
        with open(fpath, "w") as f:
            pass
            logging.info(f"Creating empty file: {fpath}.")
            
    else:
        logging.info(f"{fname} already exists!")