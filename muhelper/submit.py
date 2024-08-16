
import argparse
import glob
import os
import sys
from pathlib import Path
import importlib

def insert (source_str, insert_str, pos):
    return source_str[:pos]+insert_str+source_str[pos:]

def submit_script(script, series, blockidx, hours, log_dir, cores=1, job_name="reco", system="lsf", slurm_account="", slurm_partition="", debug=True, verbose=False):
    if system=="lsf":
        #bsub_command = f"bsub -W {hours}:00 -o {log_dir}/{series}_{blockidx}.out -e {log_dir}/{series}_{blockidx}.err"
        #full_command = f"{bsub_command} {script}"
        
        bsub_command = \
        f"""#BSUB -L /bin/bash
#BSUB -n 1
#BSUB -e {log_dir}/{series}_{blockidx}.err
#BSUB -o {log_dir}/{series}_{blockidx}.out
source /cvmfs/cdms.opensciencegrid.org/setup_cdms.sh V04-03
{script}"""

        script_path = f"{log_dir}/{series}_{blockidx}.bat"
        with open(script_path, 'w') as file:
            file.write(bsub_command)
        full_command = f"bsub -W {hours}:00 -R 'select[centos7]' -m 'deft[0001-0023]' <{script_path}" 

        if debug:
            print("    (DEBUG) LSF script:", bsub_command)
            print("    (DEBUG) LSF command:", full_command)


    elif system=="slurm":
        slurm_command = \
        f"""#!/bin/bash
#SBATCH --job-name={job_name}_{blockidx}
#SBATCH --output={log_dir}/{series}_{blockidx}.out
#SBATCH --error={log_dir}/{series}_{blockidx}.err
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cores}
#SBATCH --mem-per-cpu=2048M
#
#SBATCH --time={hours}:00:00

{script}"""
        
        if len(slurm_account)!=0:
            slurm_command = insert(slurm_command,f"#SBATCH --account={slurm_account}\n",slurm_command.find('#SBATCH --job-name'))
        if len(slurm_partition)!=0:
            slurm_command = insert(slurm_command,f"#SBATCH --partition={slurm_partition}\n",slurm_command.find('#SBATCH --job-name'))            
            
        script_path = f"{log_dir}/{series}_{blockidx}.sh"
        with open(script_path, 'w') as file:
            file.write(slurm_command)
        full_command = f"sbatch {script_path}"  
        
        if verbose:
            print("    (DEBUG) SLURM script:", slurm_command)
        if debug:
            print("    (DEBUG) SLURM command:", full_command)
    
    if not debug:
        os.system(full_command)
