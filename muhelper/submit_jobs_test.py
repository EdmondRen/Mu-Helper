#!/usr/bin/env python3

from submit import *

from pathlib import Path

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--series', nargs='+', help='Series to process')
    parser.add_argument('--debug', action='store_true', help='Only show the commands without actually submitting jobs')
    parser.add_argument('--verbose', action='store_true', help='Show the slurm script content')
    parser.add_argument('--hours', type=int, default=6, help='Requested time in hours per job')
    parser.add_argument('--system', type=str, default='slurm', help='batch system, {slurm, lsf}')
    parser.add_argument('--slurm_account', type=str, default='rrg-mdiamond', help='Slurm system account')
    parser.add_argument('--slurm_partition', type=str, default='', help='Slurm system partition')
    
    args = parser.parse_args()
    Series = args.series #["25201106_180710"]#25201106_174906
    DEBUG     = args.debug
    verbose   =args.verbose
    hours = args.hours
    system = args.system
    slurm_account = args.slurm_account
    slurm_partition = args.slurm_partition    
    
    
    
    Log_Dir = '/project/def-mdiamond/tomren/mathusla/data/fit_study_6layer/log/'
    DataDir = '/project/def-mdiamond/tomren/mathusla/data/fit_study_6layer/'

    simulation='/project/def-mdiamond/tomren/mathusla/Mu-Simulation/simulation '
    tracker='/project/def-mdiamond/tomren/mathusla/MATHUSLA-Kalman-Algorithm/tracker/build/tracker '
    tracker2="/home/tomren/mathusla/MATHUSLA-Kalman-Algorithm_debug/tracker/build/tracker"
    tracker3="/home/tomren/mathusla/MATHUSLA-Kalman-Algorithm_vertexinitial/tracker/build/tracker"
    
    # Get path to par_card.txt
    path = Path(tracker2)
    par_card_path = str(path.parent.parent.absolute())+"/run/par_card.txt"    
     

    Scripts = ['muon_gun_tom_cms.mac']
    Names = ["muon"]
    CORES = 1
    

    # make directory for log
    os.system(f"mkdir -p {Log_Dir}")
    hours_mod = hours
    
    for ijob in range(len(Scripts)):
        sim_script = Scripts[ijob]
        script_name = sim_script.split(".")[0]
        data_dir_sub = f"{DataDir}/{script_name}/"
        # if ijob==0:
        #     data_dir_sub = f"{DataDir}/XtoMuMu_M4GeV_P40GeV/"
        # elif ijob==1:
        #     data_dir_sub = f"{DataDir}/XtoMuMu_P10GeV_manual/"
        # elif ijob==2:
        # # if ijob==2:
        #     data_dir_sub = f"{DataDir}/XtoMuMu_P10GeV_manual_1m/"            
        # else:
        #     continue



        job_script=f"""mkdir -p {data_dir_sub} 
{simulation} -j1 -q  -o {data_dir_sub}  -s {sim_script}  
for f in {data_dir_sub}/*/*/run*.root; do 
    {tracker2} $f `dirname $f` 
    mv `dirname $f`/stat0.root `dirname $f`/stat_vertexmod.root -f
      cp {par_card_path} `dirname $f`/par_card_stat_vertexmod.txt -f
done 
"""

#         job_script=f"""mkdir -p {data_dir_sub} 
# for f in {data_dir_sub}/*/*/run*.root; do 
#     {tracker2} $f `dirname $f` 
#     mv `dirname $f`/stat0.root `dirname $f`/stat_vertexmod.root -f    
#     cp {par_card_path} `dirname $f`/par_card_stat_vertexmod.txt -f
# done 
# """

        # This is the core function to submit job script
        script_prefix=sim_script.split("_")[0]
        submit_script(job_script, f"{script_name}_{ijob}", blockidx=0, hours=hours_mod, cores=CORES, log_dir=Log_Dir, job_name="reco", system=system, slurm_account=slurm_account, slurm_partition='', debug=DEBUG, verbose=verbose)
    
    
            
if __name__ == "__main__":
    main()            
