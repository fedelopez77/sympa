SCRIPT = """#!/bin/bash

#SBATCH --job-name=$py_model
#SBATCH --output=/hits/basement/nlp/lopezfo/out/sympa/job-out/out-%j
#SBATCH --error=/hits/basement/nlp/lopezfo/out/sympa/job-out/err-%j
#SBATCH --time=23:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${py_nproc}
#SBATCH --partition=${py_partition}.p

# "SBATCH" --nodelist=skylake-deep-[01]
# "SBATCH" --gres=gpu:1
# "SBATCH" --partition=skylake-deep.p,pascal-deep.p,pascal-crunch.p
# "SBATCH" --exclude=skylake-deep-[01],pascal-crunch-01
# "SBATCH" --nodelist=pascal-deep-[01] to fix the node that I am requesting
# "SBATCH" --mem=250G real memory required per node WARN: exclusive with mem-per-cpu

MODEL="$py_model"
RUN=$py_run
RESULTS_FILE="out/node/$${MODEL}"
BATCH_SIZES=(16 32 64 128)
LRS=(1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2)
MAX_GRADS=(1 10)
LAYERS=(1 2 3)

. /home/lopezfo/anaconda3/etc/profile.d/conda.sh 
conda deactivate
conda deactivate
conda activate sympa
cd /home/lopezfo/run-sympa/

for BS in $${BATCH_SIZES[@]}; 
    do
    for LR in $${LRS[@]}; 
    do
        for MGN in $${MAX_GRADS[@]}; 
        do
            for LAY in $${LAYERS[@]}; 
            do
                RUN_ID=r$$MODEL-lr$$LR-mgr$$MGN-bs$$BS-lay$$LAY-$$RUN
                python nodecls.py \\
                    --run_id=$$RUN_ID \\
                    --load_model=$$MODEL \\
                    --learning_rate=$$LR \\
                    --max_grad_norm=$$MGN \\
                    --batch_size=$$BS \\
                    --layers=$$LAY \\
                    --epochs=1000 \\
                    --val_every=25 \\
                    --results_file=$$RESULTS_FILE \\
                    --seed=$$RANDOM
            done
        done
    done
done
"""

from string import Template
import itertools
import subprocess
from datetime import datetime


if __name__ == '__main__':
    template = Template(SCRIPT)

    partition = "cascade"
    nprocs = 1
    models = ["ckpt/rprod-hyhy6-glass-lr1e-3-mgr10-bs512-1-best-1000ep"]
    runs = [1]
    timestamp = str(datetime.now().strftime("%Y%m%d%H%M%S"))

    for i, (model, run) in enumerate(itertools.product(models, runs)):
        do_pull = 1 if i == 0 else 0

        vars = {"py_model": model, "py_nproc": nprocs, "py_run": run, "py_partition": partition}
        final_script = template.substitute(vars)

        file_name = "job_script.sh"
        with open(file_name, "w") as f:
            f.write(final_script)

        op_res = subprocess.run(["sbatch", file_name], capture_output=True, check=True)
        print(f"{timestamp} - Job {i} vars: {vars} PID: {op_res.stdout.decode()}")
    print("Done!")
