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

RUN=$py_run
BRANCH="master"
MODEL="$py_model"
DIMS=$py_dim
PREP="$py_prep"
RESULTS_FILE="out/recosys/$${MODEL}$${DIMS}d-$${PREP}"
BATCH_SIZES=(2048 512 128)
LRS=(1e-2 2e-3)
MAX_GRADS=(10 50 250)
SEED=$$RANDOM

# if [[ $$(hostname -s) = pascal-* ]] || [[ $$(hostname -s) = skylake-* ]]; then
#     module load CUDA/9.2.88-GCC-7.3.0-2.30
# fi

# if [[ $$(hostname -s) = cascade-* ]]; then
#     module load CUDA/10.1.243-GCC-8.3.0
# fi

. /home/lopezfo/anaconda3/etc/profile.d/conda.sh 
conda deactivate
conda deactivate
conda activate sympa
cd /home/lopezfo/run-sympa/

# if [ "$py_do_pull" == "1" ]; then
#     git co -- .
#     git pull
#     git co $$BRANCH
#     git pull
# fi

for BS in $${BATCH_SIZES[@]}; 
    do
    for LR in $${LRS[@]}; 
    do
        for MGN in $${MAX_GRADS[@]}; 
        do
            MYPORT=`shuf -i 2049-48000 -n 1`
            RUN_ID=r$$MODEL$$DIMS-$$PREP-lr$$LR-mgr$$MGN-bs$$BS-$$RUN
            python -m torch.distributed.launch --nproc_per_node=${py_nproc} --master_port=$$MYPORT train.py \\
                --n_procs=${py_nproc} \\
                --prep=$$PREP \\
                --run_id=$$RUN_ID \\
                --model=$$MODEL \\
                --dims=$$DIMS \\
                --train_bias=1 \\
                --learning_rate=$$LR \\
                --val_every=50 \\
                --patience=50 \\
                --max_grad_norm=$$MGN \\
                --batch_size=$$BS \\
                --epochs=1000 \\
                --results_file=$$RESULTS_FILE \\
                --job_id=$$SLURM_JOB_ID \\
                --seed=$$SEED > /hits/basement/nlp/lopezfo/out/sympa/recosys/runs/$${RUN_ID}
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
    nprocs = 10
    models = ["bounded", "upper", "bounded-fone", "upper-fone"]
    dims = [3]
    preps = ["software"]
    runs = [1]
    timestamp = str(datetime.now().strftime("%Y%m%d%H%M%S"))

    for i, (model, dim, prep, run) in enumerate(itertools.product(models, dims, preps, runs)):
        do_pull = 1 if i == 0 else 0

        vars = {"py_model": model, "py_dim": dim, "py_prep": prep, "py_nproc": nprocs,
                "py_run": run, "py_partition": partition, "py_do_pull": do_pull}
        final_script = template.substitute(vars)

        file_name = "job_script.sh"
        with open(file_name, "w") as f:
            f.write(final_script)

        op_res = subprocess.run(["sbatch", file_name], capture_output=True, check=True)
        print(f"{timestamp} - Job {i} vars: {vars} PID: {op_res.stdout.decode()}")
    print("Done!")
