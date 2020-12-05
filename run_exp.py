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
RESULTS_FILE="out/$${MODEL}$${DIMS}d-$${PREP}"
LRS=(1e-2 5e-3 1e-3)
MAX_GRADS=(5 50 250)
BATCH_SIZES=(128 512 2048)
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
            RUN_ID=r$$MODEL$$DIMS-$$PREP-lr$$LR-mgr$$MGN-bs$$BS-$$RUN
            python -m torch.distributed.launch --nproc_per_node=${py_nproc} --master_port=${py_port} train.py \\
                --n_procs=${py_nproc} \\
                --data=$$PREP \\
                --run_id=$$RUN_ID \\
                --model=$$MODEL \\
                --dims=$$DIMS \\
                --learning_rate=$$LR \\
                --val_every=25 \\
                --patience=50 \\
                --max_grad_norm=$$MGN \\
                --batch_size=$$BS \\
                --epochs=1500 \\
                --results_file=$$RESULTS_FILE \\
                --seed=$$SEED
        done
    done
done
"""
#> /hits/basement/nlp/lopezfo/out/sympa/runs/$${RUN_ID}

from string import Template
import itertools
import subprocess
import random


if __name__ == '__main__':
    template = Template(SCRIPT)

    partition = "cascade"
    nprocs = 10
    models = ["bounded", "upper"]
    dims = [2, 3]
    preps = ["grid3d-125", "grid4d-256"]
    runs = [1, 2]

    for i, (model, dim, prep, run) in enumerate(itertools.product(models, dims, preps, runs)):
        do_pull = 1 if i == 0 else 0
        port = random.randint(2048, 48000)  # TCP available ports

        vars = {"py_model": model, "py_dim": dim, "py_prep": prep, "py_nproc": nprocs,
                "py_run": run, "py_partition": partition, "py_do_pull": do_pull,
                "py_port": port}
        final_script = template.substitute(vars)

        file_name = "job_script.sh"
        with open(file_name, "w") as f:
            f.write(final_script)

        op_res = subprocess.run(["sbatch", file_name], capture_output=True, check=True)
        print(f"Job {i} vars: {vars}\nJob number: {op_res.stdout.decode()}")
    print("Done!")
