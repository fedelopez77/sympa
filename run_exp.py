SCRIPT = """#!/bin/bash

#SBATCH --job-name=sympa
#SBATCH --output=/hits/basement/nlp/lopezfo/out/sympa/job-out/out-%j
#SBATCH --error=/hits/basement/nlp/lopezfo/out/sympa/job-out/err-%j
#SBATCH --time=1-23:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=${py_partition}-deep.p
#SBATCH --nodelist=${py_partition}-deep-[0${py_instance}]

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
LRS=(1e-2)
MAX_GRADS=(500)
BATCH_SIZES=(2048 1024)


if [[ $$(hostname -s) = pascal-* ]] || [[ $$(hostname -s) = skylake-* ]]; then
    module load CUDA/9.2.88-GCC-7.3.0-2.30
fi

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
            RUN_ID=r$$MODEL$$DIMS-$$PREP-lr$$LR-mgr$$MGN-bs$$BS-$$RUN
            python train.py \\
                --data=$$PREP \\
                --run_id=$$RUN_ID \\
                --model=$$MODEL \\
                --dims=$$DIMS \\
                --learning_rate=$$LR \\
                --val_every=5 \\
                --patience=25 \\
                --max_grad_norm=$$MGN \\
                --batch_size=$$BS \\
                --epochs=1500 \\
                --results_file=$$RESULTS_FILE \\
                --seed=-1 > /hits/basement/nlp/lopezfo/out/sympa/runs/$${RUN_ID}
        done
    done
done
"""

from string import Template
import itertools
import subprocess

INSTANCES = {"skylake": 8, "pascal": 4}


if __name__ == '__main__':
    template = Template(SCRIPT)

    partition = "skylake"
    models = ["euclidean", "poincare"]
    dims = [6, 12]
    preps = ["tree3-3", "grid3d-64"]
    runs = [1, 2]

    for i, (model, dim, prep, run) in enumerate(itertools.product(models, dims, preps, runs)):
        instance = (i % INSTANCES[partition]) + 1

        vars = {"py_model": model, "py_dim": dim, "py_prep": prep,
                "py_run": run, "py_partition": partition, "py_instance": instance}
        final_script = template.substitute(vars)

        file_name = "job_script.sh"
        with open(file_name, "w") as f:
            f.write(final_script)

        op_res = subprocess.run(["sbatch", file_name], capture_output=True, check=True)
        print(f"Job {i} vars: {vars}\nJob number: {op_res.stdout.decode()}")
    print("Done!")
