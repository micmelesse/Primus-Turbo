#!/bin/bash
export NNODES=2
export NODELIST=gpu-47,gpu-55
srun -N "${NNODES}" \
     --jobid=36667 \
     --exclusive \
     --ntasks-per-node=1 \
     -t 04:30:00 \
     --cpus-per-task="${CPUS_PER_TASK:-256}" \
     --nodelist="${NODELIST}" \
     bash -c "
            readarray -t node_array < <(scontrol show hostnames \"\$SLURM_JOB_NODELIST\")
                if [ \"\$SLURM_NODEID\" = \"0\" ]; then
                    echo \"========== Slurm cluster info ==========\"
                    echo \"SLURM_NODELIST: \${node_array[*]}\"
                    echo \"SLURM_NNODES: \${SLURM_NNODES}\"
                    echo \"SLURM_PROCID: \${SLURM_PROCID}\"
                    echo \"\"
                fi
            export MASTER_ADDR=\${node_array[0]}
            export MASTER_PORT=\${MASTER_PORT}
            export NNODES=\${SLURM_NNODES}
            podman exec primus_hz bash -c 'cd /workspace/Primus-Turbo && torchrun --nnodes '\${SLURM_NNODES}' --nproc-per-node 1 --node-rank='\${SLURM_PROCID}' --master-addr '\${MASTER_ADDR}' test_internode.py '

     "
