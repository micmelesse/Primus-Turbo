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
                    echo \"SLURM_GPUS_ON_NODE: \${SLURM_GPUS_ON_NODE}\"
                    echo \"\"
                fi
            export MASTER_ADDR=\${node_array[0]}
            export MASTER_PORT=\${MASTER_PORT}
            export NNODES=\${SLURM_NNODES}
            export NODE_RANK=\${SLURM_PROCID}
            export GPUS_PER_NODE=\${SLURM_GPUS_ON_NODE}
            export HSA_NO_SCRATCH_RECLAIM=\${HSA_NO_SCRATCH_RECLAIM}
            export NVTE_CK_USES_BWD_V3=\${NVTE_CK_USES_BWD_V3}
            export NCCL_IB_HCA=\${NCCL_IB_HCA}
            export GLOO_SOCKET_IFNAME=\${GLOO_SOCKET_IFNAME}
            export NCCL_SOCKET_IFNAME=\${NCCL_SOCKET_IFNAME}
            export REBUILD_BNXT=\${REBUILD_BNXT}
            export PATH_TO_BNXT_TAR_PACKAGE=\${PATH_TO_BNXT_TAR_PACKAGE}
            podman exec primus_hz bash -c 'cd Primus-Turbo && bash compile.sh'

     "
