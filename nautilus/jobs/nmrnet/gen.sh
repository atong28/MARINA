#!/bin/bash
# Generate MARINA-DB-NMRNET-MERGED (v3) flagship-recipe Jobs (Arm A plain / Arm B typejit) for Nautilus.
# Usage: ARM=uniqmult SEEDS="0 1 2" STAMP=$(date -u +%Y%m%d-%H%M) [NGPU=1 ACCUM=2 CPU_REQ=.. MEM_REQ=..] bash gen.sh > jobs.yaml
# Global batch = 256 x NGPU x ACCUM (512 for NGPU=2 ACCUM=1 or NGPU=1 ACCUM=2).
set -euo pipefail
ARM=${ARM:-uniqmult}; SEEDS=${SEEDS:-"0 1 2"}; STAMP=${STAMP:?}; COMMIT=${COMMIT:-6e8bafbee6ae549ed6cc9855a7625b34423d07ab}
NGPU=${NGPU:-1}; ACCUM=${ACCUM:-2}
CPU_REQ=${CPU_REQ:-8}; CPU_LIM=${CPU_LIM:-9}; MEM_REQ=${MEM_REQ:-48Gi}; MEM_LIM=${MEM_LIM:-56Gi}; EXTRA=${EXTRA:-}
for s in $SEEDS; do cat <<YAML
---
# MARINA-DB-NMRNET-MERGED (v3, zip 0ae421b4) flagship recipe, arm ${ARM}, seed ${s}. MARINA @ ${COMMIT}.
# ${NGPU} GPU(s) x bs256 x accum${ACCUM} = global 512, bf16, --resume (ckpts on the PVC, Moonshot/results/).
apiVersion: batch/v1
kind: Job
metadata:
  name: atong-mdbnmrnet-${ARM}-s${s}-${STAMP}
  namespace: guru-research
  labels: { owner: atong, campaign: marina-db-nmrnet }
spec:
  backoffLimit: 6
  activeDeadlineSeconds: 864000
  ttlSecondsAfterFinished: 172800
  template:
    spec:
      restartPolicy: Never
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
              - matchExpressions:
                  - key: nvidia.com/gpu.product
                    operator: In
                    values: [NVIDIA-GeForce-RTX-3090, NVIDIA-GeForce-RTX-4090, NVIDIA-A10, NVIDIA-RTX-A5000, NVIDIA-RTX-A6000, NVIDIA-A40, NVIDIA-L40, NVIDIA-L40S, NVIDIA-RTX-5000-Ada-Generation, NVIDIA-A100-PCIE-40GB, NVIDIA-A100-80GB-PCIe, NVIDIA-A100-SXM4-80GB]
                  - key: kubernetes.io/hostname
                    operator: NotIn
                    values: [nautilus-ext-gpu01.fullerton.edu, hcc-chase-shor-c4705.unl.edu, k8s-3090-01.usd.edu, hcc-nrp-shor-c6017.unl.edu]
      containers:
        - name: marina
          image: gitlab-registry.nrp-nautilus.io/a8tong/smart-moonshot/pixi-cuda:12.8.1
          env:
            - { name: COMMIT, value: "${COMMIT}" }
            - { name: ZIP, value: "/root/gurusmart/marina-db-nmrnet/v3-merged/MARINA-DB-NMRNET-MERGED.zip" }
            - { name: ZIP_SHA, value: "0ae421b4be9cf90a1981a3f7d23fe35ad7894683ae00ed4a152f3409240091ef" }
            - { name: SMART_RUN_ID, value: "mdbnmrnet-${ARM}-s${s}-${STAMP}" }
            - { name: PIXI_CACHE_DIR, value: /code/.pixi-cache }
          command: ["bash", "-c"]
          args:
            - >-
              set -euo pipefail;
              mkdir -p ~/.ssh; cp /root/gurusmart/.ssh/id_rsa ~/.ssh/id_rsa; chmod 600 ~/.ssh/id_rsa;
              export GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no";
              git clone -q git@github.com:atong28/MARINA.git /code; cd /code; git checkout -q "\$COMMIT";
              git log -1 --format='[bootstrap] MARINA @ %h %s'; nvidia-smi --query-gpu=index,name,memory.total --format=csv;
              cp "\$ZIP" /workspace/ds.zip; echo "\$ZIP_SHA  /workspace/ds.zip" | sha256sum -c;
              unzip -q /workspace/ds.zip -d /workspace; rm /workspace/ds.zip;
              pixi install; pixi run python scripts/dataset/pack_arrow.py --root /workspace;
              pixi run torchrun --standalone --nproc_per_node ${NGPU} --module src.main marina
              --experiment_name marina-db-nmrnet-${ARM}-s${s} --fp_type RankingEntropyUniqueMultiplicity --seed ${s}
              --batch_size 256 --accumulate_grad_batches_num ${ACCUM} --precision bf16-mixed
              --input_types hsqc c_nmr h_nmr mass_spec mass_spec_neg mw formula --resume ${EXTRA};
              echo "ALL DONE";
          volumeMounts:
            - { mountPath: /root/gurusmart, name: atong-spectre }
            - { mountPath: /dev/shm, name: dshm }
            - { mountPath: /code, name: code-workdir }
            - { mountPath: /workspace, name: data-local }
          resources:
            requests: { cpu: "${CPU_REQ}", memory: ${MEM_REQ}, nvidia.com/gpu: "${NGPU}", ephemeral-storage: 60Gi }
            limits:   { cpu: "${CPU_LIM}", memory: ${MEM_LIM}, nvidia.com/gpu: "${NGPU}", ephemeral-storage: 80Gi }
      volumes:
        - { name: atong-spectre, persistentVolumeClaim: { claimName: atong-spectre } }
        - { name: dshm, emptyDir: { medium: Memory, sizeLimit: 16Gi } }
        - { name: code-workdir, emptyDir: {} }
        - { name: data-local, emptyDir: {} }
YAML
done
