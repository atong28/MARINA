"""Emit the seed-replicate jobs for the dataset comparison.

Nine near-identical Jobs differing only in dataset, seed, and a couple of flags;
hand-writing them invites a copy-paste error in a checkpoint path or seed. Run:

    python nautilus/jobs/gen_dataset_comparison_seeds.py > nautilus/jobs/dataset-comparison-seeds.yaml
"""

GPUS = ["NVIDIA-TITAN-RTX", "NVIDIA-A40", "NVIDIA-GeForce-RTX-3090", "NVIDIA-RTX-A6000",
        "NVIDIA-RTX-A5000", "NVIDIA-A10", "NVIDIA-GeForce-RTX-4090"]
BAD_NODES = ["gpu-13.nrp.mghpcc.org", "hcc-chase-shor-c4705.unl.edu",
             "hcc-chase-shor-c4709.unl.edu", "hcc-chase-shor-c4715.unl.edu",
             "ry-gpu-08.sdsc.optiputer.net"]

# MARINA1 final-run teachers, one per seed. Each seed directory holds two timestamped
# runs; the later one is the valid re-run and is what the distillation sweep used.
TEACHER = {
    0: "/root/gurusmart/Moonshot/results/marina-final-run-seed-0/2026-05-07_23-56-46/epoch_epoch=738.ckpt",
    1: "/root/gurusmart/Moonshot/results/marina-final-run-seed-1/2026-05-07_23-56-57/epoch_epoch=615.ckpt",
    2: "/root/gurusmart/Moonshot/results/marina-final-run-seed-2/2026-05-07_23-58-54/epoch_epoch=689.ckpt",
}

TEMPLATE = """apiVersion: batch/v1
kind: Job
metadata:
  name: atong-{name}
  labels:
    owner: atong
spec:
  template:
    spec:
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
              - matchExpressions:
                  - key: nvidia.com/gpu.product
                    operator: In
                    values:
{gpus}
                  - key: kubernetes.io/hostname
                    operator: NotIn
                    values:
{bad_nodes}
      containers:
        - name: demo
          image: gitlab-registry.nrp-nautilus.io/a8tong/smart-moonshot/pixi-cuda:12.8.1
          command: ["/bin/bash"]
          args:
            - "-c"
            - >-
              set -euxo pipefail &&
              bash /root/gurusmart/startup.sh {zipfile} &&
              cd /code &&{checkout}
              pixi run torchrun --nproc_per_node 4 --module src.main marina --input_types {{hsqc,c_nmr,h_nmr,mw,mass_spec}} --seed {seed}{extra} --experiment_name {name}
          volumeMounts:
            - mountPath: "/root/gurusmart"
              name: atong-spectre
            - mountPath: "/root/datasets"
              name: smart-datasets
            - mountPath: /dev/shm
              name: dshm
            - mountPath: /workspace
              name: data-unzip
            - mountPath: /code
              name: code-workdir
          resources:
            limits:
              memory: 180Gi
              cpu: "8"
              ephemeral-storage: 600Gi
              nvidia.com/gpu: "4"
            requests:
              memory: 180Gi
              cpu: "8"
              ephemeral-storage: 600Gi
              nvidia.com/gpu: "4"
      volumes:
        - name: atong-spectre
          persistentVolumeClaim:
            claimName: atong-spectre
        - name: smart-datasets
          persistentVolumeClaim:
            claimName: smart-datasets
        - name: dshm
          emptyDir:
            medium: Memory
        - name: data-unzip
          emptyDir: {{}}
        - name: code-workdir
          emptyDir: {{}}
      restartPolicy: Never
      securityContext:
        runAsUser: 0
        runAsGroup: 0
        fsGroup: 0
  backoffLimit: 0
"""

CHECKOUT = "\n              git fetch origin &&\n              git checkout exp/marina234 &&"

jobs = []
for seed in (1, 2):
    jobs.append(dict(name=f"marina-m2-s{seed}", zipfile="MARINA2.zip", seed=seed,
                     checkout="", extra=""))
    jobs.append(dict(name=f"marina-m4-s{seed}", zipfile="MARINA4.zip", seed=seed,
                     checkout="", extra=""))
    jobs.append(dict(name=f"marina-drop0668-s{seed}", zipfile="MARINA1.zip", seed=seed,
                     checkout=CHECKOUT, extra=" --modality_drop_override 0.668"))

# All three finetune arms, seed N warm-started from final-run-seed-N so the three
# chains are independent end to end and comparable with MARINA2's three seeds.
for seed in (0, 1, 2):
    jobs.append(dict(
        name=f"marina-m1-m3-ft-s{seed}", zipfile="MARINA3.zip", seed=seed, checkout=CHECKOUT,
        extra=f" --lr 5e-5 --modality_drop_override 0.0 --load_from_checkpoint {TEACHER[seed]}"))

blocks = [TEMPLATE.format(
    gpus="\n".join(f"                    - {g}" for g in GPUS),
    bad_nodes="\n".join(f"                    - {n}" for n in BAD_NODES),
    **j) for j in jobs]
print("---\n".join(blocks), end="")
