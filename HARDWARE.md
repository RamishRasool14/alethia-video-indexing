# Hardware

The full 50-file run was done on a rented GPU: a **RunPod** pod (Secure Cloud,
data centre EUR-IS-2).

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4090, 24 GB (24,564 MiB), driver 595.112.07 |
| CPU | AMD EPYC 7452 32-Core Processor (host: 2 sockets, 128 threads). The pod is limited to 16 vCPU, with a cgroup CPU quota of 13.6 cores |
| RAM | 125 GB allocated to the pod (cgroup limit) |
| Disk | Container disk: 30 GB overlay on a local Samsung NVMe SSD (PM9A3, 7 TB). Data and code volume: RunPod network volume (MooseFS over the network, `/workspace`) |
| OS | Ubuntu 22.04.5 LTS (container), Linux kernel 6.8.0 |
| Python | 3.11.10 |

Full run of all 50 files (start to finish): _filled in after the final run_.
