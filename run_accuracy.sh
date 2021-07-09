#!/bin/bash

export DGL_CPU_INTEL_KERNEL_ENABLED=1

python graphsage_full_graph_short.py
python graphsage_mini_batch_short.py