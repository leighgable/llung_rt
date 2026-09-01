#### Some experiments in runtimes, distillation, power Sequential Monte Carlo

##### Installation:
 If you have nix, you can just clone this repo, type `nix develop` and you will be in a developer environment with rust nightly, python 3.14, pytorch (for AMD cards see the flake.nix file) and everything you need.

 Otherwise, you will need rust nightly for all the llung-rt code in src. For the python branch, you will need python 3.14 and the UV package manager. You should be able to `cd python && uv init` and get a virtual environment with pytorch, etc.

 llung_rt/src/continual_ops.rs - tract for an onnx runtime that targets cpu/gpu
     for fast inference on resource-constrained devices with some continual
     learning capabilities.

 python/src/llung_rt/lora - LoRA adapters trained with continual backprop

 python/src/llung_rt/particle_distil - self distillation experiments

 python/src/llung_rt/rl - "streaming" reinforcement learning with a replay buffer

