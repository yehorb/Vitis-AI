# Vitis AI Native GPU Installation Guide

This guide explains how to recreate the GPU-enabled PyTorch and TensorFlow 2 environments that the Vitis AI Docker images provide, but directly on an Ubuntu host. It is written so you can implement the installer scripts from scratch without referencing the existing source.

## General Flow

- **Host prerequisites**
  - Ubuntu 20.04 or 22.04 with recent NVIDIA driver (`nvidia-smi` works).
  - Local user has `sudo` privileges and network access to AMD/Xilinx package mirrors or internal mirrors.
- **Installation sequence**
  1. Install host build/run dependencies (matching the Docker base layer, minus container-only extras).
  2. Install Miniforge (Conda) and `mamba` into a dedicated Vitis AI prefix.
  3. Make the Vitis AI conda channel available (tarball unpack or direct URL).
  4. Create the GPU PyTorch environment and layer the required pip packages.
  5. Create the GPU TensorFlow 2 environment and layer the required pip packages.
  6. Export the compiler architecture descriptors (`vaic/arch`) into the shared Vitis AI directory.
  7. (Optional) Install XRT/XRM and Vitis AI runtime `.deb` packages if you need local DPU runtime support.
- **Shell integration**
  - Source `"$VAI_ROOT/conda/etc/profile.d/conda.sh"` in your shell startup and activate either `vitis-ai-pytorch` or `vitis-ai-tensorflow2` as needed.

## Atomic Operations

Each step below is independent and can be turned into its own script/target. The default path assumes `VAI_ROOT=${HOME}/vitis_ai`.

1. **Base system setup**
   - Ensure `sudo apt-get update` succeeds.
   - Install required developer, multimedia, and Python packages (build-essential, cmake, git, curl, python3, python3-dev, python3-pip, OpenCV/GStreamer headers, etc.).
   - Install the json-c library (either via Ubuntu packages or by building `json-c-0.18-20240915` from source to match Docker).
   - Create and chmod a scratch directory if you plan to unpack archives (e.g., `/scratch`).

2. **Conda bootstrap**
   - Download Miniforge (`Miniforge3-25.3.1-0-Linux-x86_64.sh`).
   - Install to `"$VAI_ROOT/conda"` with `bash <installer> -b -p "$VAI_ROOT/conda"`.
   - Initialize conda for the current shell and install `mamba` in the base environment (`conda install -n base -y mamba -c conda-forge`).
   - Append `. "$VAI_ROOT/conda/etc/profile.d/conda.sh"` to the user’s `~/.bashrc`.

3. **Vitis AI conda channel**
   - If you have a tarball (`conda-channel-<ver>.tar.gz`), unpack it to an accessible directory (e.g., `/scratch/conda-channel`) and use `file:///scratch/conda-channel` as the channel URI.
   - Otherwise, reference the hosted URL directly.
   - You only need to append the channel at the environment level (`conda config --env --append channels <uri>`).

4. **PyTorch GPU environment**
   - Activate conda and create the env: `mamba env create -f docker/conda/gpu_conda/vitis-ai-pytorch.yml`.
   - Activate `vitis-ai-pytorch`.
   - Install CUDA-enabled wheels: `pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117`.
   - Install supplemental packages (`scipy`, `numpy==1.22`, `protobuf==3.20.3`, `tensorboard`, `graphviz==0.19.1`, `imageio`, `scikit-image`, `natsort`, `nibabel`, `easydict`, `yacs`, `fire`, `numba`, `loguru`).
   - Copy the compiler arch data: `cp -r "$CONDA_PREFIX/lib/python3.8/site-packages/vaic/arch" "$VAI_ROOT/compiler/arch"`.
   - Optionally run a sanity check: `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"`.

5. **TensorFlow 2 GPU environment**
   - Activate conda and create the env: `mamba env create -f docker/conda/gpu_conda/vitis-ai-tensorflow2.yml`.
   - Activate `vitis-ai-tensorflow2`.
   - Install GPU TF stack: `pip install tensorflow==2.12 keras==2.12` (change versions if you adopt unified GPU wheels).
   - Install additional Python packages:
     - Conda/mamba: `pydot`, `pyyaml`, `jupyter`, `ipywidgets`, `dill`, `progressbar2`, `pytest`, `pandas`, `matplotlib`, `pillow`.
     - Pip: requirements from `docker/conda/pip_requirements.txt`, plus `transformers`, `pycocotools`, `scikit-learn`, `scikit-image`, `tqdm`, `easydict`, `onnx==1.13.0`, `numpy==1.22`.
   - Pin compatibility libraries: uninstall `h5py`, then `mamba install -y --override-channels --force-reinstall h5py=2.10.0 -c conda-forge`, and `pip install --force-reinstall protobuf==3.20.3 numpy==1.22`.
   - Copy compiler arch data as above.
   - Optional validation: `python -c "import tensorflow as tf; print(tf.__version__, tf.test.is_built_with_cuda(), tf.config.list_physical_devices('GPU'))"`.

6. **Optional runtime components**
   - **XRT/XRM**: download the `.deb` files referenced by `XRT_URL` and `XRM_URL`, install via `sudo apt install ./xrt.deb ./xrm.deb`.
   - **Vitis AI runtime**: either unpack `VAI_DEB_CHANNEL` tarball and `apt install` all contained `.deb`s, or add the repo and install `libunilog`, `libtarget-factory`, `libxir`, `libvart`, `libvitis_ai_library`, `librt-engine`, `aks`. Re-run `sudo ldconfig` and adjust `libvart-dpu-runner.so` symlink if needed.

7. **Cleanup (optional)**
   - Run `conda clean -y --force-pkgs-dirs` in each environment.
   - Remove temporary tarballs/unpacked channels if space matters.

## Optimizations

These adjustments keep the native install lean while staying functionally equivalent to the Docker GPU images.

- **Skip container scaffolding**: no need to create service users, adjust `/etc/sudoers`, or install `gosu`.
- **Avoid NVIDIA repo tweaks**: the host driver stack should remain intact; remove the `sed` that comments NVIDIA apt entries.
- **Toolchain simplification**: unless you compile Vitis AI from source, skip multi-version GCC setup and `update-alternatives` juggling.
- **Use distro packages where possible**: prefer Ubuntu’s `libjson-c-dev` instead of building json-c, unless specific versions are required.
- **Selective dependencies**: many multimedia/OpenCL packages cater to broad demos. Install only what your workloads need.
- **Modern TensorFlow option**: upgrading to TF 2.15+ unified GPU wheels eliminates the `h5py==2.10.0`, `protobuf==3.20.3`, `numpy==1.22` pins.
- **Skip runtime `.deb`s for dev-only hosts**: XRT/XRM and Vitis AI runtime libraries are unnecessary unless you run DPU workloads locally.
- **Channel management**: configure conda channels per environment rather than system-wide, and minimize redundant channel add/remove cycles.
- **Temporary storage**: use `mktemp` or user directories instead of `/scratch` if you prefer not to modify system-wide paths.

Following the flow above reproduces the PyTorch and TensorFlow 2 GPU environments from the Vitis AI Docker builds on a standard Ubuntu machine using only public documentation and artifacts.

## Installation instructions (draft)

Following instructions assume that *this* repository is cloned to `/tools/Xilinx/Vitis-AI`. Most options are configurable, but it is the default expected location.

- `sudo ./native/scripts/install_base.sh install_base`
- `sudo ./native/scripts/install_base.sh install_compilers`
- `sudo -u xilinx ./native/scripts/install_conda.sh install_conda`
  - Assuming you created `xilinx` user and want to install to `/home/xilinx/conda`
- `sudo -u xilinx ./native/scripts/install_torch.sh install_vai_conda_channel`
  - Install and protect Vitis-AI `conda` channel using `xilinx` user
- `./native/scripts/install_torch.sh install_torch`
  - Install Python dependencies as ordinary user to simplify user experience
