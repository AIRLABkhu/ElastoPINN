export ENV_NAME="pinn"

conda create -n $ENV_NAME python=3.9 -y
conda activate $ENV_NAME 

pip install ipykernel ipywidgets tqdm
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 cpuonly -c pytorch -y

conda install -c iopath iopath -y
conda install jupyter -y
pip install scikit-image matplotlib imageio plotly opencv-python open3d yacs

conda install -c fvcore -c conda-forge fvcore -y
pip install black usort flake8 flake8-bugbear flake8-comprehensions

git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
pip install -e .
cd ..

pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cpu.html

