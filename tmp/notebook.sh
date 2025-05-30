#!/bin/bash
#SBATCH -J jn
#SBATCH -A L00120230003
#SBATCH --gres=gpu:3
#SBATCH -p p-A800
#SBATCH -N 1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4


cd /home/chenyupeng/yupeng/jaggi-lr

# get tunneling info
XDG_RUNTIME_DIR=""
port=$(shuf -i8000-9999 -n1) # 可以固定下来
node=$(hostname -s)
user=$(whoami)
cluster=$(hostname -f | awk -F"." '{print $2}')

### 在这里添加你的服务器地址
clusterurl="10.26.6.81"

export PATH=$PATH:~/.local/bin

# print tunneling instructions jupyter-log
echo -e "
MacOS or linux terminal command to create your ssh tunnel:
ssh -N -L ${port}:${node}:${port} ${user}@${clusterurl}
 
 Here is the MobaXterm info:

 Forwarded port:same as remote port
 Remote server: ${node}
 Remote port: ${port}
 SSH server: ${cluster}.${clusterurl}
 SSH login: $user
 SSH port: 22

 Use a Browser on your local machine to go to:
 localhost:${port} (prefix w/ https:// if using password)

 or copy the token from the error file"

 # load modules or conda environments here
 # e.g. farnam:
 # module load Python/2.7.13-foss-2016b
 # conda env activate mx
 # DON'T USE ADDRESS BELOW. 
 # DO USE TOKEN BELOWa
 source /cm/shared/apps/anaconda3/etc/profile.d/conda.sh
 conda activate gpt
 jupyter-notebook --no-browser --port=${port} --ip=${node} --NotebookApp.token='' --NotebookApp.password=''