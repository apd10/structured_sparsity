# installation
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv --seed --python 3.12
source .venv/bin/activate

# install pytorch according to the cuda version



# run the data collection
bash data.sh



# Results

## A6000 ADA (cuda 12.5)

| **M=N (K=10240) Benchmark** | **Speedup** | **Simple Benchmark** | **Speedup** |
| ----------------- | ----------: | -------------------- | ----------: |
| 2048 × 10240      |       1.29× | 128                  |       0.02× |
| 4096 × 10240      |       1.58× | 256                  |       0.02× |
| 8192 × 10240      |       1.53× | 512                  |       0.02× |
| 10240 × 10240     |       1.12× | 1024                 |       0.04× |
| 16384 × 10240     |       1.04× | 2048                 |       0.28× |
| 20480 × 10240     |       0.97× | 4096                 |       1.27× |
|                   |             | 8192                 |       1.66× |
|                   |             | 16384                |       0.93× |

