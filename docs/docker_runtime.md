# HFM Runtime Release

## 集群 GLIBCXX 问题

MATLAB Runtime R2025a 需要较新的 C++ 运行库。如果 HFM 日志中出现：

```text
/lib64/libstdc++.so.6: version `GLIBCXX_3.4.30' not found
```

说明集群系统自带的 `/lib64/libstdc++.so.6` 版本过旧。优先使用 conda 环境里的运行库：

```bash
conda activate compete_env
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
grep -a GLIBCXX_3.4.30 "${CONDA_PREFIX}/lib/libstdc++.so.6"
```

如果 conda 环境里的库也没有该符号，需要更新：

```bash
conda install -n compete_env -c conda-forge "libstdcxx-ng>=12" "libgcc-ng>=12"
```

`python.slurm` 会在启动 HFM 前检查该符号，避免 HFM 启动失败后才表现为端口 timeout。


## 系统依赖

大多数 Ubuntu 机器需要补齐以下运行库：

```bash
apt-get update
apt-get install -y \
  libgbm1 libasound2 libnss3 \
  libxt6 libxext6 libxmu6 libice6 libsm6 libxrender1 \
  libglu1-mesa libfontconfig1 libfreetype6 libxtst6 \
  libxi6 libxft2 libxss1 libxcb-xinerama0 libxcb-dri3-0
```

如果机器已经装过相近版本的 MATLAB Runtime 依赖，这一步可能不需要重复执行。

首次解压后建议先补执行权限：

```bash
chmod +x run_hfm_server.sh start_hfm_pool.sh env/run_hfm_socket_server.sh env/hfm_socket_server
```

## 快速开始

单实例脚本支持以下环境参数：

- `LISTEN_PORT`
  默认 `5558`
- `MATLAB_SOCKET_TIMEOUT`
  默认 `600`
- `MATLAB_IDLE_TIMEOUT`
  默认 `600`

示例：

```bash
LISTEN_PORT=3333 MATLAB_SOCKET_TIMEOUT=900 MATLAB_IDLE_TIMEOUT=900 ./run_hfm_server.sh
```

多实例脚本支持以下环境参数：

- `--num`
  启动实例数
- `--base-port`
  起始端口
- `--timeout`
  传给每个实例的 socket 和 idle timeout，默认 `600`

示例：

```bash
./start_hfm_pool.sh --num 2 --base-port 2300 --timeout 900
```

这会启动两个端口：

- `127.0.0.1:2230`
- `127.0.0.1:2231`

对应的日志文件分别为：

- `logs/hfm_2230.log`
- `logs/hfm_2231.log`

## 停止和清理

前端实例关闭：

- IDE执行 `Ctrl+C`

后端实例关闭：

```bash
pkill -f '/env/hfm_socket_server'
```

清理日志和运行缓存：

```bash
rm -rf logs runtime_cache ~/.MathWorks/MatlabRuntimeCache
mkdir -p logs
```
