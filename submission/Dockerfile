# 赛事方“手动启动”版本：容器保持运行，由赛事方执行 /app/run.sh 启动推理评测。
# 选手请勿修改本 Dockerfile（含路径、工作目录、注释）。
ARG BASE_IMAGE=crpi-q4qg69o2szruzmaa.cn-beijing.personal.cr.aliyuncs.com/enn_smart/competition_test:r2025a-py311-v1
FROM ${BASE_IMAGE}

# 环境变量：submission 服务端口与 Python 路径（不可改）
ENV SUBMISSION_HOST=0.0.0.0 \
    SUBMISSION_PORT=8000 \
    SUBMISSION_PORT2=8001 \
    SAISDATA_ROOT=/saisdata/40 \
    INFER_RESULT_PATH=/saisresult/infer_result.json \
    SAISDATA_SHOTS_CONFIG=/saisdata/40/inference/shots.yaml \
    PYTHONPATH=/app:/saisdata/40/standalone-env:/saisdata/40/standalone-env/environment

WORKDIR /app

# 创建评测机挂载与结果目录（路径固定）
RUN mkdir -p /saisdata /saisdata/40 /saisresult \
    && chmod 755 /saisdata /saisdata/40 /saisresult

# 安装选手依赖
COPY submission/requirements.txt /app/submission/requirements.txt
RUN python3 --version \
    && pip3 install --no-cache-dir --timeout 120 -r /app/submission/requirements.txt

# 拷贝 submission 代码（含 start_infer.sh、run.sh，选手勿改）
COPY submission /app/submission

RUN chmod +x /app/submission/start_infer.sh

# 赛事方手动启动入口：/app/run.sh 调用 start_infer.sh（选手勿改）
COPY submission/run.sh /app/run.sh
RUN chmod +x /app/run.sh

EXPOSE 8000 8001 5558

# 容器默认不跑评测，赛事方通过执行 /app/run.sh 启动
ENTRYPOINT ["sleep", "infinity"]
