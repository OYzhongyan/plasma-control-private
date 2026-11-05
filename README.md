# 服务器模型评测

本分支 `server/offline-grading` 由主办方部署；交给选手的 baseline 位于 `main` 分支。

评分程序按手册 4.5 从比赛平台拉取待评模型 ZIP，在本地 CPU 上推理并调用 HFM 仿真器生成轨迹，使用主办方目标计算七项贡献分，回传平台更新排行榜。

安装、HFM 启动、平台配置、本地模型核验和常驻服务部署见 [server_evaluator/README.md](server_evaluator/README.md)。

平台综合分公式保持为 `[0:Ip]+[0:LCFS]+[0:X]+[0:strike]+[0:psiX]+[0:X2]+[0:psiX2]`，降序排名，满分 30 分。
