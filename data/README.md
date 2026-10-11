# 数据

[references/](references/) 保存 P2 的两份收敛参考解：ε=1 和 10⁻³，空间网格 1024、角度节点 512，精度级别 B。NPZ 中包含 x、velocity、weights、f、r、j、rho；同名 JSON 记录收敛条件和运行信息。

P1、P3、P4 等制造解问题使用解析解；训练配点在线采样，不存在单独的监督学习训练集。定义见 [problems](../src/oeaprfm/problems/)，神经网络采样见 [data_pipeline.py](../src/baselines/oe_apnn/data_pipeline.py)。P5 的其他参考解未收集到本目录，需要按复现实验说明生成。

生成 P2 参考数据：`python experiments/generate_reference.py --help`。相关入口默认读取本目录的 references/。

[manifest.json](manifest.json) 记录本次收集的 279 个数据、模型和结果文件的原始位置、大小和 SHA-256。运行 `python scripts/check_assets.py` 可独立核验，无须本地历史目录。
