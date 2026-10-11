# 模型

[index.csv](index.csv) 按方法内的实验组、问题、ε、随机种子和训练步数索引全部 48 组 OE-APNN 模型：30 组完整训练和 18 组成本实验。

目录格式：`oe_apnn/{baseline,cost}/p*/eps_*/steps_*/seed_*/`。每组包含 r.pt、j.pt、config.json、metrics.json、history.npz。权重为两个网络的 state_dict，配置和指标保持原样，历史来源见 [数据清单](../data/manifest.json)。

安装 requirements-apnn.txt 后，从仓库根目录加载：

```python
import torch
from models.load_apnn import load_model

solver, nets = load_model(
    "models/oe_apnn/baseline/p1/eps_1e-03/steps_20000/seed_11"
)
x = torch.tensor([[0.5]], dtype=torch.float64)
v = torch.tensor([[0.2]], dtype=torch.float64)
with torch.no_grad():
    f = solver.reconstruct(nets, x, v)
```

二维问题传入 x、y、theta 三列张量。加载器在 CPU 上恢复双精度网络，并通过原求解器施加奇偶投影。

[OE-APRFM 模型说明](oe_aprfm/README.md)：已收集的稳态结果未保存可重载系数，不能把预测场当作模型。其他对比方法未找到可发布的独立权重，因此不创建空模型记录。
