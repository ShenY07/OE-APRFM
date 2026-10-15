# OE-APRFM 模型

`baseline/p{1..5}/eps_{1e+00,1e-03}/seed_{7,11,17}/` 保存与 OE-APNN 基准组对应的模型。每组包含 `model.npz`（求解系数和随机特征参数）、`config.json`（重建配置和问题定义）、`metrics.json`（求解指标）。

```python
from models.load_aprfm import load_model
predict = load_model("models/oe_aprfm/baseline/p1/eps_1e-03/seed_11")
f = predict([0.2, 0.8], [0.3, -0.3])
```

从仓库根目录设置 `PYTHONPATH=src:.`。二维模型传入 x、y、theta。加载器核对保存的随机特征参数，并恢复可直接预测的函数。每组运行均与保存的预测场核对重载结果。

`extensions/angular_domains/` 保存从现有角域对比系数恢复、并与原预测场逐点核对的 12 组完整模型。

`extensions/e6_periodic/` 保留含参数、系数及时间轨迹的 E6 状态。E3/E5 仅含系数的旧文件保留在对应实验结果中，不再作为完整模型重复列出。

[comparison_index.csv](../comparison_index.csv) 对照两个方法的问题、ε、种子及实际模型目录。P5 基准是方孔制造解；非均匀介质 P5 属于扩展实验。OE-APNN 的 `cost/` 是训练步数研究，没有一一对应的随机特征训练步数。
