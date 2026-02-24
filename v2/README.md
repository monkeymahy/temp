# V2版本代码README文件

## SF标签映射说明

- SF 数据默认使用离线标签映射流程：先在数据侧完成标签映射，再启动训练与可视化。
- 代码中的映射函数位于 `v2/utils/data_utils.py`：`map_sf_labels`。
- `SFDataset` 不再做在线标签映射，直接读取磁盘中的标签值。
- 推荐执行顺序：先运行离线标签映射脚本，再执行训练/测试与可视化流程。

## CLI运行命令

```bash
# SF离线标签映射脚本
python v2\dataset\offline_map_sf_labels.py --root_dir C:\Data\SF-JSON

# 训练脚本
python v2\main.py fit --data [DataModule类名] --config [配置文件]
# 例如：python v2\main.py fit --data SFDataModule --config v2\configs\sf_csy.yaml
# 例如：python v2\main.py fit --data MFCAD2DataModule --config v2\configs\MFCAD2_csy.yaml

# 测试脚本
python v2\main.py test --data [DataModule类名] --config [配置文件] --ckpt_path [权重文件]
# 例如：python v2\main.py test --data SFDataModule --config v2\configs\sf.yaml --ckpt_path output/checkpoints/epoch=18-val_seg_acc_avg=0.64.ckpt

# 启动可视化工具
python v2\utils\qt5_visualization.py
```
