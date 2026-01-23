# V2版本代码README文件

## CLI运行命令

```bash
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
