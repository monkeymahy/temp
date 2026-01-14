# V2版本代码README文件

## CLI运行命令

```bash
# 训练脚本
python v2\main.py fit --config [配置文件]
# 例如：python v2\main.py fit --config v2\configs\sf.yaml

# 测试脚本
python v2\main.py test --config [配置文件] --ckpt_path [权重文件]
# 例如：python v2\main.py test --config v2\configs\sf.yaml --ckpt_path output/checkpoints/epoch=18-val_seg_acc_avg=0.64.ckpt
```
