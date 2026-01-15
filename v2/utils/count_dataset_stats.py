import os
import sys
import numpy as np

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def count_dataset_statistics():
    """
    计算数据集统计信息：
    1. 训练、验证、测试集的样本数量
    2. 数据集中0-24各个类别的数量（原始25类别）
    """
    from v2.dataset.MFCAD2datamodule import MFCAD2Dataset
    
    print("=== 计算MFCAD2数据集统计信息（原始25类别）===")
    
    # 数据集根路径
    data_root = r"E:\AAGnetV2\aagnet\MFCAD2"
    
    # 统计信息字典 - 原始25类别
    stats = {
        "train": {"total": 0, "classes": {i: 0 for i in range(25)}},
        "val": {"total": 0, "classes": {i: 0 for i in range(25)}},
        "test": {"total": 0, "classes": {i: 0 for i in range(25)}}
    }
    
    # 遍历训练、验证、测试集
    for split in ["train", "val", "test"]:
        print(f"\n处理 {split} 集...")
        
        # 创建数据集实例，使用原始25类别
        dataset = MFCAD2Dataset(
            root_dir=data_root,
            split=split,
            use_9category=False  # 不使用9类别映射，使用原始25类别
        )
        
        # 计算总样本数量
        stats[split]["total"] = len(dataset)
        print(f"  总样本数量: {stats[split]['total']}")
        
        # 计算每个类别的数量
        total_nodes = 0
        for i in range(len(dataset)):
            try:
                # 获取数据样本
                sample = dataset[i]
                labels = sample["graph"].ndata["y"].numpy()
                
                # 统计每个类别的数量
                for label in labels:
                    if 0 <= label <= 24:
                        stats[split]["classes"][label] += 1
                        total_nodes += 1
                        
            except Exception as e:
                print(f"  处理样本 {i} 时出错: {e}")
                continue
        
        # 打印类别统计信息
        print(f"  总节点数量: {total_nodes}")
        print(f"  类别分布（所有25个类别）:")
        # 打印所有25个类别
        for cls in range(25):
            count = stats[split]["classes"][cls]
            percentage = (count / total_nodes * 100) if total_nodes > 0 else 0
            print(f"    类别 {cls}: {count} ({percentage:.2f}%)")
    
    # 计算总体统计信息
    print("\n=== 总体统计信息 ===")
    total_samples = sum(stats[split]["total"] for split in ["train", "val", "test"])
    print(f"总样本数量: {total_samples}")
    print(f"训练集: {stats['train']['total']} ({stats['train']['total']/total_samples*100:.2f}%)")
    print(f"验证集: {stats['val']['total']} ({stats['val']['total']/total_samples*100:.2f}%)")
    print(f"测试集: {stats['test']['total']} ({stats['test']['total']/total_samples*100:.2f}%)")
    
    # 计算总体类别分布
    total_classes = {i: 0 for i in range(25)}
    total_all_nodes = 0
    for split in ["train", "val", "test"]:
        for cls in range(25):
            total_classes[cls] += stats[split]["classes"][cls]
        total_all_nodes += sum(stats[split]["classes"].values())
    
    print("\n总体类别分布（所有25个类别）:")
    for cls in range(25):
        count = total_classes[cls]
        percentage = (count / total_all_nodes * 100) if total_all_nodes > 0 else 0
        print(f"  类别 {cls}: {count} ({percentage:.2f}%)")
    
    return stats

if __name__ == "__main__":
    count_dataset_statistics()
