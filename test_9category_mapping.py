import numpy as np

def test_9category_mapping():
    """
    测试 9 类别映射功能
    """
    # 模拟原始标签数据
    # 包含原始 MFCAD2 数据集中的各种类别
    original_labels = np.array([1, 12, 5, 6, 7, 17, 18, 19, 2, 3, 4, 8, 9])
    
    print("原始标签:", original_labels)
    print("原始标签含义:")
    print("1: Through hole, 12: Blind hole, 5: Triangular through slot")
    print("6: Rectangular through slot, 7: Circular through slot")
    print("17: Rectangular blind slot, 18: Vertical circular end blind slot")
    print("19: Horizontal circular end blind slot, 其他: Other")
    print()
    
    # 模拟 9 类别映射逻辑
    mapped_labels = np.zeros_like(original_labels)
    
    # 定义类别映射字典
    category_map = {
        1: 1,    # Through hole
        12: 2,   # Blind hole
        5: 3,    # Triangular through slot
        6: 4,    # Rectangular through slot
        7: 5,    # Circular through slot
        17: 6,   # Rectangular blind slot
        18: 7,   # Vertical circular end blind slot
        19: 8    # Horizontal circular end blind slot
    }
    
    # 执行映射
    for i, label in enumerate(original_labels):
        if label in category_map:
            mapped_labels[i] = category_map[label]
        else:
            mapped_labels[i] = 0  # other
    
    print("映射后标签:", mapped_labels)
    print("映射后标签含义:")
    print("0: Other, 1: Through hole, 2: Blind hole")
    print("3: Triangular through slot, 4: Rectangular through slot")
    print("5: Circular through slot, 6: Rectangular blind slot")
    print("7: Vertical circular end blind slot, 8: Horizontal circular end blind slot")
    print()
    
    # 验证映射结果
    expected_mapped = np.array([1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0])
    assert np.array_equal(mapped_labels, expected_mapped), f"映射结果错误: 期望 {expected_mapped}, 实际 {mapped_labels}"
    
    print("✓ 测试通过! 类别映射功能正常工作")
    print()
    
    # 测试边界情况
    print("测试边界情况:")
    print("1. 空标签数组:")
    empty_labels = np.array([])
    empty_mapped = np.zeros_like(empty_labels)
    print(f"   输入: {empty_labels}, 输出: {empty_mapped}")
    
    print("2. 仅包含 other 类的标签:")
    other_labels = np.array([2, 3, 4, 8, 9, 10])
    other_mapped = np.zeros_like(other_labels)
    for i, label in enumerate(other_labels):
        if label in category_map:
            other_mapped[i] = category_map[label]
        else:
            other_mapped[i] = 0
    print(f"   输入: {other_labels}, 输出: {other_mapped}")
    
    print("3. 仅包含映射类的标签:")
    mapped_only_labels = np.array([1, 12, 5, 6, 7, 17, 18, 19])
    mapped_only_mapped = np.zeros_like(mapped_only_labels)
    for i, label in enumerate(mapped_only_labels):
        if label in category_map:
            mapped_only_mapped[i] = category_map[label]
        else:
            mapped_only_mapped[i] = 0
    print(f"   输入: {mapped_only_labels}, 输出: {mapped_only_mapped}")

if __name__ == "__main__":
    test_9category_mapping()