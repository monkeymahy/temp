import os


def ensure_directories_exist(dir_paths):
    """
    确保所需的目录结构已创建，如不存在则自动创建。

    参数:
        dir_paths (list[str]): 需要检查和创建的目录路径列表。

    说明:
        - 跳过空字符串和重复路径，避免无效操作。
        - 使用os.makedirs，exist_ok=True确保幂等性。
    """
    # 类型检查，确保输入为字符串列表
    assert isinstance(dir_paths, list) and all(isinstance(d, str) for d in dir_paths)

    # 使用生成器表达式去除空字符串和首尾空白，并用集合去重
    unique_dirs = {d.strip() for d in dir_paths if d and d.strip()}
    for directory in unique_dirs:
        # os.makedirs: 递归创建目录，exist_ok=True表示目录已存在时不会抛出异常
        os.makedirs(directory, exist_ok=True)
