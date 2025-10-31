from decimal import Decimal
import json
import ijson
import os
from pathlib import Path
import tempfile
from typing import Optional, Tuple


def _json_serializer(obj):
    """
    JSON 默认序列化回调函数（用于 json.dump(..., default=...)）
    - 支持 Decimal：尽量保持整数形式，否则转为 float
    - 可以扩展其它类型（date/datetime/bytes/Path/set 等），这里仅实现 Decimal 的安全处理
    """
    if isinstance(obj, Decimal):
        try:
            # 如果 Decimal 表示的是整数值，就转成 int，避免 1.0 之类的不必要浮点
            if obj == obj.to_integral():
                return int(obj)
        except Exception:
            # 若出现任何异常，退回到把 Decimal 转为 float
            pass
        return float(obj)
    # 对于未知类型，仍然让 json 抛出 TypeError，以便调用方发现并处理
    return obj


def split_large_json_stream(
    input_path: str,
    output_dir: Optional[str] = None,
    ijson_prefix: str = "item",
    start_index: int = 0,
    index_width: int = 8,
    compact: bool = True,
    progress_interval: int = 1000,
    encoding: str = "utf-8",
) -> int:
    """
    将一个大的 JSON 文件按顶层流式元素拆成多个小文件（逐项写入，避免一次性加载到内存）。
    适用于顶层为数组（prefix="item"）或对象中某个数组（例如 "data.item"）。
    返回值：实际写入的小文件数量（int）。

    参数说明（中文）：
        - input_path: 待拆分的大 JSON 文件路径（本函数对本地文件进行流式解析）
        - output_dir: 若为 None，则在输入文件同目录创建 "<basename>_split" 文件夹
        - ijson_prefix: ijson.items 使用的 prefix，默认 "item"（顶层数组元素）。
                        若大 JSON 为 {"data": [...]} 则使用 "data.item"
        - start_index: 输出文件编号的起始值（默认 0）
        - index_width: 输出文件编号的零填充宽度（例如 8 -> 00000001）
        - compact: 是否以紧凑格式写入 JSON（去掉空格）
        - progress_interval: 每写入多少个文件打印一次进度；设为 <=0 则不打印
        - encoding: 输出文件的字符编码（默认 utf-8）

    实现要点：
        - 使用 ijson 流式解析，避免把整个文件读取到内存
        - 对每个条目先写入目录下的临时文件，写完后用 os.replace 原子性替换到目标文件名
        - 在出现 KeyboardInterrupt 时友好中断并返回已写入数量；在异常时抛出异常以便上层处理
    """
    in_path = Path(input_path)
    if not in_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {in_path}")

    # 输出目录默认值：<输入文件名>_split（与输入在同一目录）
    out_dir = (
        Path(output_dir) if output_dir else in_path.parent / f"{in_path.stem}_split"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # 基础文件名前缀（使用输入文件 stem）
    file_base = in_path.stem

    written_count = 0  # 实际写入计数

    # 以二进制模式打开输入文件供 ijson 使用（ijson 要求二进制或文件对象）
    try:
        with open(in_path, "rb") as f:
            # 注意：ijson.items 返回的是一个生成器，按元素流式产生单个 JSON 对象
            for idx, item in enumerate(ijson.items(f, ijson_prefix), start=start_index):
                # 生成目标文件名，例如 basename_00000001.json
                filename = f"{file_base}_{idx:0{index_width}d}.json"
                final_path = out_dir / filename

                # 先写入临时文件（使用输出目录作为临时文件目录，避免跨目录重命名问题）
                # Windows 下不能重命名一个已打开的文件，因此使用 delete=False 并在 with 块外 replace
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding=encoding,
                    delete=False,
                    dir=str(out_dir),
                ) as tmp:
                    # 使用统一的序列化回调，compact 模式去掉多余空白
                    if compact:
                        json.dump(
                            obj=item,
                            fp=tmp,
                            ensure_ascii=False,
                            separators=(",", ":"),
                            default=_json_serializer,
                        )
                    else:
                        json.dump(
                            obj=item,
                            fp=tmp,
                            ensure_ascii=False,
                            indent=2,
                            default=_json_serializer,
                        )
                    tmp_name = tmp.name

                # 原子性替换：如果目标文件已存在，会被替换（os.replace 在大多数平台上为原子操作）
                os.replace(tmp_name, final_path)

                written_count += 1

                # 定期打印进度
                if (
                    progress_interval
                    and progress_interval > 0
                    and (written_count % progress_interval == 0)
                ):
                    print(f"written {written_count} files, latest: {final_path}")

    except KeyboardInterrupt:
        # 用户中断，打印已完成的数量并返回
        print(f"Interrupted by user. {written_count} files written so far.")
        return written_count
    except Exception as exc:
        # 在异常发生时报告当前已写入数量，并把异常继续抛出以便上层处理或日志记录
        print(f"Error after writing {written_count} files: {exc}")
        raise

    return written_count


if __name__ == "__main__":
    # 示例：仅作演示。请根据实际路径修改。
    count = split_large_json_stream(
        input_path="C:/Data/graphs.json",
        output_dir="C:/Data/MFCAD2_split1",
        ijson_prefix="item",
        progress_interval=1000,
    )
    print(f"Total files written: {count}")
