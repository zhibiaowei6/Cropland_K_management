# -*- coding: utf-8 -*-
"""
本脚本适用于 Windows + Anaconda Prompt，可稳定运行多进程大规模数据处理
原始版本的所有多进程崩溃问题（特别是数十万 future 造成进程池崩溃）
已经在该版本中彻底修复。
"""

import os
import pandas as pd
import numpy as np
from scipy.interpolate import make_interp_spline
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
import gc


warnings.filterwarnings('ignore')


# ================================================================
# 处理函数
# ================================================================
def find_first_small_diff_detailed(arr):
    """在升序排列的数组中，找到后一个数比前一个数首次差值小于2的位置"""
    diffs = arr[1:] - arr[:-1]
    small_diff_indices = np.where(diffs < 2)[0]
    if len(small_diff_indices) > 0:
        first_index = small_diff_indices[0]
        return first_index, arr[first_index]
    else:
        return len(arr) - 1, arr[-1]


def process_row(row_data_all, row_idx, measur_names, K_num, K_smooth):
    """
    单行数据处理 —— 找最大10个措施，平均产量序列，再做拐点分析
    """
    try:
        # 最后一列（产量最大值）取前 10大
        last_values = row_data_all[:, -1]                     # shape = (96,)
        top10_idx = np.argsort(last_values)[-10:]               # 取最大10个 measure 的索引（升序）
        top10_idx = top10_idx[::-1]                             # 按从大到小排序

        top10_values = last_values[top10_idx]                   # 10个最大值
        top10_measures = [measur_names[i] for i in top10_idx]   # 10个对应措施名称

        # 取这10个措施的完整产量序列，做平均
        # row_data_all: shape = (96, 9)
        top10_yields = row_data_all[top10_idx, :]               # shape = (10, 9)
        yields_mean = top10_yields.mean(axis=0)                # shape = (9,)

        # 样条拟合
        spline = make_interp_spline(K_num, yields_mean, k=3)
        y_smooth = spline(K_smooth)

        # 找拐点
        knee_point, knee_yield = find_first_small_diff_detailed(y_smooth)

        # 输出信息
        print(
            f"第 {row_idx+1} 栅格: 最大10个产量 {top10_values.tolist()}, "
            f"拐点产量 {knee_yield:.2f}, 拐点 {knee_point}, 措施 {top10_measures}"
        )

        # 结果字典
        result = {
            "索引": row_idx,
            "最大产量1": top10_values[0],
            "最大产量2": top10_values[1],
            "最大产量3": top10_values[2],
            "最大产量4": top10_values[3],
            "最大产量5": top10_values[4],
            "最大产量6": top10_values[5],
            "最大产量7": top10_values[6],
            "最大产量8": top10_values[7],
            "最大产量9": top10_values[8],
            "最大产量10": top10_values[9],
            "措施1": top10_measures[0],
            "措施2": top10_measures[1],
            "措施3": top10_measures[2],
            "措施4": top10_measures[3],
            "措施5": top10_measures[4],
            "措施6": top10_measures[5],
            "措施7": top10_measures[6],
            "措施8": top10_measures[7],
            "措施9": top10_measures[8],
            "措施10": top10_measures[9],
            "拐点产量": knee_yield,
            "拐点": knee_point
        }

        return result

    except Exception as e:
        print(f"行 {row_idx} 处理出错: {e}")
        return None

    finally:
        del last_values, top10_idx, top10_values, top10_measures, top10_yields, yields_mean, spline, y_smooth

# 必须在顶层定义 wrapper（Windows 多进程要求）
def process_row_wrapper(args):
    """适配多进程传参"""
    try:
        return process_row(*args)
    except Exception as e:
        print(f"行处理出错: {e}")
        return None


# ================================================================
# 数据加载
# ================================================================
def load_data_in_batches(file_paths, batch_size=5):
    """分批加载数据（避免 12GB 峰值）"""
    all_data = []

    for i in range(0, len(file_paths), batch_size):
        batch_files = file_paths[i:i + batch_size]
        batch_data = []

        for file_path in batch_files:
            try:
                df = pd.read_csv(file_path)
                batch_data.append(df.values)
                print(f"已加载: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"加载 {file_path} 出错: {e}")

        if batch_data:
            all_data.extend(batch_data)

        del batch_data
        gc.collect()

    return np.array(all_data)


# ================================================================
# 主程序
# ================================================================
def main(folder_path):

    # 获取所有 CSV
    item_paths = []
    measur_names = []

    for item in os.listdir(folder_path):
        if item.endswith(".csv"):
            full_path = os.path.join(folder_path, item)
            item_paths.append(full_path)

            try:
                measure_name = full_path.split("Table_")[1].split("_predictions.csv")[0]
            except:
                measure_name = os.path.basename(full_path)

            measur_names.append(measure_name)

    if len(item_paths) == 0:
        print("未找到 CSV 文件")
        return

    print(f"开始加载 {len(item_paths)} 个数据文件...")

    # 加载数据
    all_data_array = load_data_in_batches(item_paths, batch_size=5)

    if len(all_data_array) == 0:
        print("未加载到数据")
        return

    measures, num_rows, num_vars = all_data_array.shape
    print(f"数据加载完成，形状: {all_data_array.shape}")

    # 插值参数
    K_num = np.array([0, 15, 30, 45, 60, 75, 90, 105, 120])
    K_smooth = np.linspace(K_num.min(), K_num.max(), 121)

    print("开始处理数据...")

    # ============================================================
    # ★★★ 最重要的改动：按批次提交任务（每批 2000 行） ★★★
    # 避免一次性提交 63 万个 future 导致进程池崩溃
    # ============================================================

    max_workers = min(os.cpu_count(), 4)
    batch_size = 2000

    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:

        for start in range(0, num_rows, batch_size):
            end = min(start + batch_size, num_rows)
            print(f"\n提交批次: {start} - {end}")

            futures = []

            # 提交本批次任务
            for row_idx in range(start, end):
                args = (all_data_array[:, row_idx, :], row_idx, measur_names, K_num, K_smooth)
                futures.append(executor.submit(process_row_wrapper, args))

            # 获取结果
            for future in as_completed(futures):
                r = future.result()
                if r is not None:
                    results.append(r)

            print(f"批次完成: {start} - {end}（共 {len(results)} 条结果）")

            # 释放内存
            del futures
            gc.collect()

    # ============================================================
    # 保存结果
    # ============================================================
    if results:
        df = pd.DataFrame(results).sort_values(by="索引")
        df.to_csv("结果数据_稳定版_top10_2.csv", index=False, encoding="utf-8-sig")
        print(f"\n处理完成，共 {len(results)} 条记录，已保存为 结果数据_稳定版_top10_2.csv\n")
    else:
        print("无有效结果")


# ================================================================
# Windows 必须使用 main 保护
# ================================================================
if __name__ == "__main__":
    folder_path = r'E:\Jupyter\K\Wheat2\prediction-smalldata2'
    main(folder_path)