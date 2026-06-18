# import pandas as pd
# import os

# def move_target_rows_to_end(input_csv_path, output_csv_path=None, target_smi="O=c1[nH]cc(F)c(=O)[nH]1"):
#     """
#     将CSV中包含指定druga_smi或drugb_smi的行移至文件末尾
    
#     参数：
#     input_csv_path: 输入CSV文件路径（必填）
#     output_csv_path: 输出CSV文件路径（默认与输入同目录，文件名加"_reordered"后缀）
#     target_smi: 目标分子结构（默认是需求中的O=c1[nH]cc(F)c(=O)[nH]1）
#     """
#     # 1. 读取CSV文件（确保index列作为普通数据列读取，不设为行索引）
#     df = pd.read_csv(input_csv_path)
    
#     # 2. 检查必要列是否存在
#     required_cols = ["index", "druga_smi", "drugb_smi", "cell_id", "synergy"]
#     missing_cols = [col for col in required_cols if col not in df.columns]
#     if missing_cols:
#         raise ValueError(f"CSV文件缺少必要列：{', '.join(missing_cols)}")
    
#     # 3. 区分两类行：包含目标smi的行 和 不包含目标smi的行
#     # 条件：druga_smi等于目标smi OR drugb_smi等于目标smi
#     has_target = (df["druga_smi"] == target_smi) | (df["drugb_smi"] == target_smi)
#     no_target_df = df[~has_target].copy()  # 不包含目标smi的行（保留原始顺序）
#     target_df = df[has_target].copy()     # 包含目标smi的行（保留原始顺序）
    
#     # 4. 合并：先放非目标行，再放目标行
#     reordered_df = pd.concat([no_target_df, target_df], ignore_index=True)
    
#     # 5. 处理输出路径（默认覆盖前加后缀）
#     if output_csv_path is None:
#         input_dir, input_filename = os.path.split(input_csv_path)
#         filename, ext = os.path.splitext(input_filename)
#         output_csv_path = os.path.join(input_dir, f"{filename}_reordered{ext}")
    
#     # 6. 保存结果（不保留pandas自动生成的行索引）
#     reordered_df.to_csv(output_csv_path, index=False)
    
#     # 7. 打印执行信息
#     print(f"处理完成！")
#     print(f"原始总行数：{len(df)}")
#     print(f"包含目标smi的行数：{len(target_df)}（已移至末尾）")
#     print(f"不包含目标smi的行数：{len(no_target_df)}（保留原始顺序）")
#     print(f"输出文件路径：{output_csv_path}")

# # ------------------- 示例：根据你的需求修改路径 -------------------
# if __name__ == "__main__":
#     # 替换为你的输入CSV文件路径（例如"/home/cyw/ESA/13/synergyDMPNN2.csv"）
#     INPUT_CSV = "/home/lkp/cywhome/PITSynergy/PITSynergy/13/synergyDMPNN2.csv"
#     # 可选：指定输出路径，不指定则默认在输入同目录生成"xxx_reordered.csv"
#     OUTPUT_CSV = '/home/lkp/cywhome/PITSynergy/PITSynergy/13/synergyDMPNN.csv' # 例如"/path/to/your/output_reordered.csv"
    
#     # 调用函数执行重排
#     move_target_rows_to_end(
#         input_csv_path=INPUT_CSV,
#         output_csv_path=OUTPUT_CSV,
#         target_smi="O=c1[nH]cc(F)c(=O)[nH]1"  # 固定目标分子结构
#     )

import pandas as pd
import itertools

# ========= 1. 读取已有 CSV =========
input_csv = "/home/lkp/cywhome/PITSynergy/PITSynergy/13/synergyDMPNN2.csv"
df = pd.read_csv(input_csv)

# ========= 2. 提取药物 & 细胞系全集 =========
all_drugs = sorted(set(df["druga_smi"]).union(set(df["drugb_smi"])))
all_cells = sorted(df["cell_id"].unique())

print("All drugs:", all_drugs)
print("All cells:", all_cells)

# ========= 3. 构造“已存在组合”的集合 =========
# 用 (drugA, drugB, cell) 且 drugA < drugB 规范化
existing_set = set()

for _, row in df.iterrows():
    d1, d2 = sorted([row["druga_smi"], row["drugb_smi"]])
    c = row["cell_id"]
    existing_set.add((d1, d2, c))

# ========= 4. 构造“全组合空间” =========
all_combinations = set()

for d1, d2 in itertools.combinations(all_drugs, 2):
    for c in all_cells:
        all_combinations.add((d1, d2, c))

# ========= 5. 找缺失的组合 =========
missing_combinations = all_combinations - existing_set

print(f"Missing combinations: {len(missing_combinations)}")

# ========= 6. 写成新的 DataFrame =========
missing_df = pd.DataFrame(
    list(missing_combinations),
    columns=["druga_smi", "drugb_smi", "cell_id"]
)

missing_df["synergy"] = 1.0
missing_df.insert(0, "index", range(len(missing_df)))

# ========= 7. 保存 =========
output_csv = "/home/lkp/cywhome/PITSynergy/PITSynergy/13/synergypre.csv"
missing_df.to_csv(output_csv, index=False)

print(f"Saved missing combinations to: {output_csv}")
