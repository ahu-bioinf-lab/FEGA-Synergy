import pandas as pd
import requests
import time
from tqdm import tqdm
import sys

# 增加默认编码，防止Tqdm在某些环境中出现编码错误
# 在某些系统上，tqdm可能会要求stdout使用UTF-8编码，这里尝试设置它
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    # 忽略不支持reconfigure的旧Python版本
    pass


def get_reactome_pathways(uniprot_id):
    """
    通过 UniProt ID 从 Reactome Content Service 获取相关通路。
    """
    url = f"https://reactome.org/ContentService/data/mapping/UniProt/{uniprot_id}/pathways"
    try:
        r = requests.get(url, timeout=3)
        r.raise_for_status()  # 对 4xx/5xx 状态码抛出异常

        # 检查响应是否为空或不是JSON
        if not r.text:
            return ["N/A"]

        data = r.json()

        # 确保数据是一个列表，并且包含 displayName
        pathways = [entry.get('displayName') for entry in data if isinstance(entry, dict) and entry.get('displayName')]

        # 返回去重后的通路列表
        return list(set(pathways)) if pathways else ["N/A"]
    except requests.exceptions.RequestException as e:
        print(f"\n⚠️ 抓取 UniProt ID {uniprot_id} 失败：{e}")
        return ["N/A"]
    except Exception as e:
        print(f"\n⚠️ 处理 UniProt ID {uniprot_id} 数据失败：{e}")
        return ["N/A"]


def process_file(input_csv="drugs.csv", output_csv="/home/lkp/cywhome/PITSynergy/PITSynergy/dataset/drug_target_pathway_reactome_drugdb子集.csv"):
    """
    处理输入 CSV 文件，抓取 Reactome 通路数据并保存。
    假设输入文件包含两列关键信息：药物名称和靶点 UniProt ID 列表。
    """
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"❌ 错误：未找到输入文件 {input_csv}")
        return

    results = []

    # -----------------------------------------------------------
    # !!! 请注意 !!!
    # 如果您的药物名称列名不是 'Drug_Name'，请修改下面的 'Drug_Name'
    # 例如，如果列名是 'BindingDB Ligand Name'，则改为 row['BindingDB Ligand Name']
    # -----------------------------------------------------------

    # 假设药物名列为 'Drug_Name'，靶点ID列为 'Entry_ID'
    DRUG_NAME_COL = 'Drug_Name'
    if 'BindingDB Ligand Name' in df.columns:
        DRUG_NAME_COL = 'BindingDB Ligand Name'  # 基于您提供的文件片段，使用此列名
        print(f"检测到列名 '{DRUG_NAME_COL}'，将使用此列作为药物名称。")
    elif 'drug_name' in df.columns:
        DRUG_NAME_COL = 'drug_name'
    elif 'Drug' in df.columns:
        DRUG_NAME_COL = 'Drug'

    # 检查所需的列是否存在
    if DRUG_NAME_COL not in df.columns:
        print(
            f"❌ 错误：输入文件 {input_csv} 中未找到药物名称列 '{DRUG_NAME_COL}'。请手动修改脚本中的 DRUG_NAME_COL 变量以匹配您文件的实际列名。")
        return

    if 'Entry_ID' not in df.columns:
        print(f"❌ 错误：输入文件 {input_csv} 中未找到 UniProt ID 列 'Entry_ID'。")
        return

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="抓取 Reactome 通路"):

        # *** 1. 抓取药物名称，替换 SMILES ***
        drug_name = row[DRUG_NAME_COL]

        # 2. 处理 UniProt ID 列表
        entry_ids_raw = row['Entry_ID']

        # 安全处理 NaN 或空值
        if pd.isna(entry_ids_raw) or entry_ids_raw == '':
            continue

        try:
            # 清理和分割 UniProt ID
            entry_ids = str(entry_ids_raw).replace('"', '').split(',')
        except Exception as e:
            print(f"\n⚠️ 处理药物 {drug_name} 的 Entry_ID 列表失败: {e}")
            continue

        for eid in entry_ids:
            eid = eid.strip()
            if not eid:
                continue

            pathways = get_reactome_pathways(eid)

            for pw in pathways:
                results.append({
                    "Drug_Name": drug_name,  # *** 替换输出列为 Drug_Name ***
                    "uniprot_id": eid,
                    "pathway": pw
                })

            # 保持请求间隔，防止被屏蔽
            time.sleep(1)

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_csv, index=False)
    print(f"\n✅ Reactome pathway 数据抓取完成，并保存至：{output_csv}")


# 执行主函数
if __name__ == "__main__":
    # 请确保将 'drugs.csv' 替换为您实际的文件名
    # 例如：process_file(input_csv="筛选掩码结果.csv")
    process_file(input_csv="/home/lkp/cywhome/PITSynergy/PITSynergy/dataset/药物-靶点聚合.csv")
# import pandas as pd
# import requests
# import time
# import sys
# import logging
# from tqdm import tqdm
# from concurrent.futures import ThreadPoolExecutor, as_completed
#
# # 增加默认编码，防止Tqdm在某些环境中出现编码错误
# try:
#     sys.stdout.reconfigure(encoding='utf-8')
# except AttributeError:
#     pass
#
# # 配置日志
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger()
#
#
# def get_reactome_pathways(uniprot_id):
#     """
#     通过 UniProt ID 从 Reactome Content Service 获取相关通路。
#     """
#     url = f"https://reactome.org/ContentService/data/mapping/UniProt/{uniprot_id}/pathways"
#     try:
#         r = requests.get(url, timeout=3)
#         r.raise_for_status()  # 对 4xx/5xx 状态码抛出异常
#
#         if not r.text:
#             return ["N/A"]
#
#         data = r.json()
#         pathways = [entry.get('displayName') for entry in data if isinstance(entry, dict) and entry.get('displayName')]
#         return list(set(pathways)) if pathways else ["N/A"]
#
#     except requests.exceptions.RequestException as e:
#         logger.warning(f"抓取 UniProt ID {uniprot_id} 失败：{e}")
#         return ["N/A"]
#     except Exception as e:
#         logger.error(f"处理 UniProt ID {uniprot_id} 数据失败：{e}")
#         return ["N/A"]
#
#
# def process_file(input_csv="drugs.csv", output_csv="drug_target_pathway_reactome_drugdb子集.csv"):
#     """
#     处理输入 CSV 文件，抓取 Reactome 通路数据并保存。
#     假设输入文件包含两列关键信息：药物名称和靶点 UniProt ID 列表。
#     """
#     try:
#         df = pd.read_csv(input_csv)
#     except FileNotFoundError:
#         logger.error(f"未找到输入文件 {input_csv}")
#         return
#
#     results = []
#
#     # 确认列名
#     DRUG_NAME_COL = 'Drug_Name'
#     if 'BindingDB Ligand Name' in df.columns:
#         DRUG_NAME_COL = 'BindingDB Ligand Name'
#         logger.info(f"检测到列名 '{DRUG_NAME_COL}'，将使用此列作为药物名称。")
#     elif 'drug_name' in df.columns:
#         DRUG_NAME_COL = 'drug_name'
#     elif 'Drug' in df.columns:
#         DRUG_NAME_COL = 'Drug'
#
#     # 检查所需的列是否存在
#     if DRUG_NAME_COL not in df.columns:
#         logger.error(f"未找到药物名称列 '{DRUG_NAME_COL}'。请手动修改脚本中的 DRUG_NAME_COL 变量以匹配实际列名。")
#         return
#     if 'Entry_ID' not in df.columns:
#         logger.error(f"未找到 UniProt ID 列 'Entry_ID'。")
#         return
#
#     def process_entry(drug_name, entry_ids_raw):
#         """处理每个药物的 UniProt ID 并抓取通路"""
#         try:
#             # 清理和分割 UniProt ID
#             entry_ids = str(entry_ids_raw).replace('"', '').split(',')
#             for eid in entry_ids:
#                 eid = eid.strip()
#                 if not eid:
#                     continue
#
#                 pathways = get_reactome_pathways(eid)
#
#                 for pw in pathways:
#                     results.append({
#                         "Drug_Name": drug_name,
#                         "uniprot_id": eid,
#                         "pathway": pw
#                     })
#         except Exception as e:
#             logger.error(f"处理药物 {drug_name} 的 Entry_ID 列表失败: {e}")
#
#     # 使用 ThreadPoolExecutor 来并发抓取数据
#     with ThreadPoolExecutor(max_workers=10) as executor:
#         futures = []
#         for idx, row in tqdm(df.iterrows(), total=len(df), desc="抓取 Reactome 通路"):
#             drug_name = row[DRUG_NAME_COL]
#             entry_ids_raw = row['Entry_ID']
#
#             if pd.isna(entry_ids_raw) or entry_ids_raw == '':
#                 continue
#
#             futures.append(executor.submit(process_entry, drug_name, entry_ids_raw))
#
#         # 等待所有任务完成
#         for future in as_completed(futures):
#             pass
#
#     # 将结果保存到 CSV 文件
#     out_df = pd.DataFrame(results)
#     out_df.to_csv(output_csv, index=False)
#     logger.info(f"Reactome pathway 数据抓取完成，并保存至：{output_csv}")
#
#
# # 执行主函数
# if __name__ == "__main__":
#     process_file(input_csv="药物-靶点聚合.csv")
