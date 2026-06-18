import os
import argparse
import os.path as osp
import time
import pandas as pd
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader
import torch
import tqdm
from model.FEGA import *
from model.DeepSynergy import DeepSynergy
from utils6异构copy import (EarlyStopping,  load_data,
                        set_random_seed, train, validate, get_TensorDataset, get_DataList)
from mol_to_linegraph import *
import json
from model.DeepDDS import DeepDDSGCNNet
import torch.optim as optim
from collections import Counter


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0,
                        help='seed  2 4 8 16 42')
    parser.add_argument('--fold', type=int, default=0,
                        help='fold')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size (default: 64)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--epochs', type=int, default=300,
                        help='maximum number of epochs (default: 500)')
    parser.add_argument('--patience', type=int, default=50,
                        help='patience for earlystopping (default: 50)')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='the path of pretrained_model')
    parser.add_argument('--omic', type=str, default='exp,mut,cn,eff,dep,met',
                        help="omics_data included in this training, separated by commas, for example: exp,mut,cn,eff,dep,met")
    parser.add_argument('--workdir', type=str, default=os.getcwd(),
                        help='workdir of running this model')
    parser.add_argument('--celldataset', type=int, default=2,
                        help='Using which geneset to train the model(1 for 18498g, 2 for 4079g, 3 for 963g)')
    parser.add_argument('--dataset_index', type=int, default=2,
                        help='using which dataset(0 for Oneil,1 for almanac, 2 for OncologyScreen, 3 for DrugcombDB,4 for Drugcomb)')
    parser.add_argument('--dataset_name', type=str, default="indep2-OncologyScreen",
                        help='using which dataset(indep0-oneil, indep1-almanac, indep2-OncologyScreen, indep3-DrugCombDB)')
    parser.add_argument('--cellencoder', type=str, default='cellCNNTrans',
                        help='cell encoder(cellTrans or cellCNNTrans)')
    parser.add_argument('--model', type=str, default='FEGA',
                        help='DeepSynergy FEGA SynergyX DeepDDSGCNNet')
    parser.add_argument('--saved_model', type=str,
                        help='the path of trained_model',
                        default='/home/lkp/cywhome/PITSynergy/experiment_only_one/ best0.3/PISynergy_0_fold_early_stop.pth')
    parser.add_argument('--saved_model_CIMI', type=str,
                        help='the path of trained_model',
                        default='./experiment/20240531_1952/Interpreter_None_fold_early_stop.pth')
    parser.add_argument('--mode', type=str, default='train',
                        help='train or test')
    parser.add_argument('--TIA_strategy', type=str, default='Multiplication',
                        help='Summation , Transposed Multiplication, Multiplication')
    parser.add_argument('--train_interpreter', type=bool, default=True,
                        help='True  False')
    parser.add_argument('--is_regression', type=bool, default=False,
                        help='regression or classifier True  False')
    parser.add_argument('--lr_cimi', type=float, default=0.00001,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight_decay rate')
    parser.add_argument('--alpha', type=int, default=1,
                        help='1 2 3')
    parser.add_argument('--noise', type=float, default=0.1, help='noise')
    return parser.parse_args()
import random



## main 函数 (修改为只跑最后一折)
def main():
    # 解析参数
    args = arg_parse()
    set_random_seed(args.seed)
    SEED=0
    random.seed(SEED)  # 固定Python内置random库
    np.random.seed(SEED)  # 固定numpy
    torch.manual_seed(SEED)  # 固定PyTorch CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)  # 固定PyTorch GPU
        torch.cuda.manual_seed_all(SEED)  # 多GPU场景
        # 禁用cuDNN的随机性（确保卷积操作一致）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # 自动选择设备 (GPU 或 CPU)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"当前设备: {device}")

    # 创建保存当前实验结果的文件夹
    timestamp = time.strftime('%Y%m%d_%H%M', time.localtime())
    expt_folder = osp.join('experiment_only_one_cell/', f'{timestamp}_{args.model}_CV_4_1_only_last_fold')  # 文件夹名标注“只跑最后一折”
    if not os.path.exists(expt_folder):
        os.makedirs(expt_folder)
    print(f"实验结果将保存至: {expt_folder}")

    # 保存环境信息和命令行参数
    print('\n--------参数----------')
    for k_arg in list(vars(args).keys()):
        print('%s: %s' % (k_arg, vars(args)[k_arg]))
    print('\n')

    # 加载数据
    synergy_data = load_data(args)

    if args.mode == "train":
        # 初始化 KFold 进行 5 折交叉验证（划分逻辑不变，确保与原5折的第5折划分一致）
        kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)

        # 列表用于存储最后一折的验证结果
        all_val_best_results = []

        # 关键修改1：获取所有折的索引，仅提取最后一折（索引为4，因为fold从0开始计数）
        # 先将kf.split的结果转为列表，方便索引取值
        all_fold_splits = list(kf.split(synergy_data))
        target_fold_idx = 3 # 最后一折的索引（5折对应索引0-4，第5折是索引4）
        fold, (train_index, val_index) = target_fold_idx, all_fold_splits[target_fold_idx]

        # 强制设置当前折为最后一折
        args.fold = fold
        print(f"\n--- 只运行最后一折（第 {fold + 1}/5 折） ---")

        # 划分当前折叠的训练集和验证集（与原5折的第5折完全一致）
        train_data_df = synergy_data.iloc[train_index]
        val_data_df = synergy_data.iloc[val_index]

        # 将 DataFrame 转换为 DataList
        train_data = get_DataList(train_data_df)
        val_data = get_DataList(val_data_df)

        # 创建数据加载器
        tr_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                   drop_last=True)
        val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                    drop_last=True)

        # 创建一个用于训练集验证的数据加载器
        tr_val_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                       drop_last=True)

        # --- 以下部分与原始代码完全一致 ---
        # hetero_data = torch.load("/home/lkp/cywhome/PITSynergy/PITSynergy/13/hetero_graph_小数据集_ALL_PCA256_药物已有.pt",weights_only=False)
        # with open("/home/lkp/cywhome/PITSynergy/PITSynergy/13/drugid小.json", "r") as f:
        #  drug2id = json.load(f)
        hetero_data = torch.load("/home/lkp/cywhome/PITSynergy/PITSynergy/13/hetero_graph_小数据集_ALL_768_sapbert.pt",weights_only=False)
        with open("/home/lkp/cywhome/PITSynergy/PITSynergy/13/drugid小.json", "r") as f:
                drug2id = json.load(f)

        print("正在为最后一折加载模型...")
        if args.model == "FEGA":
            model = FEGA_Synergy(args=args,hetero_data=hetero_data,drug2id=drug2id).to(device)
            #model = PISynergynet(args=args).to(device)
        elif args.model == "DeepSynergy":
            model = DeepSynergy(args=args).to(device)
        elif (args.model == "DeepDDSGCNNet"):
            model = DeepDDSGCNNet(args=args).to(device)
        else:
            raise ValueError(f"未知模型: {args.model}")

        model.init_weights()
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=args.patience // 2  # 通常是 early_stopping_patience 的一半
        )
        start_epoch = 0
        print("模型已创建并为最后一折初始化。")
        print("数据加载器已准备就绪。")
        start_time = time.time()
        print(f'最后一折（第 {fold + 1} 折）训练正在开始。开始时间:{timestamp}')

        # 为最后一折初始化 EarlyStopping
        stopper = EarlyStopping(mode='higher', metric='roc_auc_score', patience=args.patience,
                                n_fold=fold, folder=expt_folder, model=args.model)

        for epoch in range(start_epoch, args.epochs):
            train_loss = train(model=model, criterion=criterion, opt=optimizer, dataloader=tr_dataloader,
                               device=device, args=args)
            val_auc, val_aupr, val_acc, val_precision, val_recall, val_f1 = \
                validate(model=model, dataloader=val_dataloader, device=device, args=args)

            # 在训练集上进行评估
            tr_auc, tr_aupr, tr_acc, tr_precision, tr_recall, tr_f1 = \
                validate(model=model, dataloader=tr_val_dataloader, device=device, args=args)

            print(f'Epoch {epoch}, 训练损失 {train_loss:.4f}')
            print('训练集结果: auc:{:.4f} aupr:{:.4f} acc:{:.4f} precision:{:.4f} recall:{:.4f} f1:{:.4f}'.format(
                tr_auc, tr_aupr, tr_acc, tr_precision, tr_recall, tr_f1))
            print('验证集结果: auc:{:.4f} aupr:{:.4f} acc:{:.4f} precision:{:.4f} recall:{:.4f} f1:{:.4f}'.format(
                val_auc, val_aupr, val_acc, val_precision, val_recall, val_f1))

            val_list = [epoch, val_auc, val_aupr, val_acc, val_precision, val_recall, val_f1]
            val_data_log = pd.DataFrame([val_list])
            val_epoch_csv_path = f"/val_epoch_metrics_fold_{fold}.csv"
            val_data_log.to_csv(expt_folder + val_epoch_csv_path, mode='a', header=(epoch == 0), index=False)

            scheduler.step(val_auc)

            early_stop = stopper.step(val_auc, model)
            if early_stop:
                print(f'最后一折（第 {fold + 1} 折）触发早期停止！结束训练。')
                break

        # 输出最后一折训练后的性能
        end_time = time.time()
        print(f'最后一折（第 {fold + 1} 折）训练完成！训练时间:{(end_time - start_time) / 60:.2f} 分钟')

        print(f'开始最后一折（第 {fold + 1} 折）的最终评估...')
        stopper.load_checkpoint(model)

        # 使用最佳模型获取验证集上的最终指标
        final_val_auc, final_val_aupr, final_val_acc, final_val_precision, final_val_recall, final_val_f1 = \
            validate(model=model, dataloader=val_dataloader, device=device, args=args)

        print(
            '验证集结果 (最佳模型) for 最后一折（第 {} 折）: auc:{:.4f} aupr:{:.4f} acc:{:.4f} precision:{:.4f} recall:{:.4f} f1:{:.4f}'.format(
                fold + 1, final_val_auc, final_val_aupr, final_val_acc, final_val_precision, final_val_recall,
                final_val_f1))

        # 保存最后一折的最佳验证结果
        all_val_best_results.append(
            [fold, final_val_auc, final_val_aupr, final_val_acc, final_val_precision, final_val_recall,
             final_val_f1])

        # 将最后一折的最佳验证结果保存到CSV文件
        fold_val_df_results = pd.DataFrame([all_val_best_results[-1]],
                                           columns=['Fold', 'AUC', 'AUPR', 'Accuracy', 'Precision', 'Recall', 'F1'])
        fold_val_csv_path = osp.join(expt_folder, f"val_results_last_fold_{fold}.csv")
        fold_val_df_results.to_csv(fold_val_csv_path, index=False)

        # 所有折叠（仅最后一折）完成后，保存汇总结果
        print("\n--- 最后一折训练完成 ---")

        # 汇总最后一折的验证结果
        val_summary_df = pd.DataFrame(all_val_best_results,
                                      columns=['Fold', 'AUC', 'AUPR', 'Accuracy', 'Precision', 'Recall', 'F1'])
        val_summary_df.to_csv(osp.join(expt_folder, "overall_val_metrics_last_fold.csv"), index=False)
        print("\n最后一折验证指标 (最佳模型):")
        print(val_summary_df)
        print("\n最后一折验证指标（无跨折平均，仅单折结果）:")
        print(val_summary_df.drop('Fold', axis=1).iloc[0])  # 输出单折具体数值，而非平均
# def main():
#     # 解析参数
#     args = arg_parse()
#     set_random_seed(args.seed)
#     # 自动选择设备 (GPU 或 CPU)
#     device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
#     print(f"当前设备: {device}")

#     # 创建保存当前实验结果的文件夹
#     timestamp = time.strftime('%Y%m%d_%H%M', time.localtime())
#     expt_folder = osp.join('experiment/', f'{timestamp}_{args.model}_single_run')
#     if not os.path.exists(expt_folder):
#         os.makedirs(expt_folder)
#     print(f"实验结果将保存至: {expt_folder}")

#     # 保存环境信息和命令行参数
#     print('\n--------参数----------')
#     for k_arg in list(vars(args).keys()):
#         print('%s: %s' % (k_arg, vars(args)[k_arg]))
#     print('\n')

#     # 加载数据
#     synergy_data = load_data(args)

#     if args.mode == "train":
#         # 获取数据集总大小
#         total_size = len(synergy_data)
#         # 设定测试集+验证集的大小
#         test_val_size = 261
#         # 计算训练集大小
#         train_size = total_size - test_val_size

#         # 划分训练集和 测试集+验证集
#         train_data_df = synergy_data.iloc[:train_size]
#         test_val_data_df = synergy_data.iloc[train_size:]
        
#         # 将测试集+验证集 再一分为二
#         val_size = len(test_val_data_df) // 2
#         test_size = len(test_val_data_df) - val_size

#         # 划分验证集和测试集
#         val_data_df = test_val_data_df.iloc[:val_size]
#         test_data_df = test_val_data_df.iloc[val_size:]

#         print(f"数据集已划分: 训练集 {len(train_data_df)} 条数据")
#         print(f"验证集 {len(val_data_df)} 条数据, 测试集 {len(test_data_df)} 条数据。")

#         # 将 DataFrame 转换为 DataList
#         train_data = get_DataList(train_data_df)
#         val_data = get_DataList(val_data_df)
#         test_data = get_DataList(test_data_df)

#         # 创建数据加载器
#         tr_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
#         val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)
#         test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)
        
#         # --- 以下部分与原始代码一致 ---
#         hetero_data = torch.load("/home/lkp/cywhome/PITSynergy/PITSynergy/13/hetero_graph_小数据集_ALL_PCA256_药物已有.pt",weights_only=False)
#         with open("/home/lkp/cywhome/PITSynergy/PITSynergy/13/drugid小.json", "r") as f:
#          drug2id = json.load(f)

#         print("正在加载模型...")
#         if args.model == "PISynergy":
#             model = PISynergynet(args=args,hetero_data=hetero_data,drug2id=drug2id).to(device)
#         elif args.model == "DeepSynergy":
#             model = DeepSynergy(args=args).to(device)
#         elif args.model == "SynergyX":
#             model = SynergyxNet(args=args).to(device)
#         elif (args.model == "DeepDDSGCNNet"):
#             model = DeepDDSGCNNet(args=args).to(device)
#         else:
#             raise ValueError(f"未知模型: {args.model}")

#         model.init_weights()
#         criterion = torch.nn.BCELoss()
#         optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
#         scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#             optimizer,
#             mode='max',
#             factor=0.5,
#             patience=args.patience // 2
#         )
#         start_epoch = 0
#         print("模型已创建并初始化。")
#         print("数据加载器已准备就绪。")
#         start_time = time.time()
#         print(f'训练正在开始。开始时间:{timestamp}')

#         # 初始化 EarlyStopping，现在监控验证集
#         stopper = EarlyStopping(mode='higher', metric='roc_auc_score', patience=args.patience,
#                                 n_fold=0, folder=expt_folder, model=args.model)

#         for epoch in range(start_epoch, args.epochs):
#             train_loss = train(model=model, criterion=criterion, opt=optimizer, dataloader=tr_dataloader,
#                                device=device, args=args)
            
#             # 评估模型在验证集上的性能
#             val_auc, val_aupr, val_acc, val_precision, val_recall, val_f1 = \
#                 validate(model=model, dataloader=val_dataloader, device=device, args=args)

#             print(f'Epoch {epoch}, 训练损失 {train_loss:.4f}')
#             print('验证集结果: auc:{:.4f} aupr:{:.4f} acc:{:.4f} precision:{:.4f} recall:{:.4f} f1:{:.4f}'.format(
#                 val_auc, val_aupr, val_acc, val_precision, val_recall, val_f1))
            
#             # 使用验证集上的AUC作为调度器的监控指标
#             scheduler.step(val_auc)

#             # 使用验证集上的AUC作为早停的监控指标
#             early_stop = stopper.step(val_auc, model)
#             if early_stop:
#                 print(f'触发早期停止！结束训练。')
#                 break

#         # 输出训练后的性能
#         end_time = time.time()
#         print(f'训练完成！总训练时间:{(end_time - start_time) / 60:.2f} 分钟')

#         print(f'开始最终评估...')
#         # 加载训练过程中保存的最佳模型 (基于验证集表现)
#         stopper.load_checkpoint(model)

#         # 使用最佳模型获取测试集上的最终指标
#         final_test_auc, final_test_aupr, final_test_acc, final_test_precision, final_test_recall, final_test_f1 = \
#             validate(model=model, dataloader=test_dataloader, device=device, args=args)

#         print(
#             '测试集结果 (最佳模型): auc:{:.4f} aupr:{:.4f} acc:{:.4f} precision:{:.4f} recall:{:.4f} f1:{:.4f}'.format(
#                 final_test_auc, final_test_aupr, final_test_acc, final_test_precision, final_test_recall,
#                 final_test_f1))

#         # 保存测试集的最佳结果
#         test_best_results = [
#             final_test_auc, final_test_aupr, final_test_acc, final_test_precision, final_test_recall, final_test_f1
#         ]

#         # 将测试集结果保存到CSV文件
#         test_results_df = pd.DataFrame([test_best_results],
#                                        columns=['AUC', 'AUPR', 'Accuracy', 'Precision', 'Recall', 'F1'])
#         test_csv_path = osp.join(expt_folder, "final_test_results.csv")
#         test_results_df.to_csv(test_csv_path, index=False)

#         print("\n最终测试结果 (最佳模型) 已保存至 'final_test_results.csv'")


if __name__ == '__main__':
    main()