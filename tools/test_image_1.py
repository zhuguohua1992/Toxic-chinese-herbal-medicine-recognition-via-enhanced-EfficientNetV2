import os
import sys
sys.path.insert(0, os.getcwd())
import argparse

import copy
import random
import shutil
import numpy as np
import torch.nn as nn
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, precision_score
import matplotlib.pyplot as plt
from PIL import Image
from numpy import mean
from tqdm import tqdm
from terminaltables import AsciiTable

import torch
# import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
import time
import csv

from utils.dataloader import Mydataset, collate
from utils.train_utils import get_info, file2dict, set_random_seed
from models.build import BuildNet
from core.evaluations import evaluate
from utils.inference import init_model


def compute_ece(confidences, correctness, n_bins=15):
    """
    confidences: 1D numpy array, 每个样本的最大 softmax 概率
    correctness: 1D numpy array, 1 表示预测正确, 0 表示预测错误
    """
    import numpy as np

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)

    for i in range(n_bins):
        left, right = bins[i], bins[i+1]
        mask = (confidences > left) & (confidences <= right)
        if not np.any(mask):
            continue
        bin_acc = correctness[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return ece


def coverage_accuracy_curve(confidences, correctness, num_points=20):
    """
    返回不同置信度阈值下的覆盖率 (coverage) 和准确率 (accuracy)
    confidences: numpy [N]
    correctness: numpy [N], 0/1
    """
    import numpy as np

    thresholds = np.linspace(0.0, 1.0, num_points)
    coverages = []
    accuracies = []

    for t in thresholds:
        mask = confidences >= t
        if not np.any(mask):
            continue
        cov = mask.mean()
        acc = correctness[mask].mean()
        coverages.append(cov)
        accuracies.append(acc)

    return thresholds[:len(coverages)], np.array(coverages), np.array(accuracies)



def get_metrics_output(eval_results, metrics_output,classes_names, indexs, APs):
    f = open(metrics_output,'a', newline='')
    writer = csv.writer(f)
    
    """
    输出并保存Accuracy、Precision、Recall、F1 Score、Confusion matrix结果
    """
    p_r_f1 = [['Classes','Precision','Recall','F1 Score', 'Average Precision']]
    for i in range(len(classes_names)):
        data = []
        data.append(classes_names[i])
        data.append('{:.2f}'.format(eval_results.get('precision')[indexs[i]]))
        data.append('{:.2f}'.format(eval_results.get('recall')[indexs[i]]))
        data.append('{:.2f}'.format(eval_results.get('f1_score')[indexs[i]]))
        data.append('{:.2f}'.format(APs[indexs[i]]*100))
        p_r_f1.append(data)
    TITLE = 'Classes Results'
    TABLE_DATA_1 = tuple(p_r_f1)
    table_instance = AsciiTable(TABLE_DATA_1,TITLE)
    #table_instance.justify_columns[2] = 'right'
    print()
    print(table_instance.table)
    writer.writerows(TABLE_DATA_1)
    writer.writerow([])
    print()

    TITLE = 'Total Results'    
    TABLE_DATA_2 = (
    ('Top-1 Acc', 'Top-5 Acc', 'Mean Precision', 'Mean Recall', 'Mean F1 Score'),
    ('{:.2f}'.format(eval_results.get('accuracy_top-1',0.0)), '{:.2f}'.format(eval_results.get('accuracy_top-5',100.0)), '{:.2f}'.format(mean(eval_results.get('precision',0.0))),'{:.2f}'.format(mean(eval_results.get('recall',0.0))),'{:.2f}'.format(mean(eval_results.get('f1_score',0.0)))),
    )
    table_instance = AsciiTable(TABLE_DATA_2,TITLE)
    #table_instance.justify_columns[2] = 'right'
    print(table_instance.table)
    writer.writerows(TABLE_DATA_2)
    writer.writerow([])
    print()


    writer_list     = []
    writer_list.append([' '] + [str(c) for c in classes_names])
    for i in range(len(eval_results.get('confusion'))):
        writer_list.append([classes_names[i]] + [str(x) for x in eval_results.get('confusion')[i]])
    TITLE = 'Confusion Matrix'
    TABLE_DATA_3 = tuple(writer_list)
    table_instance = AsciiTable(TABLE_DATA_3,TITLE)
    print(table_instance.table)
    writer.writerows(TABLE_DATA_3)
    print()

def get_prediction_output(preds,targets,image_paths,classes_names,indexs,prediction_output):
    nums = len(preds)
    f = open(prediction_output,'a', newline='')
    writer = csv.writer(f)
    
    results = [['File', 'Pre_label', 'True_label', 'Success']]
    results[0].extend(classes_names)
    
    for i in range(nums):
        temp = [image_paths[i]]
        pred_label = classes_names[indexs[torch.argmax(preds[i]).item()]]
        true_label = classes_names[indexs[targets[i].item()]]
        success = True if pred_label == true_label else False
        class_score = preds[i].tolist()
        temp.extend([pred_label,true_label,success])
        temp.extend(class_score)
        results.append(temp)
        
    writer.writerows(results)

def plot_ROC_curve(preds, targets, classes_names, savedir):
    rows = len(targets)
    cols = len(preds[0])
    ROC_output = os.path.join(savedir, 'ROC')
    PR_output = os.path.join(savedir, 'P-R')
    os.makedirs(ROC_output)
    os.makedirs(PR_output)
    APs = []
    for j in range(cols):
        gt, pre, pre_score = [], [], []
        for i in range(rows):
            if targets[i].item() == j:
                gt.append(1)
            else:
                gt.append(0)
            
            if torch.argmax(preds[i]).item() == j:
                pre.append(1)
            else:
                pre.append(0)

            pre_score.append(preds[i][j].item())

        # ROC
        ROC_csv_path = os.path.join(ROC_output,classes_names[j] + '.csv')
        ROC_img_path = os.path.join(ROC_output,classes_names[j] + '.png')
        ROC_f = open(ROC_csv_path,'a', newline='')
        ROC_writer = csv.writer(ROC_f)
        ROC_results = []

        FPR,TPR,threshold=roc_curve(targets.tolist(), pre_score, pos_label=j)

        AUC=auc(FPR,TPR)
        
        ROC_results.append(['AUC', AUC])
        ROC_results.append(['FPR'] + FPR.tolist())
        ROC_results.append(['TPR'] + TPR.tolist())
        ROC_results.append(['Threshold'] + threshold.tolist())
        ROC_writer.writerows(ROC_results)

        plt.figure()
        plt.title(classes_names[j] + ' ROC CURVE (AUC={:.2f})'.format(AUC))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.01])
        plt.plot(FPR,TPR,color='g')
        plt.plot([0, 1], [0, 1], color='m', linestyle='--')
        plt.savefig(ROC_img_path)

        # AP (gt为{0,1})
        AP = average_precision_score(gt, pre_score)
        APs.append(AP)

        # P-R
        PR_csv_path = os.path.join(PR_output,classes_names[j] + '.csv')
        PR_img_path = os.path.join(PR_output,classes_names[j] + '.png')
        PR_f = open(PR_csv_path,'a', newline='')
        PR_writer = csv.writer(PR_f)
        PR_results = []
        
        PRECISION, RECALL, thresholds = precision_recall_curve(targets.tolist(), pre_score, pos_label=j)

        PR_results.append(['RECALL'] + RECALL.tolist())
        PR_results.append(['PRECISION'] + PRECISION.tolist())
        PR_results.append(['Threshold'] + thresholds.tolist())
        PR_writer.writerows(PR_results)

        plt.figure()
        plt.title(classes_names[j] + ' P-R CURVE (AP={:.2f})'.format(AP))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.01])
        plt.plot(RECALL,PRECISION,color='g')
        plt.savefig(PR_img_path)

    return APs
        


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--device', help='device used for training. (Deprecated)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed training)')
    args = parser.parse_args()
    return args

def main(): 
    args = parse_args()
    model_cfg,train_pipeline,val_pipeline,data_cfg,lr_config,optimizer_cfg = file2dict(args.config)

    """
    创建评估文件夹、metrics文件、混淆矩阵文件
    """
    dirname = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    save_dir = os.path.join('eval_results',model_cfg.get('backbone').get('type'),dirname)
    metrics_output = os.path.join(save_dir,'metrics_output.csv')
    prediction_output = os.path.join(save_dir,'prediction_results.csv')
    os.makedirs(save_dir)
    
    """
    获取类别名以及对应索引、获取标注文件
    """
    classes_map = 'data_3/annotations.txt' 
    test_annotations    = 'data_3/test_47.txt'
    classes_names, indexs = get_info(classes_map)
    with open(test_annotations, encoding='utf-8') as f:
        test_datas   = f.readlines()
    
    """
    设置各种随机种子确保结果可复现
    """
    set_random_seed(33, False)
    
    """
    生成模型、加载权重
    """
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = BuildNet(model_cfg)
    
    if device != torch.device('cpu'):
        model = DataParallel(model,device_ids=[args.gpu_id])
    model = init_model(model, data_cfg, device=device, mode='eval')
#     print(model.module.backbone)
    """
    制作测试集并喂入Dataloader
    """
#     print(model)
    def print_all_conv_and_linear_layers(model):
        print("📌 All Conv and Linear layers in the model:\n")
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.modules.conv._ConvNd)) or module.__class__.__name__.startswith("Conv2d"):
                print(f"[Conv ] {name:<50} -> {module.__class__.__name__} {tuple(module.weight.shape)}")
            elif isinstance(module, nn.Linear):
                print(f"[Linear] {name:<50} -> Linear {tuple(module.weight.shape)}")

    # 兼容 DataParallel
    model_backbone = model.module.backbone if isinstance(model, nn.DataParallel) else model.backbone

    # 打印所有卷积层和线性层
    print_all_conv_and_linear_layers(model_backbone)
    
    
    
    
    feature_maps = {}  # 全局字典，用于保存 hook 提取的特征图
    
#     def save_cbam_output_hook(name):
#         def hook(module, input, output):
#             feature_maps[name] = output.detach().clone()
#             print(f"[Hook] Saved feature map from {name}: {output.shape}")
#         return hook

#     model_backbone = model.module.backbone if isinstance(model, nn.DataParallel) else model.backbone
#     model_backbone.cbam_fused.register_forward_hook(save_cbam_output_hook("cbam_fused"))

    def register_hook_for_layer7_conv(model):
        # 支持 DataParallel 包裹的模型
        backbone = model.module.backbone if isinstance(model, nn.DataParallel) else model.backbone

        # 确保层存在
        target_layer = backbone.layers[7].conv

        def hook_fn(module, input, output):
            feature_maps['layer7_conv'] = output.detach()
            print(f"[Hook] Feature map from layers.7.conv: {output.shape}")

        # 注册 forward hook
        target_layer.register_forward_hook(hook_fn)
    
    register_hook_for_layer7_conv(model)
    
    def visualize_feature_map(tensor, save_path_prefix="./feature_maps/"):
        tensor = tensor.detach().cpu()
        B, C, H, W = tensor.shape
        os.makedirs(save_path_prefix, exist_ok=True)

        for i in range(min(10000, C)):  # 只保存前16个通道
            img = tensor[0, i]  # 只看第一个样本的第i通道
            img = (img - img.min()) / (img.max() - img.min() + 1e-5)  # 归一化
            img = (img * 255).byte().numpy()
            Image.fromarray(img).save(f"{save_path_prefix}/channel_{i}.png")
            print(f"✅ Saved {save_path_prefix}/channel_{i}.png")
    
    val_pipeline = copy.deepcopy(val_pipeline)
    # 由于val_pipeline是用于推理，此处用做评估还需处理label
    val_pipeline = [data for data in val_pipeline if data['type'] != 'Collect']
    val_pipeline.extend([dict(type='ToTensor', keys=['gt_label']), dict(type='Collect', keys=['img', 'gt_label'])])
    
    test_dataset = Mydataset(test_datas, val_pipeline)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=1, num_workers=data_cfg.get('num_workers'), pin_memory=True, collate_fn=collate)
    
    """
    计算Precision、Recall、F1 Score、Confusion matrix
    """
    weights = 'efficientnetv2_mul_att'
    with torch.no_grad():
        logits_list = []
        targets_list = []

        with tqdm(total=len(test_loader)) as pbar:
            for _, batch in enumerate(test_loader):
                images, target, image_path = batch
                images = images.to(device)
                target = target.to(device)

                # model 输出，一般是 logits（未 softmax）
                outputs = model(images, return_loss=False)   # [B, num_classes]

                logits_list.append(outputs.cpu())
                targets_list.append(target.cpu())

                pbar.update(1)

    # 拼成完整的 [N, C] 和 [N]
    logits = torch.cat(logits_list, dim=0)           # [N, C]
    targets = torch.cat(targets_list, dim=0).view(-1)  # [N]
    
        # 计算 softmax 概率
    probs = torch.softmax(logits, dim=1)         # [N, C]
    max_conf, pred = probs.max(dim=1)            # [N]
    correct = (pred == targets).float()          # [N]

    # 转成 numpy 方便后面计算
    max_conf_np = max_conf.numpy()
    correct_np = correct.numpy()

        # ====== 1) 计算 ECE ======
    ece = compute_ece(max_conf_np, correct_np, n_bins=15)
    print(f"Expected Calibration Error (ECE): {ece:.4f}")
    
        # ====== 2) 阈值-弃权（abstention）分析 ======
    thresholds, coverages, accuracies = coverage_accuracy_curve(max_conf_np, correct_np, num_points=20)

    # 保存成 CSV，方便画图或放到 SI
    calib_csv = os.path.join(save_dir, "coverage_accuracy.csv")
    with open(calib_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold", "coverage", "accuracy"])
        for t, cov, acc in zip(thresholds, coverages, accuracies):
            writer.writerow([f"{t:.3f}", f"{cov:.4f}", f"{acc:.4f}"])

    print(f"Coverage-Accuracy curve saved to: {calib_csv}")


        # 可选：画 coverage vs accuracy 曲线
    plt.figure()
    plt.plot(coverages, accuracies, marker='o')
    plt.xlabel("Coverage")
    plt.ylabel("Accuracy")
    plt.title("Coverage–Accuracy trade-off with confidence thresholding")
    plt.grid(True)
    plt.tight_layout()
    curve_png = os.path.join(save_dir, "coverage_accuracy_curve.png")
    plt.savefig(curve_png, dpi=300)
    plt.close()
    print(f"Coverage–Accuracy curve figure saved to: {curve_png}")

    
    
if __name__ == "__main__":
    # python3 tools/test_image.py models/efficientnetv2/efficientnetv2_s.py
    main()
