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

from thop import profile, clever_format
import time

from thop import profile, clever_format
import time
import torch
from torch.nn.parallel import DataParallel

def profile_model(model, device, input_size=(1, 3, 224, 224), repeat=100):
    """
    返回 Params(M), FLOPs(G), Latency(ms)
    """

    dummy = torch.randn(*input_size).to(device)

    # -------- 1. Params --------
    params = sum(p.numel() for p in model.parameters())

    # -------- 2. FLOPs --------
    macs, _ = profile(model, inputs=(dummy,), verbose=False)
    macs, params_cf = clever_format([macs, params], "%.3f")

    # -------- 3. Latency --------
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(repeat):
            _ = model(dummy)
        torch.cuda.synchronize()
    t1 = time.time()

    latency = (t1 - t0) * 1000 / repeat  # 单位 ms

    return params_cf, macs, latency



def profile_model(model, input_size=(1, 3, 224, 224), repeat=100):
    """
    返回: (params_str, flops_str, latency_ms)
    自动兼容 DataParallel / 单卡模型，并走 return_loss=False 推理分支
    """
    # 1) 如果是 DataParallel，取里面的真正模型
    if isinstance(model, DataParallel):
        model_to_profile = model.module
    else:
        model_to_profile = model

    device = next(model_to_profile.parameters()).device

    # 2) 包一层 Wrapper，让 forward(x) 调用 return_loss=False
    class ClsWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            # 注意：你的 BuildNet 接收的是 (x, return_loss=False)
            return self.m(x, return_loss=False)

    wrapped = ClsWrapper(model_to_profile).to(device)
    wrapped.eval()

    dummy = torch.randn(*input_size).to(device)

    # -------- Params --------
    params = sum(p.numel() for p in model_to_profile.parameters())

    # -------- FLOPs (thop) --------
    macs, _ = profile(wrapped, inputs=(dummy,), verbose=False)
    macs_str, params_str = clever_format([macs, params], "%.3f")

    # -------- Latency --------
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    with torch.no_grad():
        for _ in range(repeat):
            _ = wrapped(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
    t1 = time.time()

    latency_ms = (t1 - t0) * 1000.0 / repeat

    return params_str, macs_str, latency_ms



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


class ClsWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model  # 这里是 BuildNet 或 model.module

    def forward(self, x):
        # 强制走推理分支，不需要 targets
        return self.model(x, return_loss=False)






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
        model = DataParallel(model, device_ids=[args.gpu_id])

    model = init_model(model, data_cfg, device=device, mode='eval')

    # 👇 这里不再传 device，直接调模型
    params, flops, latency = profile_model(model, input_size=(1, 3, 224, 224), repeat=100)
    print("💡 Params:", params)
    print("💡 FLOPs:", flops)
    print(f"💡 Latency: {latency:.3f} ms")


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
        preds,targets, image_paths = [],[],[]
        with tqdm(total=len(test_datas)//data_cfg.get('batch_size')) as pbar:
            for _, batch in enumerate(test_loader):
                images, target, image_path = batch
                print(image_path)
                if not image_path[0].endswith('85ea2db167a761dd0ea509ac20d802dd.jpeg'):
                    continue
                outputs = model(images.to(device),return_loss=False)
                pred_score, pred_label = outputs.topk(1, dim=1)
                top_1_pred = pred_label.reshape(-1).float()
#                 if top_1_pred.cpu().numpy()[0] != target.cpu().numpy()[0]:
#                     print(image_path[0].split('/')[-2])
#                     if not os.path.exists(os.path.join('test_image', weights, image_path[0].split('/')[-2])):
#                         os.makedirs(os.path.join('test_image', weights, image_path[0].split('/')[-2]))
#                     shutil.copyfile(image_path[0], os.path.join('test_image', weights, image_path[0].split('/')[-2], image_path[0].split('/')[-1]))

                image_id = os.path.basename(image_path[0]).split('.')[0]
                fmap_dir = os.path.join(save_dir, 'feature_maps', image_id)
                os.makedirs(fmap_dir, exist_ok=True)

                # 可视化 MSFF+CBAM 输出
                if 'layer7_conv' in feature_maps:
                    visualize_feature_map(feature_maps['layer7_conv'], os.path.join(fmap_dir, 'layer7_conv'))
#                 if 'cbam_fused' in feature_maps:
#                     visualize_feature_map(feature_maps['cbam_fused'], os.path.join(fmap_dir, 'cbam_fused'))
                preds.append(outputs.cpu())
                targets.append(target.cpu())
                image_paths.extend(image_path)
                pbar.update(1)
                break
if __name__ == "__main__":
    # python3 tools/test_image.py models/efficientnetv2/efficientnetv2_s.py
    main()
