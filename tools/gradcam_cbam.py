import os
import sys
sys.path.insert(0, os.getcwd())

import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image

from utils.train_utils import file2dict
from models.build import BuildNet
from utils.inference import init_model


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        # 防止 __del__ 报错
        self.fwd_handle = None
        self.bwd_handle = None

        # 注册 forward hook
        self.fwd_handle = target_layer.register_forward_hook(self._forward_hook)

        # 注册 backward hook（兼容 PyTorch 版本）
        try:
            self.bwd_handle = target_layer.register_full_backward_hook(self._backward_hook)
        except:
            self.bwd_handle = target_layer.register_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __del__(self):
        if self.fwd_handle is not None:
            self.fwd_handle.remove()
        if self.bwd_handle is not None:
            self.bwd_handle.remove()

    def generate(self, logits, class_idx=None):

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()

        score = logits[0, class_idx]
        score.backward(retain_graph=True)

        A = self.activations[0]
        G = self.gradients[0]

        weights = G.mean(dim=(1, 2))

        cam = torch.zeros_like(A[0])
        for i, w in enumerate(weights):
            cam += w * A[i]

        cam = torch.relu(cam)

        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        return cam.cpu().numpy()


def load_model_from_config(config_path, weight_mode="baseline"):
    import copy

    # 原始完整配置
    raw_cfg = file2dict(config_path)

    # 深拷贝避免被 pop 修改
    model_cfg      = copy.deepcopy(raw_cfg[0])
    train_pipeline = raw_cfg[1]
    val_pipeline   = raw_cfg[2]
    data_cfg       = copy.deepcopy(raw_cfg[3])
    lr_config      = raw_cfg[4]
    optimizer_cfg  = raw_cfg[5]

    # 构建模型（此处 BuildNet 内会 pop 掉 type）
    model = BuildNet(copy.deepcopy(model_cfg))

    # 替换权重
    if weight_mode == "baseline":
        data_cfg['test']['ckpt'] = "logs/EfficientNetV2/new_yuanban/Val_Epoch088-Acc91.406.pth"
    else:
        data_cfg['test']['ckpt'] = "logs/EfficientNetV2/att+mul/Val_Epoch100-Acc91.276.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = init_model(model, data_cfg, device=device, mode='eval')
    return model, device, val_pipeline



def preprocess_image(img_path, val_pipeline):
    img = Image.open(img_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # 更接近你 config 中 mean=127.5/std=127.5 的设定
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])
    tensor = transform(img).unsqueeze(0)  # [1, 3, 224, 224]
    return img, tensor



def overlay_cam_on_image(img_pil, cam, alpha=0.4):
    img = np.array(img_pil.resize((cam.shape[1], cam.shape[0])))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = heatmap * alpha + img * (1 - alpha)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    return overlay_bgr


def main():
    config = "models/efficientnetv2/efficientnetv2_s.py"
    img_path = "test_image/2390289.jpg"

    baseline_model, device, val_pipeline = load_model_from_config(config, weight_mode="baseline")
    backbone = baseline_model.module.backbone if isinstance(baseline_model, nn.DataParallel) else baseline_model.backbone
    target_layer_baseline = backbone.layers[7].conv    # 你之前已经验证过的层

    baseline_cam = GradCAM(baseline_model, target_layer_baseline)

    # CBAM 模型同理
    cbam_model, _, _ = load_model_from_config(config, weight_mode="cbam")
    backbone_cbam = cbam_model.module.backbone if isinstance(cbam_model, nn.DataParallel) else cbam_model.backbone
    target_layer_cbam = backbone_cbam.layers[7].conv

    cbam_cam = GradCAM(cbam_model, target_layer_cbam)


    orig_img, input_tensor = preprocess_image(img_path, val_pipeline)
    input_tensor = input_tensor.to(device)

    # 4) Baseline forward + Grad-CAM（❌ 不要 no_grad）
    input_tensor.requires_grad_(True)           # 开启输入梯度
    logits_base = baseline_model(input_tensor, return_loss=False)  # [1, 47]
    cam_base = baseline_cam.generate(logits_base)

    # 5) CBAM forward + Grad-CAM
    input_tensor_cbam = input_tensor.detach().clone().to(device)
    input_tensor_cbam.requires_grad_(True)

    logits_cbam = cbam_model(input_tensor_cbam, return_loss=False)
    cam_cbam = cbam_cam.generate(logits_cbam)


    os.makedirs("gradcam_vis", exist_ok=True)
    cv2.imwrite("gradcam_vis/baseline.png", overlay_cam_on_image(orig_img, cam_base))
    cv2.imwrite("gradcam_vis/cbam.png", overlay_cam_on_image(orig_img, cam_cbam))

    print("Saved visualizations to gradcam_vis/")


if __name__ == "__main__":
    main()
