"""Torchvision ResNet-18 Chest X-Ray classification and autograd Grad-CAM localization module."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

from medsentinel_triage.base import BaseInferenceModel
from medsentinel_triage.config import XRAY_PNEUMONIA_THRESHOLD, XRAY_RAW_SIZE, resolve_torch_device
from medsentinel_triage.preprocessing import XRayInputType, prepare_xray_batch_tensor, validate_xray_image


class TorchvisionChestXRayGradCAM(BaseInferenceModel[XRayInputType, np.ndarray, Dict[str, Any]]):
    """Transfer-learned ResNet-18 on PneumoniaMNIST with exact PyTorch Autograd Grad-CAM hooks."""

    model_name: str = "chest_xray_resnet18_gradcam"

    def __init__(self, device: Optional[torch.device] = None) -> None:
        super().__init__()
        self.device = device or resolve_torch_device()
        self.model = tv_models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)
        self.model = self.model.to(self.device)
        self._feature_maps: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._register_gradcam_hooks()

    def _register_gradcam_hooks(self) -> None:
        target_layer = self.model.layer4[-1]
        target_layer.register_forward_hook(
            lambda _m, _i, out: setattr(self, "_feature_maps", out)
        )
        target_layer.register_full_backward_hook(
            lambda _m, _gi, go: setattr(self, "_gradients", go[0])
        )

    @staticmethod
    def _prepare_batch(imgs_uint8: np.ndarray) -> torch.Tensor:
        return prepare_xray_batch_tensor(imgs_uint8)

    def validate_input(self, raw_input: XRayInputType) -> None:
        validate_xray_image(raw_input)

    def preprocess(self, raw_input: XRayInputType) -> np.ndarray:
        return validate_xray_image(raw_input)

    def predict_with_gradcam(self, img_28x28: XRayInputType) -> Dict[str, Any]:
        """Runs ResNet-18 forward + backward pass to compute pneumonia probability and 28x28 Grad-CAM hotspot."""
        img_arr = (
            img_28x28
            if isinstance(img_28x28, np.ndarray)
            and img_28x28.shape == XRAY_RAW_SIZE
            and img_28x28.dtype == np.uint8
            else self.preprocess(img_28x28)
        )

        self.model.eval()
        with torch.enable_grad():
            x = self._prepare_batch(img_arr[None, ...]).to(self.device)
            x.requires_grad_(True)
            self.model.zero_grad()
            logit = self.model(x).squeeze()
            prob = float(torch.sigmoid(logit).item())
            logit.backward()

            assert self._gradients is not None and self._feature_maps is not None
            grads = self._gradients[0]
            fmaps = self._feature_maps[0]
            weights = grads.mean(dim=(1, 2))
            cam = torch.relu(torch.sum(weights[:, None, None] * fmaps, dim=0))
            cam_up = F.interpolate(
                cam[None, None, ...],
                size=XRAY_RAW_SIZE,
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            cam_np = cam_up.detach().cpu().numpy()

        if cam_np.max() > 1e-8:
            cam_np = cam_np / cam_np.max()

        r, c = np.unravel_index(np.argmax(cam_np), cam_np.shape)
        side = "Right" if c >= 14 else "Left"
        zone = "Lower Lobe" if r >= 14 else "Upper/Mid Lobe"

        radiology_impression = (
            f"Pulmonary consolidation localized to {side} {zone}"
            if prob >= XRAY_PNEUMONIA_THRESHOLD
            else "Clear lung fields without focal opacity"
        )

        return {
            "pneumonia_probability": round(prob, 4),
            "radiology_impression": radiology_impression,
            "gradcam_hotspot_rc": [int(r), int(c)],
            "gradcam_anatomical_zone": f"{side} {zone}",
        }

    def predict(self, preprocessed_input: np.ndarray) -> Dict[str, Any]:
        return self.predict_with_gradcam(preprocessed_input)
