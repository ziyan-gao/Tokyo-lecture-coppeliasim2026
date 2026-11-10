"""GR-ConvNet: a pretrained network that proposes parallel-jaw grasps in a depth image.

Kumra, Joshi, Sahin, "Antipodal Robotic Grasping using Generative Residual Convolutional Neural
Network", IROS 2020. Code and weights: https://github.com/skumra/robotic-grasping, BSD 3-Clause,
Copyright (c) 2019 Sulabh Kumra (RIT) and (c) 2018 Douglas Morrison (QUT); the network classes below
follow that code -- see third_party/LICENSE.grconvnet for the full licence.

For every pixel of a 224 x 224 depth crop the network predicts a grasp centred there, seen from
above (4 DoF: image position, rotation about the camera axis, opening width):
    quality  how good a grasp centred at this pixel is (0..1)
    angle    the jaw direction, from cos(2a) and sin(2a) (a grasp and its 180 deg turn are the same)
    width    the opening, in pixels (network output x 150)
We use the depth-only model trained on Jacquard (a synthetic data set: closest to our rendered depth).
The weights are a pickled model; we unpickle them with torch.load(weights_only=True), which only
rebuilds the classes we allow, and keep a plain state_dict afterwards.
"""

import os
import urllib.request

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "models", "grconvnet"
)
NAME = "jacquard-d-grconvnet3-drop0-ch32_epoch_50_iou_0.94"
URL = (
    "https://raw.githubusercontent.com/skumra/robotic-grasping/master/trained-models/"
    "jacquard-d-grconvnet3-drop0-ch32/epoch_50_iou_0.94"
)
SIZE = 224  # the network's input: a 224 x 224 crop
WIDTH_SCALE = 150.0  # the width output is width_px / 150


# ================================================================ network (as in robotic-grasping)
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x_in):
        x = F.relu(self.bn1(self.conv1(x_in)))
        x = self.bn2(self.conv2(x))
        return x + x_in


class GenerativeResnet(nn.Module):
    """Encoder (3 convolutions) -> 5 residual blocks -> decoder (3 transposed convolutions) -> 4 heads."""

    def __init__(self, input_channels=1, output_channels=1, channel_size=32, dropout=False, prob=0.0):
        super().__init__()
        c = channel_size
        self.conv1 = nn.Conv2d(input_channels, c, kernel_size=9, stride=1, padding=4)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c * 2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(c * 2)
        self.conv3 = nn.Conv2d(c * 2, c * 4, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(c * 4)
        self.res1, self.res2, self.res3, self.res4, self.res5 = (ResidualBlock(c * 4, c * 4) for _ in range(5))
        self.conv4 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=4, stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(c * 2)
        self.conv5 = nn.ConvTranspose2d(c * 2, c, kernel_size=4, stride=2, padding=2, output_padding=1)
        self.bn5 = nn.BatchNorm2d(c)
        self.conv6 = nn.ConvTranspose2d(c, c, kernel_size=9, stride=1, padding=4)
        self.pos_output = nn.Conv2d(c, output_channels, kernel_size=2)
        self.cos_output = nn.Conv2d(c, output_channels, kernel_size=2)
        self.sin_output = nn.Conv2d(c, output_channels, kernel_size=2)
        self.width_output = nn.Conv2d(c, output_channels, kernel_size=2)
        self.dropout = dropout
        self.dropout_pos, self.dropout_cos, self.dropout_sin, self.dropout_wid = (nn.Dropout(p=prob) for _ in range(4))

    def forward(self, x_in):
        x = F.relu(self.bn1(self.conv1(x_in)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.res5(self.res4(self.res3(self.res2(self.res1(x)))))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)
        return self.pos_output(x), self.cos_output(x), self.sin_output(x), self.width_output(x)


# ================================================================ weights
def load_model():
    """The pretrained network in eval mode (downloaded once into scripts/data/models/grconvnet/)."""
    state_file = os.path.join(DATA_DIR, NAME + ".state_dict.pt")
    if not os.path.exists(state_file):
        os.makedirs(DATA_DIR, exist_ok=True)
        pickle_file = os.path.join(DATA_DIR, NAME + ".pickle")
        if not os.path.exists(pickle_file):
            print("  downloading GR-ConvNet weights (7.6 MB) ...", flush=True)
            with urllib.request.urlopen(URL, timeout=120) as r, open(pickle_file, "wb") as f:
                f.write(r.read())
        # the pickle names its classes by the original module paths: allow exactly these
        allowed = [
            (GenerativeResnet, "inference.models.grconvnet3.GenerativeResnet"),
            (ResidualBlock, "inference.models.grasp_model.ResidualBlock"),
            nn.Conv2d,
            nn.ConvTranspose2d,
            nn.BatchNorm2d,
            nn.Dropout,
        ]
        with torch.serialization.safe_globals(allowed):
            pickled = torch.load(pickle_file, map_location="cpu", weights_only=True)
        torch.save(pickled.state_dict(), state_file)  # from now on: plain tensors only
    model = GenerativeResnet(input_channels=1, channel_size=32)
    model.load_state_dict(torch.load(state_file, map_location="cpu", weights_only=True))
    return model.eval()


# ================================================================ inference
def predict(model, depth):
    """depth: (224, 224) metres -> quality, angle [rad], width [px] maps (224, 224).
    Preprocessing as in training: subtract the mean depth, clip to [-1, 1]."""
    x = np.clip(depth - depth.mean(), -1, 1).astype(np.float32)
    with torch.no_grad():
        q, cos2, sin2, w = (t.squeeze().numpy() for t in model(torch.from_numpy(x)[None, None]))
    angle = np.arctan2(sin2, cos2) / 2  # the angle is predicted doubled: a and a + 180 deg are one grasp
    # the smoothing of the original post-processing (skimage gaussian: sigma 2, 2, 1)
    q, angle = cv2.GaussianBlur(q, (0, 0), 2.0), cv2.GaussianBlur(angle, (0, 0), 2.0)
    return q, angle, cv2.GaussianBlur(w * WIDTH_SCALE, (0, 0), 1.0)


def best_grasps(q, angle, width, n=5, min_distance=20, threshold=0.2):
    """The n best local maxima of the quality map, at least min_distance px apart (as the original
    detect_grasps). Returns [(row, col, angle, width_px, quality)], best first."""
    peaks = (q == cv2.dilate(q, np.ones((3, 3), np.uint8))) & (q > threshold)
    found = []
    for idx in np.argsort(-q[peaks]):
        r, c = np.argwhere(peaks)[idx]
        if all((r - r2) ** 2 + (c - c2) ** 2 >= min_distance**2 for r2, c2, *_ in found):
            found.append((int(r), int(c), float(angle[r, c]), float(width[r, c]), float(q[r, c])))
            if len(found) == n:
                break
    return found


class GraspModel:
    """Load weights once; reuse the depth-only network for successive RGB-D frames."""

    def __init__(self):
        self.model = load_model()

    def predict(self, depth):
        return predict(self.model, depth)

    best_grasps = staticmethod(best_grasps)
