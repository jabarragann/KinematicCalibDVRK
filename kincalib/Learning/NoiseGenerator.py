
from __future__ import annotations
from typing import Any, Dict, Tuple
from pathlib import Path
import torch
import numpy as np
from dataclasses import dataclass
from kincalib.Learning.Dataset import Normalizer
from kincalib.Learning.Models import BestMLP2


@dataclass
class NetworkNoiseGenerator:
    """Corrupts a batch of jp measurements with nn. Injected
    noise should minimize the error between actual_jp and measured_jp"""

    model: torch.nn.Module
    input_normalizer: Normalizer
    output_normalizer: Normalizer

    @classmethod
    def create_from_files(
        cls: NetworkNoiseGenerator,
        weights_path: Path,
        input_normalizer_json: Path,
        output_normalizer_json: Path,
        input_features: int,
    ) -> NetworkNoiseGenerator:
        input_normalizer = Normalizer.from_json(input_normalizer_json)
        output_normalizer = Normalizer.from_json(output_normalizer_json)
        model = BestMLP2(input_features)
        model.load_state_dict(torch.load(weights_path, weights_only=True))
        return cls(model, input_normalizer, output_normalizer)

    def corrupt_jp_batch(
        self, input_jp: np.ndarray, prev_measured: np.ndarray, cfg: Dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        if cfg["include_prev_measured"]:
            complete_input = np.hstack((input_jp, prev_measured))
        else:
            complete_input = input_jp

        jp_norm = self.input_normalizer(complete_input)
        jp_norm = torch.tensor(jp_norm.astype(np.float32))
        offset = self.model(jp_norm).detach().numpy()
        offset = self.output_normalizer.reverse(offset)

        # Other way to do it that does not work
        # corrupted_actual2 = actual_jp
        # corrupted_actual2[:, 3:] += correction_offset[:, 3:]

        # correction_offset[:, :3] = 0  # Only apply the offset to last 3 joints
        corrupted_jp = input_jp + offset

        return corrupted_jp, offset

    def inject_errors(
        self, poses1_jp: np.ndarray, prev_measured: np.ndarray, cfg: Dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Corrupts a batch of jp measurements with nn. Injected
        noise should minimize the error between actual_jp/measured_jp
        or measured_jp/setpoint_jp depending on the used network."""

        ## THIS TWO IMPORTS HAVE SOME HEAVY ROS1 DEPENDENCIES.
        from kincalib.Kinematics import DvrkPsmKin_SRC
        from kincalib.Motion.IkUtils import batch_calculate_fk

        poses2_jp_approximate, offset = self.corrupt_jp_batch(
            poses1_jp, prev_measured, cfg
        )
        kin_model = DvrkPsmKin_SRC("classic") 
        poses2_cp_approximate = batch_calculate_fk(poses2_jp_approximate, kin_model)

        return poses2_cp_approximate, poses2_jp_approximate