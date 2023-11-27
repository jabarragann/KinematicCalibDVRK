""" 
Evaluation metrics to identify best prediction model
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from kincalib.Metrics import MarkdownTable
from kincalib.Motion import calculate_fk, calculate_ik
from kincalib.utils import calculate_orientation_error, calculate_position_error


@dataclass
class AggregatedMetrics:
    metric: str
    N: int
    mean: float
    std: float
    median: float
    max: float
    min: float

    def to_dict(self):
        return dict(
            metric=self.metric,
            N=self.N,
            mean=self.mean,
            std=self.std,
            median=self.median,
            max=self.max,
            min=self.min,
        )


@dataclass
class ExperimentMetrics:
    experiment_name: str
    q1_error: AggregatedMetrics
    q2_error: AggregatedMetrics
    q3_error: AggregatedMetrics
    q4_error: AggregatedMetrics
    q5_error: AggregatedMetrics
    q6_error: AggregatedMetrics
    pos_error: AggregatedMetrics
    ori_error: AggregatedMetrics

    def to_table(self, with_jp=True) -> MarkdownTable:
        md_table = MarkdownTable(
            headers=["metric", "N", "mean", "std", "median", "max", "min"]
        )
        if with_jp:
            md_table.add_data(**self.q1_error.to_dict())
            md_table.add_data(**self.q2_error.to_dict())
            md_table.add_data(**self.q3_error.to_dict())
            md_table.add_data(**self.q4_error.to_dict())
            md_table.add_data(**self.q5_error.to_dict())
            md_table.add_data(**self.q6_error.to_dict())

        md_table.add_data(**self.pos_error.to_dict())
        md_table.add_data(**self.ori_error.to_dict())

        return md_table


@dataclass
class MetricsCalculator:
    experiment_name: str
    input_jp: np.ndarray
    gt_offset: np.ndarray
    pred_offset: np.ndarray

    def __post_init__(self):
        self.validate_shapes()
        self.calculate_joint_error_metrics()
        self.calculate_cartesian_error_metrics()

    def validate_shapes(self):
        assert self.input_jp.shape[0] == self.gt_offset.shape[0]
        assert self.input_jp.shape[0] == self.pred_offset.shape[0]

    def get_metrics_container(self) -> ExperimentMetrics:
        experiment_metrics = ExperimentMetrics(
            experiment_name=self.experiment_name,
            q1_error=self.q1_error,
            q2_error=self.q2_error,
            q3_error=self.q3_error,
            q4_error=self.q4_error,
            q5_error=self.q5_error,
            q6_error=self.q6_error,
            pos_error=self.pos_error_metric,
            ori_error=self.ori_error_metric,
        )
        return experiment_metrics

    def calculate_joint_error_metrics(self):
        joint_error = abs(self.pred_offset - self.gt_offset)

        mean_error = np.mean(joint_error, axis=0)
        std_error = np.std(joint_error, axis=0)
        median_error = np.median(joint_error, axis=0)
        max_error = np.max(joint_error, axis=0)
        min_error = np.min(joint_error, axis=0)

        self.q1_error = AggregatedMetrics(
            metric="q1_error",
            N=joint_error.shape[0],
            mean=mean_error[0],
            std=std_error[0],
            median=median_error[0],
            max=max_error[0],
            min=min_error[0],
        )

        self.q2_error = AggregatedMetrics(
            metric="q2_error",
            N=joint_error.shape[0],
            mean=mean_error[1],
            std=std_error[1],
            median=median_error[1],
            max=max_error[1],
            min=min_error[1],
        )

        self.q3_error = AggregatedMetrics(
            metric="q3_error",
            N=joint_error.shape[0],
            mean=mean_error[2],
            std=std_error[2],
            median=median_error[2],
            max=max_error[2],
            min=min_error[2],
        )

        self.q4_error = AggregatedMetrics(
            metric="q4_error",
            N=joint_error.shape[0],
            mean=mean_error[3],
            std=std_error[3],
            median=median_error[3],
            max=max_error[3],
            min=min_error[3],
        )

        self.q5_error = AggregatedMetrics(
            metric="q5_error",
            N=joint_error.shape[0],
            mean=mean_error[4],
            std=std_error[4],
            median=median_error[4],
            max=max_error[4],
            min=min_error[4],
        )

        self.q6_error = AggregatedMetrics(
            metric="q6_error",
            N=joint_error.shape[0],
            mean=mean_error[5],
            std=std_error[5],
            median=median_error[5],
            max=max_error[5],
            min=min_error[5],
        )

    def calculate_cartesian_error_metrics(self):
        # Only first 6 entries are joint positions
        gt_cartesian = calculate_fk(self.input_jp[:, :6] + self.gt_offset)
        pred_cartesian = calculate_fk(self.input_jp[:, :6] + self.pred_offset)

        self.pos_error = calculate_position_error(gt_cartesian, pred_cartesian)
        self.ori_error = calculate_orientation_error(gt_cartesian, pred_cartesian)

        self.pos_error_metric = AggregatedMetrics(
            "pos_error",
            N=self.pos_error.shape[0],
            mean=np.mean(self.pos_error),
            std=np.std(self.pos_error),
            median=np.median(self.pos_error),
            max=np.max(self.pos_error),
            min=np.min(self.pos_error),
        )

        self.ori_error_metric = AggregatedMetrics(
            "ori_error",
            N=self.ori_error.shape[0],
            mean=np.mean(self.ori_error),
            std=np.std(self.ori_error),
            median=np.median(self.ori_error),
            max=np.max(self.ori_error),
            min=np.min(self.ori_error),
        )
