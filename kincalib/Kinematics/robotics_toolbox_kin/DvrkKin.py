"""
Kinematic model implemented in robotics toolbox library.

"""

from typing import List, Union
from roboticstoolbox.robot import DHRobot
from roboticstoolbox.robot.DHLink import RevoluteMDH, PrismaticMDH
from spatialmath.base import trnorm
from spatialmath import SE3
from numpy import pi
import numpy as np
from kincalib.utils.Logger import Logger

log = Logger(__name__).log
np.set_printoptions(precision=4, suppress=True, sign=" ")


def my_trnorm(a) -> np.ndarray:
    """Required to avoid weird issue with numpy cross product and pylance.

    https://github.com/microsoft/pylance-release/issues/3277#issuecomment-1237782014
    """
    return trnorm(a)


class DvrkPsmKin(DHRobot):
    lrcc = 0.4318
    ltool = 0.4162
    lpitch2yaw = 0.0091
    links = [
        RevoluteMDH(a=0.0, alpha=pi / 2, d=0, offset=pi / 2),
        RevoluteMDH(a=0.0, alpha=-pi / 2, d=0, offset=-pi / 2),
        PrismaticMDH(a=0.0, alpha=pi / 2, theta=0, offset=-lrcc),
        RevoluteMDH(a=0.0, alpha=0.0, d=ltool, offset=0),
        RevoluteMDH(a=0.0, alpha=-pi / 2, d=0, offset=-pi / 2),
        RevoluteMDH(a=lpitch2yaw, alpha=-pi / 2, d=0, offset=-pi / 2),
    ]

    # fmt:off
    # Base transforms based on DVRK console configuration file
    tool_offset = np.array([[ 0.0, -1.0,  0.0,  0.0  ],
                            [ 0.0,  0.0,  1.0,  0.0  ],
                            [-1.0,  0.0,  0.0,  0.0  ],
                            [ 0.0,  0.0,  0.0,  1.0  ]])
    # base_transform = np.array([[  1.0,  0.0,          0.0,          0.20],
    #                           [  0.0, -0.866025404,  0.5,          0.0 ],
    #                           [  0.0, -0.5,         -0.866025404,  0.0 ],
    #                           [  0.0,  0.0,          0.0,          1.0 ]])
    # fmt:on

    def __init__(self, tool_offset=None, base_transform=None):
        if tool_offset is None:
            self.tool_offset = SE3(my_trnorm(DvrkPsmKin.tool_offset))
        else:
            self.tool_offset = SE3(my_trnorm(tool_offset))

        if base_transform is None:
            self.base_transform = SE3(my_trnorm(np.identity(4)))
        else:
            self.base_transform = SE3(my_trnorm(base_transform))

        super(DvrkPsmKin, self).__init__(
            DvrkPsmKin.links, tool=self.tool_offset, base=self.base_transform, name="DVRK PSM"
        )

    def fkine(self, q, **kwargs) -> SE3:
        return super().fkine(q, **kwargs)

    def validate_joints_format(self, q: Union[List, np.ndarray]) -> np.ndarray:
        if isinstance(q, list):
            q = np.array(q)
        if len(q.shape) > 1:
            q = q.squeeze()
        if len(q.shape) > 1:
            raise Exception("q should be a one dimensional array containing the joints")
        if q.shape[0] > len(self.links):
            raise Exception("q should not exceed the number of joints")
        return q

    def fkine_chain(self, q: Union[List, np.ndarray], ignore_base=False, **kwargs) -> SE3:
        """Calculate the forward kinematics for an intermediate frame of the kinematic chain."""
        q = self.validate_joints_format(q)

        T = SE3.Empty()

        # Calculate the forward kinematics one joint at a time.
        Tr = SE3()
        for qi, L in zip(q, self.links):
            Tr *= L.A(qi)

        # Add base transformation if it is not ignored.
        if self.base_transform is not None and not ignore_base:
            Tr = self.base_transform * Tr
        # Add tool transfromation if all the joint values are given.
        if q.shape[0] == len(self.links):
            Tr = Tr * self.tool_offset

        T.append(Tr)
        return T.data[0]

    def estimate_position_from_joints_array(self, joints: np.ndarray) -> np.ndarray:
        """Calculate cartesian pose from joint array from (N,6) array, where `N` is the number
        of joint config to query.

        Parameters
        ----------
        joints : np.ndarray
            array of shape (N,6)

        Returns
        -------
        np.ndarray
            arrayr of shape (N, 3)
        """
        assert joints.shape[1] == 6, "joints array should be of shape (N,6)"
        cartesian_pose = self.fkine(joints)
        poses = []
        for p in cartesian_pose.data:
            poses.append(p[:3, 3].reshape((1, 3)))
        poses = np.concatenate(poses, axis=0)
        return poses


if __name__ == "__main__":
    psm = DvrkPsmKin()
    print(psm)
    j = [pi / 4, 0.0, 0.12, pi / 4, pi / 4, 0]
    # log.info("3rd kinematic chain of the DVRK")
    # log.info(psm.fkine_chain([0.0, 0.0, 0.12]))
    log.info("4th kinematic chain of the DVRK")
    log.info(psm.fkine_chain(j[:4]))
    log.info("dvrk fkine func")
    log.info(psm.fkine(j).data[0])  # .SE3.data[0] returns a ndarray
    log.info("dvrk fkine_chain func")
    log.info(psm.fkine_chain(j))
