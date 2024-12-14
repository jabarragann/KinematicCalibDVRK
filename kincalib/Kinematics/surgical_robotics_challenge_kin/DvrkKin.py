
from .import PSMKinematicData#, compute_FK, compute_IK
from .import compute_FK
from .import compute_IK

class DvrkPsmKin:

    def __init__(self, LND_type):
        """ Large needle driver kinematic model.

        Parameters
        ----------
        LND_type : _type_
           either 'classic' or 'SI' for the type of LND tool used. 
        """
        self.kinematics_data = PSMKinematicData(type=LND_type)

    def compute_FK(self, joint_pos, up_to_link):
        return compute_FK(joint_pos, up_to_link, self.kinematics_data) 

    def compute_IK(self, T_7_0):
        return compute_IK(T_7_0, self.kinematics_data)
    