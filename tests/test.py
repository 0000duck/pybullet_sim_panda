import unittest
import numpy as np
import pybullet as p
from pybullet_sim_panda.kinematics import PandaKinematics

class PandaKinematicsTest(unittest.TestCase):
    def setUp(self):
        uid = p.connect(p.DIRECT) 
        self.panda = PandaKinematics(p, uid)

    def test_get_states(self):
        states = self.panda.get_states()
        self.assertEqual(len(states), 4)

    def test_set_functions(self):
        self.panda.set_home_positions()
        self.panda.open()
        self.panda.set_arm_positions([0,0,0,0,0,0,0])
        
    def test_get_functions(self):
        joints = self.panda.get_arm_positions()
        self.assertEqual(len(joints), 7)
        for i in range(7):
            pos, ori = self.panda.get_link_pose(i)
            self.assertEqual(len(pos), 3)
            self.assertEqual(len(ori), 4)
        joints = np.random.random(7)
        jac = self.panda.get_space_jacobian(joints)
        jac = self.panda.get_space_jacobian()
        self.assertEqual(jac.shape, (6, 7))
        
        jac = self.panda.get_body_jacobian(joints)
        jac = self.panda.get_body_jacobian()
        self.assertEqual(jac.shape, (6, 7))
    
    def test_FK_IK(self):
        curr_joints = self.panda.get_arm_positions()
        joints = np.random.random(7)
        pos, ori = self.panda.FK(joints)
        is_same = np.allclose(curr_joints, self.panda.get_arm_positions())
        self.assertEqual(is_same, True)
        success, joints = self.panda.IK(pos, ori)
        self.assertEqual(len(joints), 7)
        