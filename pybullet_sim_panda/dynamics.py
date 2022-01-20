from pybullet_sim_panda.kinematics import PandaKinematics



class PandaDynamics(PandaKinematics):
    def __init__(self, client, uid, pos=None, ori=None):
        super().__init__(client, uid, pos, ori)
        self._control_mode = "torque"
        self._dof = 7

        self._position_control_gain_p = [0.01,0.01,0.01,0.01,0.01,0.01,0.01]
        self._position_control_gain_d = [1.0,1.0,1.0,1.0,1.0,1.0,1.0]
        self._max_torque = [87.,87.,87.,87.,12.,12.,12.]

        self._target_pos = self._joint_mid_positions[:]
        self._target_torque = [0.]*7
        
        self.reset()

    
    def reset(self):
        self._control_mode = "torque"
        for i in range(self._dof):
            self._target_pos[i] = self._joint_mid_positions[i]
            self._target_torque[i] = 0.
            self._client.resetJointState(self._robot, i, targetValue=self._target_pos[i])
        self.resetController()
    
    def resetController(self):
        self._client.setJointMotorControlArray(bodyUniqueId=self._robot,
                                               jointIndices=self._arm_joints,
                                               controlMode=self._client.VELOCITY_CONTROL,
                                               forces=[0. for i in range(self._dof)])

    def setControlMode(self, mode):
        if mode == "position":
            self._control_mode = "position"
        elif mode == "torque":
            if self._control_mode != "torque":
                self.resetController()
            self._control_mode = "torque"
        else:
            raise Exception('Wrong control mode!')
    
    def setTargetPositions(self, target_pos):
        self._target_pos = target_pos
        self._client.setJointMotorControlArray(bodyUniqueId=self._robot,
                                               jointIndices=self._arm_joints,
                                               controlMode=self._client.POSITION_CONTROL,
                                               targetPositions=self._target_pos,
                                               forces=self._max_torque,
                                               positionGains=self._position_control_gain_p,
                                               velocityGains=self._position_control_gain_d)

    def setTargetTorques(self, target_torque):
        self._target_torque = target_torque
        self._client.setJointMotorControlArray(bodyUniqueId=self._robot,
                                               jointIndices=self._arm_joints,
                                               controlMode=self._client.TORQUE_CONTROL,
                                               forces=self._target_torque)
    
    def setPositionControlGains(self, p=None, d=None):
        if p:
            assert len(p) == 7
            self._position_control_gain_p = p
        if d:
            assert len(d) == 7
            self._position_control_gain_d = d








if __name__ == "__main__":
    print("Panda Dynamics Model")