import os
import gym
import numpy as np
import pybullet_envs
from pybullet_envs.gym_locomotion_envs import HumanoidFlagrunBulletEnv, HumanoidBulletEnv
from pybullet_envs.robot_locomotors import HumanoidFlagrun, WalkerBase

import pkgutil
egl = pkgutil.get_loader('eglRenderer')

def debug_imshow(img):
    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.show()

class DefaultRobot(HumanoidFlagrun):
    def __init__(self):
        self.p = None
        self.camera_width = 256
        self.camera_height = 256
        self.camera_channel = 3
        self.gravity = -9.8
        super().__init__()
        
        # Reset the robot XML file:
        total_joints = 18
        WalkerBase.__init__(self,
                            f'{os.getcwd()}/erl/envs/xmls/humanoid_symmetric.xml',
                            'torso',
                            action_dim=total_joints,
                            # 10 real numbers of basic information
                            # each joint has 2 real numbers (position, velocity)
                            # WxHxC from camera image
                            obs_dim=10 + total_joints*2 + self.camera_width*self.camera_height*self.camera_channel, 
                            power=0.41)

        # Dict observation not supported
        # https://github.com/DLR-RM/stable-baselines3/issues/357
        # self.observation_space = gym.spaces.Dict({
        #     'observation': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
        #     'achieved_goal': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
        #     'desired_goal': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
        #     'torso': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(10,)),
        #     'joints': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(total_joints*2,)),
        #     'camera': gym.spaces.Box(low=0.0, high=1.0, shape=(self.camera_width, self.camera_width, self.camera_channel)),
        # })


    def get_camera_image(self, width=256, height=256):
        """ 
        Get a RGB image from the camera mounted on the robot's face.
        From an example: https://github.com/bulletphysics/bullet3/issues/1616
        """
        if self.p is None and self.scene._p:
            # Lazy load pybullet
            self.p = self.scene._p
            # Turn on the Debug GUI
            self.p.configureDebugVisualizer(self.p.COV_ENABLE_GUI, 1)
            self.p.setGravity(0,0,self.gravity)
            # Precalculate the projection matrix
            fov, aspect, nearplane, farplane = 128, 1.0, 0.01, 100
            self.projection_matrix = self.p.computeProjectionMatrixFOV(fov, aspect, nearplane, farplane)
            
            # Get Index
            self.robot_id = self.parts['head'].bodies[0]
            self.face_id = self.parts['head'].bodyPartIndex
            # Change the head to white, just to make sure the camera is mounted on the right body part
            self.p.changeVisualShape(self.robot_id, self.face_id,rgbaColor=[1,1,1,1])


        com_p, com_o, _, _, _, _ = self.p.getLinkState(self.robot_id, self.face_id, computeForwardKinematics=True)
        rot_matrix = self.p.getMatrixFromQuaternion(com_o)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)
        # Initial vectors
        init_camera_vector = (1, 0, 0) # x-axis
        init_up_vector = (0, 0, 1) # z-axis
        # Rotated vectors
        camera_vector = rot_matrix.dot(init_camera_vector)
        up_vector = rot_matrix.dot(init_up_vector)
        view_matrix = self.p.computeViewMatrix(com_p + 0.1 * camera_vector, com_p + 0.2 * camera_vector, up_vector)
        _, _, rgbPixels, depthPixels, segmentationMaskBuffer = self.p.getCameraImage(width, height, view_matrix, self.projection_matrix)
        return rgbPixels

    def calc_state(self):
        # basic_state: 46 dimensions
        basic_state = super().calc_state()

        # camera_img: 262144 dimensions (256x256x4)
        camera_img = self.get_camera_image()
        # remove the Alpha channel to save space
        # camera_img: 196608 dimensions (256x256x3)
        camera_img = camera_img[:,:,:3] / 255.0
        #(WxHxC) => (CxHxW)
        # camera_img = np.swapaxes(camera_img, 0, 2)
        camera_img = np.rollaxis(camera_img, -1)
        
        # Flatten the observation and reconstruct them in the feature extractor.
        # Hard coded warning. see vae_extractor.py
        # basic_state = [:46] and camera_img = obs[46:].view(3,256,256)
        camera_img = camera_img.flatten()
        obs = np.concatenate([basic_state, camera_img])
        return obs

class DefaultEnv(HumanoidFlagrunBulletEnv):
    def __init__(self, render=False):
        self.robot = DefaultRobot()
        HumanoidBulletEnv.__init__(self, self.robot, render)