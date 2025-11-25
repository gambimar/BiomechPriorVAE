import nimblephysics as nimble
import numpy as np
import time
from scipy.io import loadmat
import os
import threading

class PoseVisualizer:
    def __init__(self):
        self.skeleton = None
        self.gui = None

        self._dof_mapping_setting()
        self._load_skeleton()

    def _dof_mapping_setting(self):
        self.nimble_dof_names = [
            'pelvis_tilt', 'pelvis_list', 'pelvis_rotation', 'pelvis_tx', 'pelvis_ty', 'pelvis_tz',
            'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r', 'ankle_angle_r', 
            'subtalar_angle_r', 'mtp_angle_r', 'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l', 
            'knee_angle_l', 'ankle_angle_l', 'subtalar_angle_l', 'mtp_angle_l', 'lumbar_extension', 
            'lumbar_bending', 'lumbar_rotation', 'arm_flex_r', 'arm_add_r', 'arm_rot_r', 'elbow_flex_r', 
            'pro_sup_r', 'wrist_flex_r', 'wrist_dev_r', 'arm_flex_l', 'arm_add_l', 'arm_rot_l', 
            'elbow_flex_l', 'pro_sup_l', 'wrist_flex_l', 'wrist_dev_l'
        ]
        self.gait3d_dof_names = [
            'pelvis_rotation', 'pelvis_list', 'pelvis_tilt', 'pelvis_tx', 'pelvis_ty', 'pelvis_tz',
            'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r', 'ankle_angle_r', 
            'subtalar_angle_r', 'mtp_angle_r', 'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l', 
            'knee_angle_l', 'ankle_angle_l', 'subtalar_angle_l', 'mtp_angle_l', 'lumbar_extension', 
            'lumbar_bending', 'lumbar_rotation', 'arm_flex_r', 'arm_add_r', 'arm_rot_r', 'elbow_flex_r', 
            'pro_sup_r', 'arm_flex_l', 'arm_add_l', 'arm_rot_l', 'elbow_flex_l', 'pro_sup_l'
        ]

        self.dof_mapping = []
        for nimble_dof in self.nimble_dof_names:
            if nimble_dof in self.gait3d_dof_names:
                gait3d_idx = self.gait3d_dof_names.index(nimble_dof)
                self.dof_mapping.append(gait3d_idx)
            else:
                self.dof_mapping.append(-1)


    def _load_skeleton(self):
        rajagopal_opensim = nimble.RajagopalHumanBodyModel()
        self.skeleton = rajagopal_opensim.skeleton

        print("Skeleton Loaded!")

    #Autofill the missing dof of 33dof model into 37dof
    def dof_autofill(self, joint_position_33):
        if len(joint_position_33) != 33:
            raise ValueError(f"Expected 33dof joint angles, got {len(joint_position_33)}")
        
        joint_position_37 = np.zeros(37)
        for i, mapping_idx in enumerate(self.dof_mapping):
            if mapping_idx != -1:
                joint_position_37[i] = joint_position_33[mapping_idx]
            else:
                #use 0 as default value
                joint_position_37[i] = 0.0
        
        return joint_position_37

    def set_pose(self, joint_position):
        if len(joint_position) == 33:
            joint_position = self.dof_autofill(joint_position)
        elif len(joint_position) != 37:
            raise ValueError(f"Expected 33/37 dof, got {len(joint_position)}")
        
        self.skeleton.setPositions(joint_position)

    def visualize_pose(self, joint_position, port=8080):
        joint_position = np.squeeze(joint_position)
        self.set_pose(joint_position)

        self.gui = nimble.NimbleGUI()
        self.gui.serve(port)

        try:
            while True:
                self.gui.nativeAPI().renderSkeleton(self.skeleton)
                time.sleep
        except KeyboardInterrupt:
            print("Visualization stopped")

    def animate_poses(self, joint_positions, port=8080):
        self.gui = nimble.NimbleGUI()
        self.gui.serve(port)
        frame_time = 1.0/30.0

        try:
            while True:
                for i, positions in enumerate(joint_positions.T):
                    start_time = time.time()

                    self.set_pose(positions)
                    self.gui.nativeAPI().renderSkeleton(self.skeleton)

                    elapsed = time.time() - start_time
                    if elapsed < frame_time:
                        time.sleep(frame_time - elapsed)
        except KeyboardInterrupt:
            print("Animation stopped")

    def inspect_poses(self, joint_positions, port=8080):
        self.gui = nimble.NimbleGUI()
        self.gui.serve(port)

        try:
            while True:
                for i, positions in enumerate(joint_positions.T):
                    self.set_pose(positions)
                    self.gui.nativeAPI().renderSkeleton(self.skeleton)
                    input(f"frame:{i}, press 'Enter'' to continue...")
                input("All frames were displayed, press 'Enter' to the first frame...")

        except KeyboardInterrupt:
            print("Animation stopped")

def mat_visualize(mat_path):
    for filename in os.listdir(mat_path):
        if filename.endswith(".mat"):
            if filename.startswith('standingJoints'):
                standing_path = os.path.join(mat_path, filename)
            elif filename.startswith('runningJoints'):
                running_path = os.path.join(mat_path, filename)
            elif filename.startswith('curvedRunningJoints'):
                curvedrunning_path = os.path.join(mat_path, filename)

    standing_mat = loadmat(standing_path)
    standingJoints = standing_mat['standingJoints'] #(33, 1)
    # row10: knee_angle_r, row17: knee_angle_l
    standingJoints_fix = standingJoints.copy()
    standingJoints_fix[[9, 16], :] = -standingJoints_fix[[9, 16], :]

    # curvedrunning_mat = loadmat(curvedrunning_path)
    # curvedRunningJoints = curvedrunning_mat['curvedRunningJoints'] #(33, 50)
    # curvedRunningJoints_fix = curvedRunningJoints.copy()
    # curvedRunningJoints_fix[[9, 16], :] = -curvedRunningJoints_fix[[9, 16], :]

    running_mat = loadmat(running_path)
    runningJoints = running_mat['runningJoints'] #(33, 50)
    runningJoints_fix = runningJoints.copy()
    runningJoints_fix[[9, 16], :] = -runningJoints_fix[[9, 16], :]


    visualizer = PoseVisualizer()
    visualizer.visualize_pose(joint_position=standingJoints_fix, port=8080)
    visualizer.animate_poses(joint_positions=runningJoints_fix, port=8081)
    # visualizer.animate_poses(joint_positions=curvedRunningJoints_fix, port=8082)
    

if __name__ == "__main__":
    # #Test visualization
    # test_pose_33 = np.random.uniform(-0.5, 0.5, 33)
    # visualizer = PoseVisualizer()
    # visualizer.visualize_pose(joint_position=test_pose_33)

    #Mat visualization
    mat_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'result', 'mat')
    mat_visualize(mat_path)

    # #Latent space interpolation inspection
    # data_path = '../result/model/latent_analysis/'
    # interp_data_path = os.path.join(data_path, 'interpolated_poses.npy')
    # interp_data = np.load(interp_data_path)
    # visualizer = PoseVisualizer()
    # visualizer.inspect_poses(interp_data.T)



