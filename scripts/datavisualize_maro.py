import nimblephysics as nimble
import numpy as np
import time
from scipy.io import loadmat
from scipy.spatial.transform import Rotation
import os
import threading

class PoseVisualizer:
    def __init__(self, repo_path):
        self.skeleton = None
        self.gui = None
        self.repo_path = repo_path

        self._dof_mapping_setting()
        self._load_skeleton(repo_path)

    def _load_skeleton(self, repo_path):
        maro_model_path = os.path.join(repo_path, "data", "model", "gait3d_pelvis213_Innsbruck_scaled_s01_baseline.osim")
        custom_opensim: nimble.biomechanics.OpenSimFile = nimble.biomechanics.OpenSimParser.parseOsim(maro_model_path)
        self.skeleton: nimble.dynamics.Skeleton = custom_opensim.skeleton

        print("Skeleton Loaded!")
        # print(f"Number of DOFs: {self.skeleton.getNumDofs()}")
        # print("\nJoint DOF names:")
        # dof_names = [self.skeleton.getDofByIndex(i).getName() for i in range(self.skeleton.getNumDofs())]
        # for i, name in enumerate(dof_names):
        #     print(f"  {i}: {name}")

    def _dof_mapping_setting(self):
        #Here because we have converted the pelvis rotation representation from YXZ to ZXY, so the dof names are the same, we do not need to map them again, we only need to fill the missing dof with 0 (use -1 here as a flag)
        self.osim_dof_names = [
            'pelvis_rotation', 'pelvis_obliquity', 'pelvis_tilt', 'pelvis_tx', 'pelvis_ty', 'pelvis_tz',
            'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r', 'ankle_angle_r', 
            'subtalar_angle_r', 'mtp_angle_r', 'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l', 
            'knee_angle_l', 'ankle_angle_l', 'subtalar_angle_l', 'mtp_angle_l', 'lumbar_extension', 
            'lumbar_bending', 'lumbar_rotation', 'arm_flex_r', 'arm_add_r', 'arm_rot_r', 'elbow_flex_r', 
            'pro_sup_r', 'wrist_flex_r', 'wrist_dev_r', 'arm_flex_l', 'arm_add_l', 'arm_rot_l', 
            'elbow_flex_l', 'pro_sup_l', 'wrist_flex_l', 'wrist_dev_l'
        ]
        self.mat_dof_names = [
            'pelvis_rotation', 'pelvis_obliquity', 'pelvis_tilt', 'pelvis_tx', 'pelvis_ty', 'pelvis_tz',
            'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r', 'ankle_angle_r', 
            'subtalar_angle_r', 'mtp_angle_r', 'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l', 
            'knee_angle_l', 'ankle_angle_l', 'subtalar_angle_l', 'mtp_angle_l', 'lumbar_extension', 
            'lumbar_bending', 'lumbar_rotation', 'arm_flex_r', 'arm_add_r', 'arm_rot_r', 'elbow_flex_r', 
            'pro_sup_r', 'arm_flex_l', 'arm_add_l', 'arm_rot_l', 'elbow_flex_l', 'pro_sup_l'
        ]

        self.dof_mapping = []
        for osim_dof in self.osim_dof_names:
            if osim_dof in self.mat_dof_names:
                mat_dof_idx = self.mat_dof_names.index(osim_dof)
                self.dof_mapping.append(mat_dof_idx)
            else:
                self.dof_mapping.append(-1)
        # print(f"DOF Mapping: {self.dof_mapping}")

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
                time.sleep(0.01)
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

def convert_euler_yxz_to_zxy(angles_yxz):
    n_frames = angles_yxz.shape[1]
    angles_zxy = np.zeros_like(angles_yxz)
    
    for i in range(n_frames):
        rot_y = angles_yxz[0, i]  # pelvis_rotation
        rot_x = angles_yxz[1, i]  # pelvis_obliquity  
        rot_z = angles_yxz[2, i]  # pelvis_tilt
        
        r = Rotation.from_euler('YXZ', [rot_y, rot_x, rot_z], degrees=False)
        zxy_angles = r.as_euler('ZXY', degrees=False)
        angles_zxy[:, i] = zxy_angles
    
    return angles_zxy

def mat_visualize(mat_path, repo_path):
    for filename in os.listdir(mat_path):
        if filename.endswith(".mat"):
            if filename.startswith('runningJoints'):
                running_path = os.path.join(mat_path, filename)

    running_mat = loadmat(running_path)
    runningJoints = running_mat['runningJoints'] #(33, 50)
    runningJoints_fix = runningJoints.copy()

    pelvis_yxz = runningJoints[:3, :]  # [rotation_y, obliquity_x, tilt_z]
    pelvis_zxy = convert_euler_yxz_to_zxy(pelvis_yxz)  # [tilt_z, obliquity_x, rotation_y]

    runningJoints_fix[0, :] = pelvis_zxy[0, :]
    runningJoints_fix[1, :] = pelvis_zxy[1, :]
    runningJoints_fix[2, :] = pelvis_zxy[2, :]

    visualizer = PoseVisualizer(repo_path=repo_path)
    visualizer.animate_poses(joint_positions=runningJoints_fix, port=8081)
    

if __name__ == "__main__":
    #Mat visualization
    mat_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results_sim', 'script3D_1.3')
    repo_path = os.path.dirname(os.path.dirname(__file__))
    mat_visualize(mat_path, repo_path)



