import nimblephysics as nimble
import numpy as np
from tqdm import tqdm
import os

#Convert original 37dof b3d file into numpy array
class B3DConverter:
    
    def __init__(self, geometry_path):

        self.geometry_path = geometry_path
        self.skeleton = None
        
    def load_subject(self, b3d_path, processing_pass=0):
        subject = nimble.biomechanics.SubjectOnDisk(b3d_path)
        
        self.skeleton = subject.readSkel(
            processingPass=processing_pass,
            geometryFolder=self.geometry_path
        )
        
        print(f"Data joint number: {self.skeleton.getNumJoints()}")
        print(f"Data DOF number: {self.skeleton.getNumDofs()}")
        
        return subject
    
    def convert_single_trial(
        self,
        subject, 
        trial_idx, 
        processing_pass=0,
        start_frame=0,
        num_frames=None
    ):
        
        if num_frames <= 0:
            return np.array([])
        
        frames = subject.readFrames(
            trial=trial_idx,
            includeProcessingPasses=True,
            startFrame=start_frame,
            numFramesToRead=num_frames
        )
        
        #Joint position
        joint_position = []
        for frame in frames:
            positions = frame.processingPasses[processing_pass].pos
            joint_position.append(positions)
            
        return np.array(joint_position)
    
    def convert_data(
        self, 
        subject,
        processing_pass=0
    ):

        all_joint_pos = []
        
        num_trials = subject.getNumTrials()
        print(f"Extracting data of {num_trials} trial...")
        
        for trial_idx in tqdm(range(num_trials), desc="Trial data extraction"):
            trial_length = subject.getTrialLength(trial_idx)
            
            frames_to_read = trial_length
                
            joint_pos = self.convert_single_trial(
                subject=subject,
                trial_idx=trial_idx,
                processing_pass=processing_pass,
                start_frame=0,
                num_frames=frames_to_read
            )

            if len(joint_pos) > 0:
                all_joint_pos.append(joint_pos)

        if all_joint_pos:
            joint_pos_array = np.vstack(all_joint_pos)
        else:
            joint_pos_array = np.array([])

        print(f"Converting complete!")
        print(f"Total frame: {len(joint_pos_array)}")
        print(f"DOF num: {len(joint_pos_array[0]) if len(joint_pos_array) > 0 else 0}")
        
        return joint_pos_array
    
    def save_data(
        self, 
        data,
        output_path
    ):

        np.save(output_path, data)
            
        print(f"Data saved to: {output_path}")
        print(f"Data shape: {data.shape}")

#Convert original 37dof b3d file into 33dof(gait3d_pelvis213) numpy array
class Gait3dB3DConverter:
    
    def __init__(self, geometry_path):

        self.geometry_path = geometry_path
        self.skeleton = None
        #dof setting of gait3d_pelvis213 model
        self.target_dof_names = [
            'pelvis_rotation',
            'pelvis_list',   #which is the 'pelvis_obliquity' in gait3d_pelvis213 model
            'pelvis_tilt',
            'pelvis_tx',
            'pelvis_ty',
            'pelvis_tz',
            'hip_flexion_r',
            'hip_adduction_r',
            'hip_rotation_r',
            'knee_angle_r',
            'ankle_angle_r',
            'subtalar_angle_r',
            'mtp_angle_r',
            'hip_flexion_l',
            'hip_adduction_l',
            'hip_rotation_l',
            'knee_angle_l',
            'ankle_angle_l',
            'subtalar_angle_l',
            'mtp_angle_l',
            'lumbar_extension',
            'lumbar_bending',
            'lumbar_rotation',
            'arm_flex_r',
            'arm_add_r',
            'arm_rot_r',
            'elbow_flex_r',
            'pro_sup_r',
            'arm_flex_l',
            'arm_add_l',
            'arm_rot_l',
            'elbow_flex_l',
            'pro_sup_l'
        ]
        self.dof_names = []
        self.target_dof_indices = []
        
    def load_subject(self, b3d_path, processing_pass=0):
        subject = nimble.biomechanics.SubjectOnDisk(b3d_path)
        
        self.skeleton = subject.readSkel(
            processingPass=processing_pass,
            geometryFolder=self.geometry_path
        )

        #get dof names of generalmodel of nimblephysics
        self.dof_names = [self.skeleton.getDofByIndex(i).getName() 
                         for i in range(self.skeleton.getNumDofs())]
        
        print(f"Data joint number: {self.skeleton.getNumJoints()}")
        print(f"Data DOF number: {self.skeleton.getNumDofs()}")

        #get target dof indices
        self.dof_mapping()
        
        return subject
    
    def dof_mapping(self):
        self.target_dof_indices = []
        missing_dofs = []

        for target_dof in self.target_dof_names:
            try:
                original_index = self.dof_names.index(target_dof)
                self.target_dof_indices.append(original_index)
            except ValueError:
                missing_dofs.append(target_dof)
                print(f"Warning: DOF '{target_dof}' not found in original nimblephysics model")

        print(f"Target DOF number: {len(self.target_dof_names)}")
        print(f"Successfully mapped DOF number: {len(self.target_dof_indices)}")
        if missing_dofs:
            print(f"Missing DOFs: {missing_dofs}")


    def convert_single_trial(
        self,
        subject, 
        trial_idx, 
        processing_pass=0,
        start_frame=0,
        num_frames=None
    ):
        
        if num_frames <= 0:
            return np.array([])
        
        frames = subject.readFrames(
            trial=trial_idx,
            includeProcessingPasses=True,
            startFrame=start_frame,
            numFramesToRead=num_frames
        )
        
        #Joint position (joint angle)
        joint_position = []
        for frame in frames:
            positions = frame.processingPasses[processing_pass].pos
            selective_positions = [positions[idx] for idx in self.target_dof_indices]
            joint_position.append(selective_positions)
            
        return np.array(joint_position)
    
    def convert_data(
        self, 
        subject,
        processing_pass=0
    ):

        all_joint_pos = []
        
        num_trials = subject.getNumTrials()
        print(f"Extracting data of {num_trials} trial...")
        
        for trial_idx in tqdm(range(num_trials), desc="Trial data extraction"):
            trial_length = subject.getTrialLength(trial_idx)
            
            frames_to_read = trial_length
                
            joint_pos = self.convert_single_trial(
                subject=subject,
                trial_idx=trial_idx,
                processing_pass=processing_pass,
                start_frame=0,
                num_frames=frames_to_read
            )

            if len(joint_pos) > 0:
                all_joint_pos.append(joint_pos)

        if all_joint_pos:
            joint_pos_array = np.vstack(all_joint_pos)
        else:
            joint_pos_array = np.array([])

        print(f"Converting complete!")
        print(f"Total frame: {len(joint_pos_array)}")
        print(f"DOF num: {len(joint_pos_array[0]) if len(joint_pos_array) > 0 else 0}")
        
        return joint_pos_array
    
    def save_data(
        self, 
        data,
        output_path
    ):

        np.save(output_path, data)
            
        print(f"Data saved to: {output_path}")
        print(f"Data shape: {data.shape}")


if __name__ == "__main__":
    addb_path = "/home/public/data/AddBiomechanicsDataset/train/With_Arm/"
    b3d_file = "Hammer2013_Formatted_With_Arm/subject01/subject01.b3d"
    output_file = "joint_positions.npy"
    geometry_path = "../data/Geometry/"

    #kinematic pass is 0
    processing_pass = 0

    converter = Gait3dB3DConverter(geometry_path)
    subject = converter.load_subject(os.path.join(addb_path, b3d_file), processing_pass)
    
    joint_pos = converter.convert_data(
        subject, processing_pass
    )
    converter.save_data(
        joint_pos, output_file
    )
    