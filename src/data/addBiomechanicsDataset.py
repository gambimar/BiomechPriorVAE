import nimblephysics as nimble
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple
import os
import random

class InputDataKeys:
    # These are the joint quantities for the joints that we are observing
    POS = 'pos'
    VEL = 'vel'

class OutputDataKeys:
    trialname = 'trialname'

target_dof_names = [
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

trial_id_mapping = {
    'walk': 0,  
    'run': 1,
    'sit to stand': 2,
    'stair': 3,
    'gait_any': 4,
    'static': 5,
    'no_name': 6,
    'other': 7
}


class AddBiomechanicsDataset(Dataset):
    stride: int
    data_path: str
    window_size: int
    geometry_folder: str
    device: torch.device
    dtype: torch.dtype
    subject_paths: List[str]
    subjects: List[nimble.biomechanics.SubjectOnDisk]
    windows: List[Tuple[int, int, int]]  # Subject, trial, start_frame
    num_dofs: int
    num_joints: int
    contact_bodies: List[str]
    # For each subject, we store the skeleton and the contact bodies in memory, so they're ready to use with Nimble
    skeletons: List[nimble.dynamics.Skeleton]
    skeletons_contact_bodies: List[List[nimble.dynamics.BodyNode]]
    subject_indices: Dict[str, int]

    def __init__(self,
                 data_path: str,
                 window_size: int,
                 geometry_folder: str,
                 device: torch.device = torch.device('cpu'),
                 dtype: torch.dtype = torch.float32,
                 testing_with_short_dataset: bool = False,
                 stride: int = 1,
                 output_data_format: str = 'last_frame',
                 skip_loading_skeletons: bool = False):
        self.stride = stride
        self.output_data_format = output_data_format
        self.subject_paths = []
        self.subjects = []
        self.window_size = window_size
        self.geometry_folder = geometry_folder
        self.device = device
        self.dtype = dtype
        self.windows = []
        self.contact_bodies = []
        self.skeletons = []
        self.skeletons_contact_bodies = []

        if os.path.isdir(data_path):
            for root, dirs, files in os.walk(data_path):
                for file in files:
                    if file.endswith(".b3d") and "vanasdder" not in file.lower():
                        self.subject_paths.append(os.path.join(root, file))
        else:
            assert data_path.endswith(".b3d")
            self.subject_paths.append(data_path)

        if testing_with_short_dataset:
            self.subject_paths = self.subject_paths[11:12]
        self.subject_indices = {subject_path: i for i, subject_path in enumerate(self.subject_paths)}

        # Walk the folder path, and check for any with the ".b3d" extension (indicating that they are
        # AddBiomechanics binary data files)
        if len(self.subject_paths) > 0:
            # Create a subject object for each file. This will load just the header from this file, and keep that
            # around in memory
            subject = nimble.biomechanics.SubjectOnDisk(
                self.subject_paths[0])
            # Get the number of degrees of freedom for this subject
            self.num_dofs = subject.getNumDofs()
            # Get the number of joints for this subject
            self.num_joints = subject.getNumJoints()
            # Get the contact bodies for this subject, and put them into a consistent order for the dataset
            contact_bodies = subject.getGroundForceBodies()
            for body in contact_bodies:
                if body == 'pelvis':
                    continue
                if body not in self.contact_bodies:
                    self.contact_bodies.append(body)


        for i, subject_path in enumerate(self.subject_paths):
            # Check if the file size is > 10kB
            file_size = os.path.getsize(subject_path)
            if file_size < 10 * 1024:
                print(f"Skipping subject {subject_path} due to small file size ({file_size} bytes)")
                self.subjects.append(None)  # Maintain indexing
                if not skip_loading_skeletons:
                    self.skeletons.append(None)
                    self.skeletons_contact_bodies.append(None)
                continue
            # Add the skeleton to the list of skeletons
            subject = nimble.biomechanics.SubjectOnDisk(subject_path)
            if not skip_loading_skeletons:
                print('Loading skeleton ' + str(i + 1) + '/' + str(
                    len(self.subject_paths)) + f' for subject {subject_path}')
                skeleton = subject.readSkel(subject.getNumProcessingPasses() - 1, geometry_folder)
                self.skeletons.append(skeleton)
                self.skeletons_contact_bodies.append([skeleton.getBodyNode(body) for body in self.contact_bodies])
            self.subjects.append(subject)
            # Prepare the list of windows we can use for training
            if subject is not None:
                num_trials = subject.getNumTrials()
                for trial_index in range(num_trials):
                    # Validate trial index before accessing trial data
                    if trial_index >= num_trials:
                        print(f"Warning: Trial index {trial_index} exceeds number of trials {num_trials} for subject {subject_path}")
                        continue
                    try:
                        trial_length = subject.getTrialLength(trial_index)
                        probably_missing: List[bool] = [reason != nimble.biomechanics.MissingGRFReason.notMissingGRF for reason
                                                        in subject.getMissingGRF(trial_index)]
                    except Exception as e:
                        print(f"Warning: Error accessing trial {trial_index} in subject {subject_path}: {e}")
                        continue
                    for window_start in range(max(trial_length - self.window_size - 1, 0)):
                        if not any(probably_missing[window_start:window_start + self.window_size:self.stride]):
                            assert window_start + self.window_size < trial_length
                            self.windows.append((i, trial_index, window_start))

            if subject is not None:
                skel = subject.readSkel(0,'')
                # Validate correct DOF names across all subjects
                if i > 0 and 'dof_names' in locals():
                    if [skel.getDofByIndex(j).getName() for j in range(skel.getNumDofs())] != dof_names:
                        raise ValueError(f"Subject {subject_path} has different DOF names than the first subject")
                if 'dof_names' not in locals():
                    dof_names = [skel.getDofByIndex(j).getName() for j in range(skel.getNumDofs())]
        
        # Build a mapping from target DOF names to indices in the dataset DOF ordering
        self.target_dof_indices = []
        for target_dof in target_dof_names:
            try:
                original_index = dof_names.index(target_dof)
                self.target_dof_indices.append(original_index)
            except ValueError:
                print(f"Warning: DOF '{target_dof}' not found in dataset DOF names")

        print(f"Dataset initialized with {len(self.subjects)} subjects")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index: int, get_label: bool = True) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], int, int]:
        subject_index, trial, window_start = self.windows[index]

        # Read the frames from disk
        if subject_index >= len(self.subjects) or self.subjects[subject_index] is None:
            input_dict = {
                'pos': torch.zeros((self.window_size // self.stride, len(self.target_dof_indices)), dtype=self.dtype),
                'vel': torch.zeros((self.window_size // self.stride, len(self.target_dof_indices)), dtype=self.dtype),
            }
            return input_dict, {}, subject_index, trial
        subject = self.subjects[subject_index]
        
        # Validate trial index before reading frames
        if trial >= subject.getNumTrials():
            input_dict = {
                'pos': torch.zeros((self.window_size // self.stride, len(self.target_dof_indices)), dtype=self.dtype),
                'vel': torch.zeros((self.window_size // self.stride, len(self.target_dof_indices)), dtype=self.dtype),
            }
            return input_dict, {}, subject_index, trial
        
        frames: nimble.biomechanics.FrameList = subject.readFrames(trial,
                                                                   window_start,
                                                                   self.window_size // self.stride,
                                                                   stride=self.stride,
                                                                   includeSensorData=False,
                                                                   includeProcessingPasses=True)

        # Feed a different frame to the model
        if not (len(frames) == self.window_size // self.stride):
            input_dict = {
                'pos': torch.zeros((self.window_size // self.stride, len(self.target_dof_indices)), dtype=self.dtype),
                'vel': torch.zeros((self.window_size // self.stride, len(self.target_dof_indices)), dtype=self.dtype),
            }
            return input_dict, {}, subject_index, trial
        #f"Expected {self.window_size // self.stride} frames, got {len(frames)}, index {index}, subject {subject_index}, trial {trial}, window_start {window_start}"


        first_passes: List[nimble.biomechanics.FramePass] = [frame.processingPasses[-1] for frame in frames]

        input_dict: Dict[str, torch.Tensor] = {}
        label_dict: Dict[str, torch.Tensor] = {}

        with torch.no_grad():
            input_dict[InputDataKeys.POS] = torch.row_stack([
                torch.tensor(p.pos, dtype=self.dtype).detach()[self.target_dof_indices] for p in first_passes
            ])
            input_dict[InputDataKeys.VEL] = torch.row_stack([
                torch.tensor(p.vel, dtype=self.dtype).detach()[self.target_dof_indices] for p in first_passes
            ])
            
        if get_label:
            trialname = subject.getTrialName(trial)
            if trialname.lower().find('walk') != -1:
                trialname = 'walk'
            elif trialname.lower().find('run') != -1:
                trialname = 'run'
            elif trialname.lower().find('static') != -1:
                trialname = 'static'
            elif trialname.lower().find('gait') != -1:
                trialname = 'gait_any'
            elif trialname.lower().find('stair') != -1:
                trialname = 'stair'
            elif trialname.lower().startswith('t'):
                trialname = 'no_name'
            elif trialname.lower().find('sts') != -1:
                trialname = 'sit to stand'
            else:
                trialname = 'other'
            trial_id = trial_id_mapping[trialname]
            label_dict[OutputDataKeys.trialname] = torch.tensor(trial_id, dtype=torch.long)
            #


        # print(f"{numpy_output_dict[OutputDataKeys.CONTACT_FORCES]=}")
        # ###################################################
        # # Plotting
        # import matplotlib.pyplot as plt
        # x = np.arange(self.window_size)
        # # plotting each row
        # for i in range(len(self.input_dofs)):
        #     # plt.plot(x, numpy_input_dict[InputDataKeys.POS][i, :], label='pos_'+self.input_dofs[i])
        #     plt.plotx, numpy_input_dict[InputDataKeys.VEL][i, :], label='vel_' + self.input_dofs[i])
        #     plt.plot(x, numpy_input_dict[InputDataKeys.ACC][i, :], label='acc_' + self.input_dofs[i])
        # for i in range(3):
        #     plt.plot(x, numpy_input_dict[InputDataKeys.COM_ACC][i, :], label='com_acc_' + str(i))
        # # Add the legend outside the plot
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        # plt.show()
        # ###################################################

        # Return the input and output dictionaries at this timestep, as well as the skeleton pointer

        return input_dict, label_dict, subject_index, trial

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state['subjects']
        del state['skeletons']
        del state['skeletons_contact_bodies']
        return state

    def __setstate__(self, state):
        # Restore instance attributes.
        self.__dict__.update(state)
        self.subjects = []
        print('Unpickling AddBiomechanicsDataset copy in reader worker thread')
        # Create the non picklable SubjectOnDisk objects. Skip loading the skeletons and contact bodies, since these
        # are not used in the reader worker threads.
        for i, subject_path in enumerate(self.subject_paths):
            try:
                # Check file size before loading
                file_size = os.path.getsize(subject_path)
                if file_size < 10 * 1024:
                    print(f"Warning: Skipping subject {subject_path} in worker (file size: {file_size} bytes)")
                    self.subjects.append(None)  # Placeholder to maintain indexing
                    continue
                self.subjects.append(nimble.biomechanics.SubjectOnDisk(subject_path))
            except Exception as e:
                print(f"Warning: Failed to load subject {subject_path} in worker: {e}")
                self.subjects.append(None)  # Placeholder to maintain indexing