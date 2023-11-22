"""
This script is borrowed and extended from https://github.com/nkolot/SPIN/blob/master/path_config.py
path configuration
This file contains definitions of useful data stuctures and the paths
for the datasets and data files necessary to run the code.
Things you need to change: *_ROOT that indicate the path to each dataset
"""
import os

# pymaf
pymaf_data_dir = os.path.join(os.path.dirname(__file__),
                              "../../model_save/PYMAF")

SMPL_MEAN_PARAMS = os.path.join(pymaf_data_dir, "smpl_mean_params.npz")
SMPL_MODEL_DIR = os.path.join(pymaf_data_dir, "../SMPL/smpl_related/models/smpl")
MESH_DOWNSAMPLEING = os.path.join(pymaf_data_dir, "mesh_downsampling.npz")

CUBE_PARTS_FILE = os.path.join(pymaf_data_dir, "cube_parts.npy")
JOINT_REGRESSOR_TRAIN_EXTRA = os.path.join(pymaf_data_dir,
                                           "J_regressor_extra.npy")
JOINT_REGRESSOR_H36M = os.path.join(pymaf_data_dir, "J_regressor_h36m.npy")
VERTEX_TEXTURE_FILE = os.path.join(pymaf_data_dir, "vertex_texture.npy")
SMPL_MEAN_PARAMS = os.path.join(pymaf_data_dir, "smpl_mean_params.npz")
CHECKPOINT_FILE = os.path.join(pymaf_data_dir,
                               "pretrained_model/PyMAF_model_checkpoint.pt")

