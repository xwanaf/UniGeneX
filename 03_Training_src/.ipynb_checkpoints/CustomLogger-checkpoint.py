from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only
from torchvision.utils import save_image, make_grid
from pathlib import Path
import logging
import json
import scanpy as sc
import sys
import numpy as np
import pickle
from datasets import Dataset, load_dataset, concatenate_datasets

def configure_logging(logger_name):
    LOG_LEVEL = logging.DEBUG
    log_filename = logger_name+'.log'
    importer_logger = logging.getLogger('importer_logger')
    importer_logger.setLevel(LOG_LEVEL)
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')

    fh = logging.FileHandler(filename=log_filename)
    fh.setLevel(LOG_LEVEL)
    fh.setFormatter(formatter)
    importer_logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(LOG_LEVEL)
    sh.setFormatter(formatter)
    importer_logger.addHandler(sh)
    return importer_logger




class plainLogger(TensorBoardLogger):
    def __init__(self, save_dir, log_step_interval = 100):
        super().__init__(save_dir = save_dir)
#         self.save_path = save_path
        self.loggings = configure_logging(save_dir+'/log')
        self._save_dir = Path(save_dir)
        self.log_step_interval = log_step_interval
        
        
    @property
    def name(self):
        return "plainLogger"

    @property
    def version(self):
        # Return the experiment version, int or str.
        return "0.1"

#     @rank_zero_only
#     def log_hyperparams(self, params, name = 'args'):
#         # params is an argparse.Namespace
#         # your code to record hyperparameters goes here
#         self.loggings.info(params)
#         with open(self._save_dir / f'{name}.json', "w") as f:
#             json.dump(vars(params), f, indent=2)
                

    def dict_to_info(self, metrics):
        epoch = metrics.pop("epoch")
        global_step = metrics.pop("global_step")
        info = f'| epoch {epoch:3d} | global_step {global_step:3d} |'
        if 'lr' in metrics.keys():
            lr = metrics.pop('lr')
            info = info + f"lr {lr:05.20f} |"
        for k, v in metrics.items():
            info = info + f'{k} {v:5.7f} | '
        return info
    
    @rank_zero_only
    def log_metrics_tofile(self, metrics, step, epoch):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        if step % self.log_step_interval == 0:
            self.loggings.info(self.dict_to_info(dict(**{"epoch": epoch, "global_step": step}, **metrics)))
            
    @rank_zero_only
    def log_info(self, info):
        self.loggings.info(info)
    
    @rank_zero_only
    def save_h5ad(self, adata, name):
        adata.write_h5ad(self._save_dir / 'sample_adata' / f'{name}.h5ad')

    @rank_zero_only
    def save_img(self, img, name):
        save_image(img, self._save_dir / 'sample_img' / f'{name}.png')
        
    @rank_zero_only
    def save_array(self, array, name):
        np.save(self._save_dir / 'sample_adata' / f'{name}.npy', array)

    def save_parquet(self, dataset, name, folder_name = 'attn_score'):
        save_dir_tmp = self._save_dir / folder_name
        save_dir_tmp.mkdir(parents=True, exist_ok=True)
        dataset.to_parquet(save_dir_tmp / f'{name}.parquet')

    def save_dict(self, dict, name):
        save_dict_path = self._save_dir / 'attn_score' / f'{name}.json'
        with open(save_dict_path, "w") as file:
            json.dump(dict, file)

    def save_mnn_dict(self, dict, name):
        save_dict_path = self._save_dir / 'mnn_dict' / f'{name}.pkl'
        with open(save_dict_path, "wb") as file:
            pickle.dump(dict, file)
      
    
    
# class plainLogger(Logger):
#     def __init__(self, save_path, log_step_interval = 100):
#         super().__init__()
#         self.save_path = save_path
#         self.loggings = configure_logging(save_path+'/log')
#         self._save_dir = Path(save_path)
#         self.log_step_interval = log_step_interval
        
        
#     @property
#     def name(self):
#         return "plainLogger"

#     @property
#     def version(self):
#         # Return the experiment version, int or str.
#         return "0.1"

#     @rank_zero_only
#     def log_hyperparams(self, params, name = 'args'):
#         # params is an argparse.Namespace
#         # your code to record hyperparameters goes here
#         self.loggings.info(params)
#         with open(self._save_dir / f'{name}.json', "w") as f:
#             json.dump(vars(params), f, indent=2)
                

#     def dict_to_info(self, metrics):
#         epoch = metrics.pop("epoch")
#         global_step = metrics.pop("global_step")
#         info = f'| epoch {epoch:3d} | global_step {global_step:3d} |'
#         if 'lr' in metrics.keys():
#             info = info + f"lr {metrics.pop(lr):05.9f} |"
#         for k, v in metrics.items():
#             info = info + f'{k} {v:5.7f} | '
#         return info
    
#     @rank_zero_only
#     def log_metrics(self, metrics, step, epoch):
#         # metrics is a dictionary of metric names and values
#         # your code to record metrics goes here
#         if step % self.log_step_interval == 0:
#             self.loggings.info(self.dict_to_info(dict(**{"epoch": epoch, "global_step": step}, **metrics)))
            
#     @rank_zero_only
#     def log_info(self, info):
#         self.loggings.info(info)
    
#     @rank_zero_only
#     def save_h5ad(self, adata, name):
#         adata.write_h5ad(self._save_dir / f'{name}.h5ad')

#     @rank_zero_only
#     def save(self):
#         # Optional. Any code necessary to save logger data goes here
#         pass

#     @rank_zero_only
#     def finalize(self, status):
#         # Optional. Any code that needs to be run after training
#         # finishes goes here
#         pass