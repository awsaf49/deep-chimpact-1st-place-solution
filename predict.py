# IMPORT LIBRARIES
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import pandas as pd
import numpy as np
from glob import glob
import json
import yaml
from tqdm import tqdm
import seaborn as sns
sns.set(style='dark')

import tensorflow as tf

import argparse
from utils.config import dict2cfg
from utils.device import get_device
from dataset.dataset import seeding, get_gcspath, build_dataset, build_augmenter, build_decoder
from models.model import build_model
from utils.viz import plot_batch, plot_dist
from utils.submission import path2info, get_shortest_distance
import time
print('> IMPORT COMPLETE')




def predict(CFG):
    print('='*35)
    print('### Inference')
    print('='*35)
    # PREDICTION FOR ALL MODELS
    preds=[]
    for model_idx, (model_paths, dim) in enumerate(CFG.ckpt_cfg):
        start = time.time()
        model_name = model_paths[0].split('/')[-2]
        print(f'> MODEL({model_idx+1}/{len(CFG.ckpt_cfg)}): {model_name} | DIM: {dim}')
        
        # TEST PATHS
        DS_NAME      = f'atcapmihc-peed-1fps-{dim[1]}x{dim[0]}-image-dataset'
        CFG.ds_path = os.path.join('/kaggle/input', DS_NAME)
        print('>> DS_PATH:',CFG.ds_path); print()
        
        # META DATA
        test_df  = pd.read_csv(f'{CFG.ds_path}/test.csv')
        test_df['image_path'] = CFG.ds_path + '/test_images/' + test_df.video_id + '-' + test_df.time.map(lambda x: f'{x:03d}') + '.png'
        test_paths = test_df.image_path.tolist()
        
        # CONFIGURE BATCHSIZE
        if CFG.debug:
            test_paths = test_paths[:100]
        mx_dim = np.sqrt(np.prod(dim))
        if mx_dim>=768:
            CFG.batch_size = CFG.replicas * 16
        elif mx_dim>=640:
            CFG.batch_size = CFG.replicas * 32
        else:
            CFG.batch_size = CFG.replicas * 64
            
        # BUILD DATASET
        dtest = build_dataset(test_paths, batch_size=CFG.batch_size, repeat=True, 
                              shuffle=False, augment=True if CFG.tta>1 else False, cache=False,
                              decode_fn=build_decoder(with_labels=False, target_size=dim, CFG=CFG),
                              augment_fn=build_augmenter(with_labels=False, dim=dim, CFG=CFG),
                              CFG=CFG,
        )
        
        # PREDICTION FOR ONE MODEL -> N FOLDS
        for model_path in sorted(model_paths, key=lambda x: int(x.split('/')[-1].split('-')[-1].split('.')[0])):
            fold = int(model_path.split('/')[-1].split('-')[-1].split('.')[0])
            if fold not in CFG.selected_folds:
                continue
            print(f'Fold-{fold:02d}:')
            with strategy.scope():
                model = tf.keras.models.load_model(model_path, compile=False)
            pred = model.predict(dtest, steps = CFG.tta*len(test_paths)/CFG.batch_size, verbose=1)
            pred = pred[:CFG.tta*len(test_paths),:]
            pred = getattr(np, CFG.agg)(pred.reshape(CFG.tta, len(test_paths), -1), axis=0)
            preds.append(pred)
            pred_df = pd.DataFrame({'image_path':test_paths,
                                    'distance':pred.reshape(-1)})
            pred_df= pred_df.apply(path2info, axis=1)
            pred_df= pred_df.drop(['image_path'], axis=1)
            sub_df = pd.read_csv(f'{CFG.ds_path}/sample_submission.csv')
            del sub_df['distance']
            sub_df = sub_df.merge(pred_df, on=['video_id','time'], how='left')
                
            # SAVE SUBMISSION
            SUB_PATH = os.path.abspath(f'{CFG.output_dir}/submission_{fold:02d}.csv')
            sub_df.to_csv(SUB_PATH,index=False)
            print(f'\n> SUBMISSION_{fold:02d} SAVED TO: {SUB_PATH}')
            
            print()
        end = time.time()
        eta = (end-start)/60
        print(f'>>> TIME FOR {model_name}: {eta:0.2f} min')
    
    print('> PROCESSING SUBMISSION')
    # PROCESSS PREDICTION
    preds = getattr(np, CFG.agg)(preds, axis=0)   
    pred_df = pd.DataFrame({'image_path':test_paths,
                            'distance':preds.reshape(-1)})
    tqdm.pandas(desc='path2info ')
    pred_df= pred_df.progress_apply(path2info, axis=1)
    pred_df= pred_df.drop(['image_path'], axis=1)
    sub_df = pd.read_csv(f'{CFG.ds_path}/sample_submission.csv')
    del sub_df['distance']
    sub_df = sub_df.merge(pred_df, on=['video_id','time'], how='left')
    
    # POST-PROCESSING
    if CFG.rounding:
        tqdm.pandas(desc='rounding ')
        sub_df = sub_df.progress_apply(get_shortest_distance, axis=1)
        
    # SAVE SUBMISSION
    SUB_PATH = os.path.abspath(f'{CFG.output_dir}/submission.csv')
    sub_df.to_csv(SUB_PATH,index=False)
    print(F'\n> SUBMISSION SAVED TO: {SUB_PATH}')
    print(sub_df.head(2))
    
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/deep-chimpact.yaml', help='config file')
    parser.add_argument('--ckpt-cfg', type=str, default='configs/checkpoints.json', help='config file for checkpoint')
    parser.add_argument('--debug', type=int, default=None, help='process only small portion in debug mode')
    parser.add_argument('--gcs-path', type=str, default=None, help='gcs path to the data')
    parser.add_argument('--output-dir', type=str, default='submission', help='output path to save the submission')
    parser.add_argument('--selected_folds', type=int, default=None, nargs='+', help='selected folds')
    parser.add_argument('--tta', type=int, default=None, help='number of TTA')
    opt = parser.parse_args()
    
    # LOADING CONFIG
    CFG_PATH = opt.cfg
    cfg_dict = yaml.load(open(CFG_PATH, 'r'), Loader=yaml.FullLoader)
    CFG      = dict2cfg(cfg_dict) # dict to class
    # print('config:', cfg)
    
    # LOADING CKPT CFG
    CKPT_CFG_PATH = opt.ckpt_cfg
    CKPT_CFG = []
    for base_dir,dim in  json.load(open(CKPT_CFG_PATH, 'r')):
        if '.h5' not in base_dir:
            paths = sorted(glob(os.path.join(base_dir, '*h5')))
        else:
            paths = [base_dir]
        if len(paths)==0:
            raise ValueError('no model found for :',base_dir)
        CKPT_CFG.append([paths, dim])
    CFG.ckpt_cfg = CKPT_CFG
#     print('> CKPTS:',end='')
#     for cfg in CKPT_CFG:
#         print(cfg)
    
    # OVERWRITE
    if opt.debug is not None:
        CFG.debug = opt.debug
    print('> DEBUG MODE:', bool(CFG.debug))
    if opt.selected_folds is not None:
        CFG.selected_folds = opt.selected_folds
    if opt.tta is not None:
        CFG.tta = opt.tta
        
        
    # CREATE SUBMISSION DIRECTORY
    CFG.output_dir = opt.output_dir
    os.system(f'mkdir -p {CFG.output_dir}')
        
        
        
    # CONFIGURE DEVICE
    strategy, device, tpu = get_device()
    CFG.device   = device
    AUTO         = tf.data.experimental.AUTOTUNE
    CFG.replicas = strategy.num_replicas_in_sync
    print(f'> REPLICAS: {CFG.replicas}')   
    
    # SEEDING
    seeding(CFG)
    
    # DS_PATH
    DS_NAME  = f'atcapmihc-peed-1fps-{CFG.ds_size[1]}x{CFG.ds_size[0]}-image-dataset'
    if not tf.io.gfile.exists(f'{CFG.ds_path}/train.csv') and opt.gcs_path: # if provided gcs_path is not available
        CFG.ds_path = get_gcspath(DS_NAME)
    else:
        CFG.ds_path = os.path.join('/kaggle/input', DS_NAME)
    print('> DS_PATH:',CFG.ds_path)
    
    # META DATA
    ## Test Data
    test_df  = pd.read_csv(f'{CFG.ds_path}/test.csv')
    test_df['image_path'] = CFG.ds_path + '/test_images/' + test_df.video_id + '-' + test_df.time.map(lambda x: f'{x:03d}') + '.png'
    # print(test_df.head(2))

    # CHECK FILE FROM DS_PATH
    assert tf.io.gfile.exists(test_df.image_path.iloc[0])
    print('> DS_PATH: OKAY')
    
    # PLOT SOME DATA
    fold_df = test_df.iloc[100:200]
    paths  = fold_df.image_path.tolist()
    ds     = build_dataset(paths, labels=None, cache=False, batch_size=CFG.batch_size*CFG.replicas,
                           repeat=True, shuffle=False, augment=True, CFG=CFG)
    ds = ds.unbatch().batch(20)
    batch = next(iter(ds))
    plot_batch(batch, 5, output_dir=CFG.output_dir)
    
    
    # Prediction
    predict(CFG)