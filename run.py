
from torch.utils.data import TensorDataset, DataLoader
import torch
import argparse
from collections import defaultdict
import models
import DataSet
import pytorch_lightning as pl
import torch.utils.data as tdata
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import pickle
import sandesh

import numpy as np


def run(args):
    from pathlib import Path
    root = Path(args.root)
    data = DataSet.create(name=args.dataset, root=root,roomid=None,userid=None,location=None,orientation=None,receiverid= [1],sampleid=[1,2],mode=args.mode,flag_preload = True)

    train_size = int(0.8*len(data.train))
    valid_size = int(len(data.train)-train_size)
    train_dataset,valid_dataset = torch.utils.data.random_split(data.train,[train_size,valid_size])
    tr_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    va_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=False)
    te_loader = DataLoader(dataset=data.test, batch_size=args.batch_size, shuffle=False)

    if args.dataset == 'aril':
        for i, num_category in enumerate(args.num_categories):
            checkpoint_callback0 = ModelCheckpoint(
                monitor='Validation_Acc',
                filename='sample-{epoch:02d}-{Validation_Acc:.2f}',
                save_top_k=1,
                mode='max',
            )
            model_stl = models.STS_PL(inchannel=args.inchannel,group=args.group,num_categories=num_category,task_index=args.task_index[i])
            trainer = pl.Trainer(log_every_n_steps=1,max_epochs=args.epochs,gpus=1,callbacks=[checkpoint_callback0])
            trainer.fit(model_stl,tr_loader,va_loader )

    if args.dataset == 'csida' and len(args.num_categories) == 3:
        for i, num_category in enumerate(args.num_categories):
            checkpoint_callback0 = ModelCheckpoint(
                monitor='Validation_Acc',
                filename='sample-{epoch:02d}-{Validation_Acc:.2f}',
                save_top_k=1,
                mode='max',
            )
            model_stl = models.STS_PL(inchannel=args.inchannel,group=args.group,num_categories=num_category,task_index=args.task_index[i])
            trainer = pl.Trainer(log_every_n_steps=1,max_epochs=args.epochs,gpus=1,callbacks=[checkpoint_callback0])
            trainer.fit(model_stl,tr_loader,va_loader )
    
    if args.dataset == 'widar' and len(args.num_categories) == 3:
        for i, num_category in enumerate(args.num_categories):
            checkpoint_callback0 = ModelCheckpoint(
                monitor='Validation_Acc',
                filename='sample-{epoch:02d}-{Validation_Acc:.2f}',
                save_top_k=1,
                mode='max',
            )
            model_stl = models.STS_PL(inchannel=args.inchannel,group=args.group,num_categories=num_category,task_index=args.task_index[i])
            trainer = pl.Trainer(log_every_n_steps=1,max_epochs=args.epochs,gpus=1,callbacks=[checkpoint_callback0])
            trainer.fit(model_stl,tr_loader,va_loader )

    model = models.NMTS_PL(inchannel=args.inchannel,group=args.group,num_categories=args.num_categories,
                                task_index=args.task_index)
    checkpoint_callback1 = ModelCheckpoint(
            monitor='Validation_Acc',
            filename='sample-{epoch:02d}-{Validation_Acc:.2f}',
            save_top_k=1,
            mode='max',
        )
    trainer = pl.Trainer(log_every_n_steps=1,max_epochs=args.epochs,gpus=1,callbacks=[checkpoint_callback1])
    trainer.fit(model,tr_loader,va_loader )

    checkpoint_callback2 = ModelCheckpoint(
            monitor='Validation_Acc',
            filename='sample-{epoch:02d}-{Validation_Acc:.2f}',
            save_top_k=1,
            mode='max',
        )
    model = models.UMTS_PL(inchannel=args.inchannel, group=args.group, num_categories=args.num_categories,
                                    task_index=args.task_index)
    trainer = pl.Trainer(log_every_n_steps=1,max_epochs=args.epochs,gpus=1,callbacks=[checkpoint_callback2])
    trainer.fit(model,tr_loader,va_loader )

    checkpoint_callback3 = ModelCheckpoint(
            monitor='Validation_Acc',
            filename='sample-{epoch:02d}-{Validation_Acc:.2f}',
            save_top_k=1,
            mode='max',
        )
    model = models.KDMTS_PL(inchannel=args.inchannel, group=args.group, num_categories=args.num_categories,
                                    task_index=args.task_index,checkpoint_path=args.checkpoint_path, lamda= args.MTLKD_LambdaParameter)
    trainer = pl.Trainer(log_every_n_steps=1,max_epochs=args.epochs,gpus=1,callbacks=[checkpoint_callback3])
    trainer.fit(model,tr_loader,va_loader )

    checkpoint_callback4 = ModelCheckpoint(
            monitor='Validation_Acc',
            filename='sample-{epoch:02d}-{Validation_Acc:.2f}',
            save_top_k=1,
            mode='max',
        )
    model = models.KDMTS_RA_PL(inchannel=args.inchannel, group=args.group, num_categories=args.num_categories,
                                    task_index=args.task_index,checkpoint_path=args.checkpoint_path,lamda= args.MTLKDRe_LambdaParameter)
    trainer = pl.Trainer(log_every_n_steps=1,max_epochs=args.epochs,gpus=1,callbacks=[checkpoint_callback4])
    trainer.fit(model,tr_loader,va_loader )

    checkpoint_callback5 = ModelCheckpoint(
            monitor='Validation_Acc',
            filename='sample-{epoch:02d}-{Validation_Acc:.2f}',
            save_top_k=1,
            mode='max',
        )
    model_final =models.Wimuse_PL(inchannel=args.inchannel, group=args.group, num_categories=args.num_categories,
                                    task_index=args.task_index,checkpoint_path=args.checkpoint_path,
                                    lamda= args.MTLKDRePlus_LambdaParameter,
                                    temperature = args.MTLKDRePlus_Temperature)
    trainer = pl.Trainer(log_every_n_steps=1,max_epochs=args.epochs,gpus=1,callbacks=[checkpoint_callback5])
    trainer.fit(model_final,tr_loader,va_loader )

    for i in range(len(args.task_index)):
            comfusion_matrix = torch.stack(model_final.ComfusionMatrix_alltasks[i])
            comfusion_matrix_np = comfusion_matrix.numpy()
            np.save(os.path.join(trainer.logger.log_dir, 'ComfusionMatrix_'+str(i)) ,comfusion_matrix_np)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WiMu: multi-task learning in wifi-based perception')
    # the configure for dataset
    parser.add_argument('--dataset', type=str, default='aril', help="the name of dataset: aril,csida,widar")
    parser.add_argument('--root', type=str, default='\data',help="the storage path of dataset")
    parser.add_argument('--mode', type=str, default='amplitude', help="the input type of CSI")

    # the configure for task
    parser.add_argument('--num_categories', type=int, default=6, help="num_categories")
    parser.add_argument('--task_index', type=int, default=1, help="which task for testing: 1,2,.....")

    # the configure for models
    parser.add_argument('--inchannel', type=int, default=52, help="input channels")
    parser.add_argument('--group', type=int, default=1, help="group of convolution")
    parser.add_argument('--checkpoint_path', type=str, default=None, help="pretrained model for Knowledge Distillation")

    parser.add_argument('--MTLKD_LambdaParameter', type=float, default=1.0, help="the distance loss weight for KDMTS")
    parser.add_argument('--MTLKDRe_LambdaParameter', type=float, default=1.0, help="the distance loss weight for KDMTS_RA")
    parser.add_argument('--MTLKDRePlus_LambdaParameter', type=float, default=1.0, help="the distance loss weight for Wimuse")
    parser.add_argument('--MTLKDRePlus_Temperature', type=float, default=1.0, help="the logits distillation temperature of Wimuse")

    # the configure for training procession
    parser.add_argument('--epochs', type=int, default=200, help="epochs of training")
    parser.add_argument('--batch_size', type=int, default=30, help="batchsize")

    args = parser.parse_args()
    

    configuration = [
        # pretrained model paths:
        # gesture task in aril: 'state_dics/aril_task1.pth'
        # location task in aril: 'state_dics/aril_task2.pth'

        # gesture task in csida: 'state_dics/csida_task1.pth'
        # location task in csida: 'state_dics/csida_task2.pth'
        # user_identify task in csida: 'state_dics/csida_task3.pth'

        # gesture task in widar: 'state_dics/widar_task1.pth'
        # location task in widar: 'state_dics/widar_task2.pth'
        # user_identify task in widar: 'state_dics/widar_task3.pth'

        # {
        #     # gesture + location task in aril
        #     'dataset': 'aril', 'root': '/media/yk/Samsung_T5/ARIL', 'mode': 'amplitude',
        #     'inchannel': 52, 'group': 1, 'num_categories': [6,16], 'task_index': [1,2],
        #     'epochs': 400, 'batch_size': 8,  
        #     'MTLKD_LambdaParameter': 4.0, 'MTLKDRe_LambdaParameter': 4.0,
        #     'MTLKDRePlus_LambdaParameter': 8.0, 'MTLKDRePlus_Temperature':8.0,
        #     'checkpoint_path':['state_dics/aril_task1.pth', 'state_dics/aril_task2.pth',]
        # },

        # {
        #     # gesture + location task in csida
        #     'dataset': 'csida', 'root': '/media/yk/Samsung_T5/csi_301', 'mode': 'amplitude',
        #     'inchannel': 342, 'group': 3, 'num_categories': [6,5], 'task_index': [1,2],
        #     'epochs': 500, 'batch_size': 64,
        #     'MTLKD_LambdaParameter': 2.0, 'MTLKDRe_LambdaParameter': 2.0,
        #     'MTLKDRePlus_LambdaParameter': 2.0, 'MTLKDRePlus_Temperature':2.0,
        #     'checkpoint_path':['state_dics/csida_task1.pth', 'state_dics/csida_task2.pth',]
        # },
        # {
        #     # location + user_identify task in csida
        #     'dataset': 'csida', 'root': '/media/yk/Samsung_T5/csi_301', 'mode': 'amplitude',
        #     'inchannel': 342, 'group': 3, 'num_categories': [5,5], 'task_index': [2,3],
        #     'epochs': 500, 'batch_size': 64, 
        #     'MTLKD_LambdaParameter': 2.0, 'MTLKDRe_LambdaParameter': 2.0,
        #     'MTLKDRePlus_LambdaParameter': 2.0, 'MTLKDRePlus_Temperature':2.0,
        #     'checkpoint_path':['state_dics/csida_task2.pth', 'state_dics/csida_task3.pth',]
        # },
        # {
        #     # user_identify + gesture task in csida
        #     'dataset': 'csida', 'root': '/media/yk/Samsung_T5/csi_301', 'mode': 'amplitude',
        #     'inchannel': 342, 'group': 3, 'num_categories': [5,6], 'task_index': [3,1],
        #     'epochs': 500, 'batch_size': 64, 
        #     'MTLKD_LambdaParameter': 2.0, 'MTLKDRe_LambdaParameter': 2.0,
        #     'MTLKDRePlus_LambdaParameter': 2.0, 'MTLKDRePlus_Temperature':2.0,
        #     'checkpoint_path':['state_dics/csida_task3.pth',
        #       'state_dics/csida_task1.pth',]
        # },

        # {
        #     # gesture + location task in widar
        #     'dataset': 'widar', 'root': '/media/yk/Samsung_T5/Widar3.0ReleaseData/np_f2/Widar', 'mode': 'amplitude',
        #     'inchannel': 90, 'group': 3, 'num_categories': [6,5], 'task_index': [1,2],
        #     'epochs': 500, 'batch_size': 64, 
        #     'MTLKD_LambdaParameter': 2.0, 'MTLKDRe_LambdaParameter': 2.0,
        #     'MTLKDRePlus_LambdaParameter': 2.0, 'MTLKDRePlus_Temperature':2.0,
        #     'checkpoint_path':['state_dics/widar_task1.pth',
        #       'state_dics/widar_task2.pth',]
        # },
        # {
        #     # location +user_identify task in widar
        #     'dataset': 'widar', 'root': '/media/yk/Samsung_T5/Widar3.0ReleaseData/np_f2/Widar', 'mode': 'amplitude',
        #     'inchannel': 90, 'group': 3, 'num_categories': [5,17], 'task_index': [2,3],
        #     'epochs': 500, 'batch_size': 64, 
        #     'MTLKD_LambdaParameter': 2.0, 'MTLKDRe_LambdaParameter': 2.0,
        #     'MTLKDRePlus_LambdaParameter': 2.0, 'MTLKDRePlus_Temperature':2.0,
        #     'checkpoint_path':['state_dics/widar_task2.pth',
        #       'state_dics/widar_task3.pth',]
        # },
        # {
        #     # user_identify + gesture task in widar
        #     'dataset': 'widar', 'root': '/media/yk/Samsung_T5/Widar3.0ReleaseData/np_f2/Widar', 'mode': 'amplitude',
        #     'inchannel': 90, 'group': 3, 'num_categories': [17,6], 'task_index': [3,1],
        #     'epochs': 500, 'batch_size': 64,
        #     'MTLKD_LambdaParameter': 2.0, 'MTLKDRe_LambdaParameter': 2.0,
        #     'MTLKDRePlus_LambdaParameter': 2.0, 'MTLKDRePlus_Temperature':2.0,
        #     'checkpoint_path':['state_dics/widar_task3.pth',
        #       'state_dics/widar_task1.pth',]
        # },
       
        # {
        #     # gesture + location + user_identify task in csida
        #     'dataset': 'csida', 'root': '/media/yk/Samsung_T5/csi_301', 'mode': 'amplitude',
        #     'inchannel': 342, 'group': 3, 'num_categories': [6,5,5], 'task_index': [1,2,3],
        #     'epochs': 400, 'batch_size': 64, 
        #     'MTLKD_LambdaParameter': 2.0, 'MTLKDRe_LambdaParameter': 2.0,
        #     'MTLKDRePlus_LambdaParameter': 1.25, 'MTLKDRePlus_Temperature':1.25,
        #     'checkpoint_path':['state_dics/csida_task1.pth',
        #       'state_dics/csida_task2.pth',
        #       'state_dics/csida_task3.pth',]
        # },
        # {
        #     # gesture + location + user_identify task in csida
        #     'dataset': 'csida', 'root': '/media/yk/Samsung_T5/csi_301', 'mode': 'amplitude',
        #     'inchannel': 342, 'group': 3, 'num_categories': [6,5,5], 'task_index': [1,2,3],
        #     'epochs': 400, 'batch_size': 64, 
        #     'MTLKD_LambdaParameter': 2.0, 'MTLKDRe_LambdaParameter': 2.0,
        #     'MTLKDRePlus_LambdaParameter': 1.5, 'MTLKDRePlus_Temperature':1.5,
        #     'checkpoint_path':['state_dics/csida_task1.pth',
        #       'state_dics/csida_task2.pth',
        #       'state_dics/csida_task3.pth',]
        # },
        # {
        #     # gesture + location + user_identify task in csida
        #     'dataset': 'csida', 'root': '/media/yk/Samsung_T5/csi_301', 'mode': 'amplitude',
        #     'inchannel': 342, 'group': 3, 'num_categories': [6,5,5], 'task_index': [1,2,3],
        #     'epochs': 400, 'batch_size': 64, 
        #     'MTLKD_LambdaParameter': 2.0, 'MTLKDRe_LambdaParameter': 2.0,
        #     'MTLKDRePlus_LambdaParameter': 1.75, 'MTLKDRePlus_Temperature':1.75,
        #     'checkpoint_path':['state_dics/csida_task1.pth',
        #       'state_dics/csida_task2.pth',
        #       'state_dics/csida_task3.pth',]
        # },
        # {
        #     # gesture + location + user_identify task in csida
        #     'dataset': 'csida', 'root': '/media/yk/Samsung_T5/csi_301', 'mode': 'amplitude',
        #     'inchannel': 342, 'group': 3, 'num_categories': [6,5,5], 'task_index': [1,2,3],
        #     'epochs': 400, 'batch_size': 64, 
        #     'MTLKD_LambdaParameter': 2.0, 'MTLKDRe_LambdaParameter': 2.0,
        #     'MTLKDRePlus_LambdaParameter': 2.25, 'MTLKDRePlus_Temperature':2.25,
        #     'checkpoint_path':['state_dics/csida_task1.pth',
        #       'state_dics/csida_task2.pth',
        #       'state_dics/csida_task3.pth',]
        # },
        # {
        #     # gesture + location + user_identify task in csida
        #     'dataset': 'csida', 'root': '/media/yk/Samsung_T5/csi_301', 'mode': 'amplitude',
        #     'inchannel': 342, 'group': 3, 'num_categories': [6,5,5], 'task_index': [1,2,3],
        #     'epochs': 400, 'batch_size': 64, 
        #     'MTLKD_LambdaParameter': 2.0, 'MTLKDRe_LambdaParameter': 2.0,
        #     'MTLKDRePlus_LambdaParameter': 2.5, 'MTLKDRePlus_Temperature':2.5,
        #     'checkpoint_path':['state_dics/csida_task1.pth',
        #       'state_dics/csida_task2.pth',
        #       'state_dics/csida_task3.pth',]
        # },
        # {
        #     # gesture + location + user_identify task in csida
        #     'dataset': 'csida', 'root': '/media/yk/Samsung_T5/csi_301', 'mode': 'amplitude',
        #     'inchannel': 342, 'group': 3, 'num_categories': [6,5,5], 'task_index': [1,2,3],
        #     'epochs': 400, 'batch_size': 64, 
        #     'MTLKD_LambdaParameter': 2.0, 'MTLKDRe_LambdaParameter': 2.0,
        #     'MTLKDRePlus_LambdaParameter': 2.75, 'MTLKDRePlus_Temperature':2.75,
        #     'checkpoint_path':['state_dics/csida_task1.pth',
        #       'state_dics/csida_task2.pth',
        #       'state_dics/csida_task3.pth',]
        # },
        # {
        #     # gesture + location + user_identify task in csida
        #     'dataset': 'csida', 'root': '/media/yk/Samsung_T5/csi_301', 'mode': 'amplitude',
        #     'inchannel': 342, 'group': 3, 'num_categories': [6,5,5], 'task_index': [1,2,3],
        #     'epochs': 400, 'batch_size': 64, 
        #     'MTLKD_LambdaParameter': 2.0, 'MTLKDRe_LambdaParameter': 2.0,
        #     'MTLKDRePlus_LambdaParameter': 3.0, 'MTLKDRePlus_Temperature':3.0,
        #     'checkpoint_path':['state_dics/csida_task1.pth',
        #       'state_dics/csida_task2.pth',
        #       'state_dics/csida_task3.pth',]
        # },
        # {
        #     # gesture + location + user_identify task in csida
        #     'dataset': 'csida', 'root': '/media/yk/Samsung_T5/csi_301', 'mode': 'amplitude',
        #     'inchannel': 342, 'group': 3, 'num_categories': [6,5,5], 'task_index': [1,2,3],
        #     'epochs': 400, 'batch_size': 64, 
        #     'MTLKD_LambdaParameter': 2.0, 'MTLKDRe_LambdaParameter': 2.0,
        #     'MTLKDRePlus_LambdaParameter': 3.25, 'MTLKDRePlus_Temperature':3.25,
        #     'checkpoint_path':['state_dics/csida_task1.pth',
        #       'state_dics/csida_task2.pth',
        #       'state_dics/csida_task3.pth',]
        # },
        # {
        #     # gesture + location + user_identify task in widar
        #     'dataset': 'widar', 'root': '/media/yk/Samsung_T5/Widar3.0ReleaseData/np_f2/Widar', 'mode': 'amplitude',
        #     'inchannel': 90, 'group': 3, 'num_categories': [6,5,17], 'task_index': [1,2,3],
        #     'epochs': 500, 'batch_size': 64,
        #     'MTLKD_LambdaParameter': 2.0, 'MTLKDRe_LambdaParameter': 2.0,
        #     'MTLKDRePlus_LambdaParameter': 2.0, 'MTLKDRePlus_Temperature':2.0,
        #     'checkpoint_path':['state_dics/widar_task1.pth',
        #       'state_dics/widar_task2.pth',
        #       'state_dics/widar_task3.pth',]
        # },
        
        {
            #     # gesture + location + user_identify task in csida
            'dataset': 'csida', 'root': '/media/yk/Samsung_T5/csi_301', 'mode': 'amplitude',
            'inchannel': 342, 'group': 3, 'num_categories': [6,5,5], 'task_index': [1,2,3],
            'epochs': 500, 'batch_size': 64, 
            'MTLKD_LambdaParameter': 2.0, 'MTLKDRe_LambdaParameter': 2.0,
            'MTLKDRePlus_LambdaParameter': 0.4, 'MTLKDRePlus_Temperature':0.4,
            'checkpoint_path':['state_dics/csida_task1.pth',
              'state_dics/csida_task2.pth',
              'state_dics/csida_task3.pth',]
        },
        {
            #     # gesture + location + user_identify task in csida
            'dataset': 'csida', 'root': '/media/yk/Samsung_T5/csi_301', 'mode': 'amplitude',
            'inchannel': 342, 'group': 3, 'num_categories': [6,5,5], 'task_index': [1,2,3],
            'epochs': 500, 'batch_size': 64, 
            'MTLKD_LambdaParameter': 2.0, 'MTLKDRe_LambdaParameter': 2.0,
            'MTLKDRePlus_LambdaParameter': 0.6, 'MTLKDRePlus_Temperature':0.6,
            'checkpoint_path':['state_dics/csida_task1.pth',
              'state_dics/csida_task2.pth',
              'state_dics/csida_task3.pth',]
        },
    ]

    for i,x in enumerate(configuration):
        args.dataset = x['dataset']
        args.root = x['root']
        args.mode = x['mode']
        args.num_categories = x['num_categories']
        args.task_index = x['task_index']
        args.inchannel = x['inchannel']
        args.group = x['group']
        args.checkpoint_path = x['checkpoint_path']
        args.MTLKD_LambdaParameter = x['MTLKD_LambdaParameter']
        args.MTLKDRe_LambdaParameter = x['MTLKDRe_LambdaParameter']
        args.MTLKDRePlus_LambdaParameter = x['MTLKDRePlus_LambdaParameter']
        args.MTLKDRePlus_Temperature = x['MTLKDRePlus_Temperature']
        args.epochs = x['epochs']
        args.batch_size = x['batch_size']
        
        for j in range(1):
            run(args)



