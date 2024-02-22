import ipdb
import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import os
from math import *
import time
from util.visualizer import Visualizer

import wandb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/DDM_train.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'],
                        help='Run either train(training) or test(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    
    
    parser.add_argument('--nowandb', action="store_true", help="skip weights and biases logging")    
    parser.add_argument('--name', type=str, default=None, help="weights and biases run name")
    parser.add_argument('--project', type=str, default='ddm_e1', help="weights and biases project name")
    parser.add_argument('--tags', type=str, nargs="+", default="", help="weights and biases tags")
    

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)
    visualizer = Visualizer(opt)
    
    # wandb
    wandb.login()
    if wandb.run is not None:
        wandb.finish()    
    if not args.nowandb:        
        project=args.project
        run = wandb.init(
            # Set the project where this run will be logged
            project=args.project, 
            tags=args.tags, 
            notes='',
            name=args.name)

        config = {}
        if args.name is not None: 
            config['name'] = args.name        
            
            # initial_width=64,base_width=10, current_width=10,
            # dropout=True,dropout_rate=0.2,
            # epochs=600,learning_rate = 0.0001,
            # patience=100, output_size=2,batch_size=8,

        w = wandb.config = config        

        
    

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'], 'train', level=logging.INFO, screen=True)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))

    batchSize = opt['datasets']['train']['batch_size']
    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        train_set = Data.create_dataset_cardiac(dataset_opt, phase)
        train_loader = Data.create_dataloader(train_set, dataset_opt, phase)
        training_iters = int(ceil(train_set.data_len / float(batchSize)))
        
        val_set = Data.create_dataset_cardiac(dataset_opt, "test")
        val_loader = Data.create_dataloader(val_set, dataset_opt, "test")
        val_iters = int(ceil(val_set.data_len / float(batchSize)))
        
        
    logger.info('Initial Training Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_epoch = opt['train']['n_epoch']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])

    #### MOdel training ####
    while current_epoch < n_epoch:
        current_epoch += 1
        for istep, train_data in enumerate(train_loader):
            iter_start_time = time.time()
            current_step += 1
            
            
            diffusion.feed_data(train_data)
            if args.nowandb:
                wandb=None
            diffusion.optimize_parameters(wandb)
            # log
            if (istep+1) % opt['train']['print_freq'] == 0:
                logs = diffusion.get_current_log()
                t = (time.time() - iter_start_time) / batchSize
                visualizer.print_current_errors(current_epoch, istep+1, training_iters, logs, t, 'Train')
                # visualizer.plot_current_errors(current_epoch, (istep+1) / float(training_iters), logs)

        # validation at the end of each epoch... 
        if (current_epoch) % opt['train']['val_freq'] == 0:
            print(f"Validating at {current_epoch}")
            for istep_val, val_data in enumerate(val_loader):
                diffusion.feed_valdata(val_data)
                diffusion.validate(wandb)
                if istep_val >8:  # lets validate on 8 random points ...
                    break
            #     diffusion.test(continous=False)
            #     visuals = diffusion.get_current_visuals()
            #     visualizer.display_current_results(visuals, current_epoch, True)

        if current_epoch % opt['train']['save_checkpoint_epoch'] == 0:
            logger.info('Saving models and training states.')
            diffusion.save_network(current_epoch, current_step)

    # save model
    logger.info('End of training.')