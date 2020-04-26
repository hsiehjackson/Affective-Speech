import torch
import numpy as np
import sys
import os 
import torch.nn as nn
import torch.nn.functional as F
import yaml
import pickle
import glob
import re

from model_MUNIT import Generator, Discriminator
from dataset import TotalDataset
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from utils import *
import random

class Solver_MUNIT(object):
    def __init__(self, config, args):
        # config store the value of hyperparameters, turn to attr by AttrDict
        self.config = config
        #print(config)

        # args store other information
        self.args = args
        #print(self.args)

        # logger to use tensorboard
        self.logger = Logger(self.args.logdir)

        # get dataloader
        self.get_data_loaders()

        # init the model with config
        self.build_model()
        self.save_config()

        if args.load_model:
            self.load_model()

    def save_config(self):
        with open( os.path.join(self.args.store_model_path,'config.yaml'), 'w') as f:
            yaml.dump(self.config, f)
        with open(os.path.join(self.args.store_model_path,'args.yaml'), 'w') as f:
            yaml.dump(vars(self.args), f)
        return

    def save_model(self, iteration):
        # save model and their optimizer
        model = {
        'gen_a': self.gen_a.state_dict(),
        'gen_b': self.gen_b.state_dict(),
        'dis_a': self.dis_a.state_dict(),
        'dis_b': self.dis_b.state_dict(),
        'gen_opt': self.gen_opt.state_dict(),
        'dis_opt': self.dis_opt.state_dict()
        }
        torch.save(model, os.path.join(self.args.store_model_path, f'model-{iteration+1}.ckpt'))

    def load_model(self):
        print(f'Load model from {self.args.load_model_path}')
        model = torch.load(self.args.load_model_path)
        self.gen_a.load_state_dict(model['gen_a'])
        self.gen_b.load_state_dict(model['gen_b'])
        self.dis_a.load_state_dict(model['dis_a'])
        self.dis_b.load_state_dict(model['dis_b'])
        self.gen_opt.load_state_dict(model['gen_opt'])
        self.dis_opt.load_state_dict(model['dis_opt'])
        return


    def get_data_loaders(self):

        self.train_dataset_a = TotalDataset(
                os.path.join(self.args.data_dir, 'dataset.hdf5'), 
                os.path.join(self.args.data_dir, 'train_data.json'), 
                os.path.join(self.args.data_dir, 'train_speakers.txt'),
                segment_size=self.config['data_loader']['segment_size'],
                emotion=self.config['emotion']['a'])

        self.train_loader_a = DataLoader(
                dataset=self.train_dataset_a, 
                batch_size=self.config['data_loader']['batch_size'], 
                shuffle=self.config['data_loader']['shuffle'], 
                num_workers=self.config['data_loader']['num_workers'],
                drop_last=True,
                pin_memory=True)

        self.train_display_a = cc(torch.stack([torch.tensor(self.train_loader_a.dataset[i]['mel']) for i in range(self.config['display_size'])]))

        self.train_dataset_b = TotalDataset(
                os.path.join(self.args.data_dir, 'dataset.hdf5'), 
                os.path.join(self.args.data_dir, 'train_data.json'), 
                os.path.join(self.args.data_dir, 'train_speakers.txt'),
                segment_size=self.config['data_loader']['segment_size'],
                emotion=self.config['emotion']['b'])

        self.train_loader_b = DataLoader(
                dataset=self.train_dataset_b, 
                batch_size=self.config['data_loader']['batch_size'], 
                shuffle=self.config['data_loader']['shuffle'], 
                num_workers=self.config['data_loader']['num_workers'],
                drop_last=True,
                pin_memory=True)

        self.train_display_b = cc(torch.stack([torch.tensor(self.train_loader_b.dataset[i]['mel']) for i in range(self.config['display_size'])]))

        self.train_iter_a = infinite_iter(self.train_loader_a)
        self.train_iter_b = infinite_iter(self.train_loader_b)

        return

    def build_model(self): 
        # create model, discriminator, optimizers
        self.gen_a = cc(Generator(self.config))
        self.gen_b = cc(Generator(self.config))
        self.dis_a = cc(Discriminator(**self.config['Discriminator']))
        self.dis_b = cc(Discriminator(**self.config['Discriminator']))

        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
        
        optimizer = self.config['optimizer']
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=optimizer['lr'], betas=(optimizer['beta1'], optimizer['beta2']), weight_decay=optimizer['weight_decay'])

        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=optimizer['lr'], betas=(optimizer['beta1'], optimizer['beta2']), weight_decay=optimizer['weight_decay'])        
        
        self.gen_a.apply(weights_init('kaiming'))
        self.gen_b.apply(weights_init('kaiming'))
        self.dis_a.apply(weights_init('gaussian'))
        self.dis_b.apply(weights_init('gaussian'))
 
        self.s_a = cc(torch.randn(self.config['display_size'], 256))
        self.s_b = cc(torch.randn(self.config['display_size'], 256))

        self.dis_scheduler = lr_scheduler.StepLR(self.dis_opt, step_size=optimizer['step_size'], gamma=optimizer['gamma'], last_epoch=-1)
        self.gen_scheduler = lr_scheduler.StepLR(self.gen_opt, step_size=optimizer['step_size'], gamma=optimizer['gamma'], last_epoch=-1)

        return

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def sample(self, x_a, x_b):
        self.gen_a.eval()
        self.gen_b.eval()
        s_a1 = self.s_a
        s_b1 = self.s_b
        s_a2 = cc(torch.randn(x_a.size(0), 256))
        s_b2 = cc(torch.randn(x_b.size(0), 256))
        samples = []
        for i in range(x_a.size(0)):
            sample_one = []
            c_a, s_a_fake = self.gen_a.encode(x_a[i].unsqueeze(0).permute(0,2,1))
            c_b, s_b_fake = self.gen_b.encode(x_b[i].unsqueeze(0).permute(0,2,1))
            sample_one.append(x_a[i].detach().cpu().numpy())
            sample_one.append(self.gen_a.decode_infer(c_a, s_a_fake))
            sample_one.append(self.gen_b.decode_infer(c_a, s_b1[i].unsqueeze(0)))
            sample_one.append(self.gen_b.decode_infer(c_a, s_b2[i].unsqueeze(0)))
            sample_one.append(x_b[i].detach().cpu().numpy())
            sample_one.append(self.gen_b.decode_infer(c_b, s_b_fake))
            sample_one.append(self.gen_a.decode_infer(c_b, s_a1[i].unsqueeze(0)))
            sample_one.append(self.gen_a.decode_infer(c_b, s_a2[i].unsqueeze(0)))
            samples.append(sample_one)
        self.gen_a.train()
        self.gen_b.train()
        return samples

    def gradient_penalty(self, real, fake, dis):
        alpha = cc(torch.rand(real.size(0), 1, 1)).expand_as(real)
        interpolates = (alpha * real + ((1 - alpha) * fake)).requires_grad_(True)
        d_interpolates = dis(interpolates)
        fake = cc(torch.ones(d_interpolates.shape))
        dydx = torch.autograd.grad(outputs=d_interpolates,
                                   inputs=interpolates,
                                   grad_outputs=fake,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def gen_criterion(self, x_a_fake, x_b_fake):
        gen_pred_a = self.dis_a(x_a_fake)
        loss_gen_adv_a = - torch.mean(gen_pred_a) * self.config['lambda']['gan_w']
        gen_pred_b = self.dis_a(x_b_fake)
        loss_gen_adv_b = - torch.mean(gen_pred_b) * self.config['lambda']['gan_w']
        return loss_gen_adv_a, loss_gen_adv_b
    
    def dis_criterion(self, x_a_real, x_a_fake, x_b_real, x_b_fake):
        
        real_pred_a, fake_pred_a = self.dis_b(x_a_real), self.dis_b(x_a_fake)
        real_loss_a = - torch.mean(real_pred_a)
        fake_loss_a = torch.mean(fake_pred_a)
        gp_loss_a = self.gradient_penalty(x_a_real, x_a_fake, self.dis_a) * self.config['lambda']['gp_w'] 
        loss_dis_adv_a = real_loss_a + fake_loss_a + gp_loss_a 
                           
        real_pred_b, fake_pred_b = self.dis_b(x_b_real), self.dis_b(x_b_fake)
        real_loss_b = - torch.mean(real_pred_b)
        fake_loss_b = torch.mean(fake_pred_b)
        gp_loss_b = self.gradient_penalty(x_b_real, x_b_fake, self.dis_b) * self.config['lambda']['gp_w'] 
        loss_dis_adv_b = real_loss_b + fake_loss_b + gp_loss_b 
        
        meta = {'D_real': (real_loss_a.item() + real_loss_b.item()) / 2,
                'D_fake': (fake_loss_a.item() + fake_loss_b.item()) / 2,
                'D_gp': (gp_loss_a.item() + gp_loss_b.item()) / 2}

        return loss_dis_adv_a, loss_dis_adv_b, meta

    def gen_update(self, x_a, x_b):

        self.gen_opt.zero_grad()
        s_a = cc(torch.randn(x_a.size(0), 256))
        s_b = cc(torch.randn(x_b.size(0), 256))
        # encode
        c_a, s_a_prime = self.gen_a.encode(x_a)
        c_b, s_b_prime = self.gen_b.encode(x_b)
        # decode (within domain)
        x_a_recon = self.gen_a.decode(c_a, s_a_prime)
        x_b_recon = self.gen_b.decode(c_b, s_b_prime)
        # decode (cross domain)
        x_ba = self.gen_a.decode(c_b, s_a)
        x_ab = self.gen_b.decode(c_a, s_b)
        # encode again
        c_b_recon, s_a_recon = self.gen_a.encode(x_ba)
        c_a_recon, s_b_recon = self.gen_b.encode(x_ab)

        # reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a) * self.config['lambda']['recon_x_w']
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b) * self.config['lambda']['recon_x_w']
        self.loss_gen_recon_s_a = self.recon_criterion(s_a_recon, s_a) * self.config['lambda']['recon_s_w']
        self.loss_gen_recon_s_b = self.recon_criterion(s_b_recon, s_b) * self.config['lambda']['recon_s_w']
        self.loss_gen_recon_c_a = self.recon_criterion(c_a_recon, c_a) * self.config['lambda']['recon_c_w']
        self.loss_gen_recon_c_b = self.recon_criterion(c_b_recon, c_b) * self.config['lambda']['recon_c_w']
        # GAN loss
        self.loss_gen_adv_a, self.loss_gen_adv_b = self.gen_criterion(x_ba, x_ab)
        # total loss
        self.loss_gen_total = self.loss_gen_adv_a + \
                              self.loss_gen_adv_b + \
                              self.loss_gen_recon_x_a + \
                              self.loss_gen_recon_s_a + \
                              self.loss_gen_recon_c_a + \
                              self.loss_gen_recon_x_b + \
                              self.loss_gen_recon_s_b + \
                              self.loss_gen_recon_c_b
        self.loss_gen_total.backward()
        self.gen_opt.step()

        meta = {'G_rec': (self.loss_gen_recon_x_a.item() + self.loss_gen_recon_x_b.item()) / 2,
                'G_adv': (self.loss_gen_adv_a.item() + self.loss_gen_adv_b.item()) / 2,
                'G_content': (self.loss_gen_recon_c_a.item() + self.loss_gen_recon_c_b.item()) / 2,
                'G_style': (self.loss_gen_recon_s_a.item() + self.loss_gen_recon_s_b.item()) / 2}

        return meta

    def dis_update(self, x_a, x_b):

        self.dis_opt.zero_grad()
        s_a = cc(torch.randn(x_a.size(0), 256))
        s_b = cc(torch.randn(x_b.size(0), 256))

        # encode
        c_a, _ = self.gen_a.encode(x_a)
        c_b, _ = self.gen_b.encode(x_b)
        # decode (cross domain)
        x_ba = self.gen_a.decode(c_b, s_a)
        x_ab = self.gen_b.decode(c_a, s_b)
        # D loss
        self.loss_dis_adv_a, self.loss_dis_adv_b, meta = self.dis_criterion(x_a, x_ba.detach(), x_b, x_ab.detach())
        self.loss_dis_total = self.config['lambda']['gan_w'] * self.loss_dis_adv_a + \
                              self.config['lambda']['gan_w'] * self.loss_dis_adv_b
        self.loss_dis_total.backward()
        self.dis_opt.step()
        return meta

        
    def train(self, n_iterations):

        for iteration in range(n_iterations):
            data_a = next(self.train_iter_a)['mel']
            data_a  = cc(data_a).permute(0,2,1)
            data_b = next(self.train_iter_b)['mel']
            data_b  = cc(data_b).permute(0,2,1)

            dis_meta = self.dis_update(data_a, data_b)
            gen_meta = self.gen_update(data_a, data_b)

            all_meta = {
                'gen': sum([value for value in gen_meta.values()]),
                'dis': sum([value for value in dis_meta.values()])
            }

            # add to logger
            if iteration % self.args.summary_steps == 0:
                self.logger.scalars_summary(f'{self.args.tag}/GAN_train', dis_meta, iteration)
                self.logger.scalars_summary(f'{self.args.tag}/GAN_train', gen_meta, iteration)
                self.logger.scalars_summary(f'{self.args.tag}/GAN_train_all', all_meta, iteration)

            slot_value = (iteration+1, n_iterations) + tuple([value for value in gen_meta.values()]) + tuple([value for value in dis_meta.values()])
            log = '[%06d/%06d] | Grec:%.3f | Gadv:%.3f | GC:%.3f | GS:%.3f | Dreal:%.3f | Dfake:%.3f | Dgp:%.3f    '
            print(log % slot_value, end='\r')
            
            if (iteration + 1) % self.args.save_steps == 0 or iteration + 1 == n_iterations:
                self.save_model(iteration=iteration)
                with torch.no_grad():
                    train_outputs = self.sample(self.train_display_a, self.train_display_b)
                    save_samples(train_outputs, self.args.store_model_path, iteration)
                print()
        return

