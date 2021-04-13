import tqdm
import pandas as pd


import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from .SOMVAE import TrackEncoder, TrackDecoder, SOMVAE



class SOMVAEDataset(Dataset):
    def __init__(self, path, frameshape=(2,11), selpkid=None):
        super().__init__()

        self.oridf = pd.read_csv(path, sep='\t', index_col=0,
                         names = ['chr','start','end','pkid','segnum','values'])
        if selpkid is not None:
            self.seldf = self.oridf[self.oridf['pkid'].isin(selpkid)]
        else:
            self.seldf = self.oridf

        self.data = list(self.seldf.groupby('pkid'))
        self.frameshape = frameshape

    def __getitem__(self, index):
        df = self.data[index][1]
        return self.data[index][0],\
                df['values'].str.split(',',expand=True).astype(float).values.reshape(-1,*self.frameshape)

    def get_peak(self, pkids):
        if isinstance(pkids, str):
            pkids = [pkids]
        return self.oridf[self.oridf['pkid'].isin(pkids)]
    
    def __len__(self):
        return len(self.data)



class SOMVAEDataLoader(DataLoader):
    def __init__(self, dataset, shuffle):
        super().__init__(dataset, batch_size=1, shuffle=shuffle)



def construct_model(ntracks=2, ntimepoints=11, latent_dim=10, 
                    encoder_kernel_size=[3,3], encoder_strides=[1,1], encoder_hidden_channels=[5,5], 
                    predeconv_ntimepoints=7,
                    decoder_kernel_size=[3,3], decoder_strides=[1,1], decoder_hidden_channels=[5,5], 
                    som_dim = [4,4], 
                    alpha = 1.0, beta = 0.9, gamma = 1.8, tau = 1.4, ):
    encoder = TrackEncoder(ntracks, ntimepoints, latent_dim=latent_dim, 
                           kernel_size=encoder_kernel_size, strides=encoder_strides, 
                           n_channels=encoder_hidden_channels)
    decoder_e = TrackDecoder(ntracks, latent_dim=latent_dim, predeconv_timepoints=predeconv_ntimepoints, 
                             kernel_size=decoder_kernel_size, strides=decoder_strides, 
                             n_channels=decoder_hidden_channels)
    decoder_q = TrackDecoder(ntracks, latent_dim=latent_dim, predeconv_timepoints=predeconv_ntimepoints, 
                             kernel_size=decoder_kernel_size, strides=decoder_strides, 
                             n_channels=decoder_hidden_channels)
    model = SOMVAE(encoder, decoder_e, decoder_q, 
                   alpha=alpha, beta=beta, gamma=gamma, tau=tau,
                   latent_dim = latent_dim,
                   som_dim=som_dim)
    
    return model


def train(model, dataloader, 
          model_learning_rate=0.0005, prob_learning_rate=0.05, 
          model_decay_factor=0.9, prob_decay_factor=0.9, 
          model_decay_nsteps=1000, prob_decay_nsteps=1000, 
          nepoches=5, 
          disable_tqdm=False,
          callbacks=[]):
    
    model_param_list = nn.ParameterList()
    for p in model.named_parameters():
        if p[0] != 'probs':
            model_param_list.append(p[1])

    probs_param_list = nn.ParameterList()
    for p in model.named_parameters():
        if p[0] == 'probs':
            probs_param_list.append(p[1])
            
    opt_model = torch.optim.Adam(model_param_list, lr=model_learning_rate)
    opt_probs = torch.optim.Adam(probs_param_list, lr=prob_learning_rate)
    
    sc_opt_model = torch.optim.lr_scheduler.StepLR(opt_model, model_decay_nsteps, model_decay_factor)
    sc_opt_probs = torch.optim.lr_scheduler.StepLR(opt_probs, prob_decay_nsteps, prob_decay_factor)
    

    for e in tqdm.notebook.tqdm(range(nepoches), desc="Epoch", position=0, disable=False):
        for i,(pkid, batch_x) in tqdm.notebook.tqdm(enumerate(dataloader), desc="step", 
                                                    position=1, leave=False, disable=disable_tqdm):
            batch_x = batch_x.squeeze(0).float()
            opt_model.zero_grad()
            opt_probs.zero_grad()
            x_e, x_q, z_e, z_q, z_q_neighbors, k, z_dist_flat = model(batch_x)
            l = model.loss(batch_x, x_e, x_q, z_e, z_q, z_q_neighbors, k, z_dist_flat)
            l_prob = model.loss_prob(k)
            l.backward()
            opt_model.step()
            l_prob.backward()
            opt_probs.step()
            sc_opt_model.step()
            sc_opt_probs.step()

            #=============
            model.update_transition_probs()
            #=============

            for callback in callbacks:
                callback(e, i, pkid, batch_x, model, x_e=x_e, x_q=x_q, z_e=z_e, z_q=z_q, z_q_neighbors=z_q_neighbors, k=k, z_dist_flat=z_dist_flat)



def test(model, dataloader, disable_tqdm=False, callbacks=[]):
    
    with torch.no_grad():
        for i,(pkid, batch_x) in tqdm.notebook.tqdm(enumerate(dataloader), desc="step", disable=disable_tqdm):
            batch_x = batch_x.squeeze(0).float()
            x_e, x_q, z_e, z_q, z_q_neighbors, k, z_dist_flat = model(batch_x)
            l = model.loss(batch_x, x_e, x_q, z_e, z_q, z_q_neighbors, k, z_dist_flat)
            l_prob = model.loss_prob(k)

            for callback in callbacks:
                callback(0, i, pkid, batch_x, model, x_e=x_e, x_q=x_q, z_e=z_e, z_q=z_q, z_q_neighbors=z_q_neighbors, k=k, z_dist_flat=z_dist_flat)

    
