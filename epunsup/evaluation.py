import torch
import numpy as np
#from torch import nn
#from torch.utils.data import Dataset, DataLoader

def get_whole(tensor, firstn=4):
    if tensor.shape[0]>1:
        whole = torch.cat([tensor[i,:,:firstn] for i in range(len(tensor)-1)],dim=1)
        whole = torch.cat([whole, tensor[-1]], dim=1)
    else:
        whole = torch.clone(tensor).squeeze(0)
    return whole


def plot_overlapping_tracks(track_segs, ax, chrom, regpos, ticklabel=False, colors=['red','blue']):
    start,*_,end = regpos
    
    tracks = []
    for i in range(len(track_segs)):
        tmp = []
        tmp.append(track_segs[i,:,0:3].numpy())
        if i-1>=0:
            tmp.append(track_segs[i-1,:,4:7].numpy())
        if i-2>=0:
            tmp.append(track_segs[i-1,:,8:11].numpy())
        tmp = np.array(tmp).mean(axis=0)
        tracks.append(tmp)

        if i==len(track_segs)-1:
            tmp = []
            tmp.append(track_segs[i,:,4:7].numpy())
            if i-1>=0:
                tmp.append(track_segs[i-1,:,8:11].numpy())

            tmp = np.array(tmp).mean(axis=0)
            tracks.append(tmp)

            tracks.append(track_segs[i,:,7:11].numpy())
    tracks = np.hstack(tracks)
    
    
    
    xs = np.linspace(start, end, tracks.shape[1]).astype(int)
    for track,color in zip(tracks, colors):
        ax.plot(xs,track, color=color)
        
    ax.set_ylim(-0.05,1.05)
    ax.axhline(0, linewidth=0.5, color='k')
    
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=ticklabel)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    regposlabel = [str(x) for x in regpos]
    regposlabel[0] = f'{chrom}:{regposlabel[0]}'
    ax.set_xticks(regpos)
    ax.set_xticklabels(regposlabel)
    
    return ax


def plot_tracks(tracks, ax, chrom, regpos, ticklabel=False, colors=['red','blue']):
    start,*_,end = regpos
    
    xs = np.linspace(start, end, tracks.shape[1]).astype(int)
    for track,color in zip(tracks, colors):
        ax.plot(xs,track, color=color)
        
    ax.set_ylim(-0.05,1.05)
    ax.axhline(0, linewidth=0.5, color='k')
    
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=ticklabel)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    regposlabel = [str(x) for x in regpos]
    regposlabel[0] = f'{chrom}:{regposlabel[0]}'
    ax.set_xticks(regpos)
    ax.set_xticklabels(regposlabel)
    
    return ax


def plot_indicators(whole_x, ax, pkdf):
    for y, (s,e) in enumerate( pkdf[['start','end']].values ):
        ax.plot([s,e],[y%3,y%3], color='grey')
    ax.set_ylim(-0.5,3.5)
    ax.axis('off')
    return ax

