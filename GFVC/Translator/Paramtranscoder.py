import matplotlib
matplotlib.use('Agg')
import os, sys
import yaml
from argparse import ArgumentParser
from tqdm import tqdm
import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
import torch.nn as nn 

from scipy.spatial import ConvexHull
import scipy.io as io
import json
import cv2
import torch.nn.functional as F
import struct, time
from pathlib import Path


from GFVC.CFTE.sync_batchnorm import DataParallelWithCallback


def load_CFTE_Enc_translator(checkpoint_path, cpu=False):
    
    translator = CFTE_Enc(N=64)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')  

    translator.load_state_dict(checkpoint['state_dict']['CFTE_Enc'])
        
        
    if not cpu:
        translator.cuda()
        translator = DataParallelWithCallback(translator)

    translator.eval()
    return translator


def load_CFTE_Dec_translator(checkpoint_path, cpu=False):
    
    translator = CFTE_Dec(N=64)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')   
    translator.load_state_dict(checkpoint['state_dict']['CFTE_Dec'])
        
        
    if not cpu:
        translator.cuda()
        translator = DataParallelWithCallback(translator)

    translator.eval()
    return translator


def load_FOMM_Enc_translator(checkpoint_path, cpu=False):
    
    translator = FOMM_Enc(N=64)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')   
    translator.load_state_dict(checkpoint['state_dict']['FOMM_Enc'])
        
        
    if not cpu:
        translator.cuda()
        translator = DataParallelWithCallback(translator)

    translator.eval()
    return translator


def load_FOMM_Dec_translator(checkpoint_path, cpu=False):
    
    translator = FOMM_Dec(N=64)
    
        
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    translator.load_state_dict(checkpoint['state_dict']['FOMM_Dec'])
        
        
    if not cpu:
        translator.cuda()
        translator = DataParallelWithCallback(translator)

    translator.eval()
    return translator

def load_FV2V_Enc_translator(checkpoint_path, cpu=False):
    
    translator = NVIDIA_Enc(N=64)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    translator.load_state_dict(checkpoint['state_dict']['NVIDIA_Enc'])
        
        
    if not cpu:
        translator.cuda()
        translator = DataParallelWithCallback(translator)

    translator.eval()
    return translator

def load_FV2V_Dec_translator(checkpoint_path, cpu=False):
    
    translator = NVIDIA_Dec(N=64)

    checkpoint = torch.load(checkpoint_path, map_location='cpu') 
    translator.load_state_dict(checkpoint['state_dict']['NVIDIA_Dec'])
        
        
    if not cpu:
        translator.cuda()
        translator = DataParallelWithCallback(translator)

    translator.eval()
    return translator



class CFTE_Enc(nn.Module):
    def __init__(self, N=256):
        super(CFTE_Enc, self).__init__()
    
        self.CFTE = nn.Sequential(
            nn.Linear(16,N),
            nn.ReLU(inplace=True),
            nn.Linear(N,N),   
            nn.ReLU(inplace=True),
            nn.Linear(N,N),   
            nn.ReLU(inplace=True),
            nn.Linear(N,N), 
            
        )
    

    def forward(self,x):
        mid = self.CFTE(x.reshape(x.size(0),16))
        
        return mid

class CFTE_Dec(nn.Module):
    def __init__(self, N=256):
        super(CFTE_Dec, self).__init__()

        self.CFTE = nn.Sequential(
            nn.Linear(N,N),
            nn.ReLU(inplace=True),
            nn.Linear(N,N),   
            nn.ReLU(inplace=True),
            nn.Linear(N,N),   
            nn.ReLU(inplace=True),
            nn.Linear(N,16), 
            
        )
       

    def forward(self,mid):
        params = self.CFTE(mid).reshape(mid.size(0),4,4)
        
        return params

class FOMM_Enc(nn.Module):
    def __init__(self, N=256):
        super(FOMM_Enc, self).__init__()
    
        self.FOMM = nn.Sequential(
            nn.Linear(60,N),
            nn.ReLU(inplace=True),
            nn.Linear(N,N),
            nn.ReLU(inplace=True),
            nn.Linear(N,N), 
            nn.ReLU(inplace=True),
            nn.Linear(N,N),   
        )

    def forward(self,kp, jacobian):
        mid = self.FOMM(torch.cat([kp.reshape(kp.size(0),20),jacobian.reshape(kp.size(0),40)],dim=1))
        return mid      

class FOMM_Dec(nn.Module):
    def __init__(self, N=256):
        super(FOMM_Dec, self).__init__()
    
        self.FOMM = nn.Sequential(
            nn.Linear(N,N),
            nn.ReLU(inplace=True),
            nn.Linear(N,N),
            nn.ReLU(inplace=True),
            nn.Linear(N,N), 
            nn.ReLU(inplace=True),
            nn.Linear(N,60),   
        )

    def forward(self,mid):
        kp, jacobian = self.FOMM(mid).split([20,40],dim=1)
        kp = kp.reshape(kp.size(0),10,2)
        jacobian = jacobian.reshape(kp.size(0),10,2,2)
        
        return kp, jacobian


class NVIDIA_Enc(nn.Module):
    def __init__(self, N=256):
        super(NVIDIA_Enc, self).__init__()
    
        self.NVIDIA = nn.Sequential(
            nn.Linear(57,N),
            nn.ReLU(inplace=True),
            nn.Linear(N,N),
            nn.ReLU(inplace=True),
            nn.Linear(N,N), 
            nn.ReLU(inplace=True),
            nn.Linear(N,N),   
        )

    def forward(self, kp, rot, trans):
        mid = self.NVIDIA(torch.cat([kp,rot.reshape(kp.size(0),9),trans],dim=1))
        return mid


class NVIDIA_Dec(nn.Module):
    def __init__(self, N=256):
        super(NVIDIA_Dec, self).__init__()

        self.NVIDIA = nn.Sequential(
            nn.Linear(N,N),
            nn.ReLU(inplace=True),
            nn.Linear(N,N),
            nn.ReLU(inplace=True),
            nn.Linear(N,N), 
            nn.ReLU(inplace=True),
            nn.Linear(N,57),   
        )    

    def forward(self,mid):
        kp, rot, trans = self.NVIDIA(mid).split([45,9,3],dim=1)
        rot = rot.reshape(mid.size(0),3,3)
        return kp, rot, trans    
