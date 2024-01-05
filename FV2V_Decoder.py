import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm
import imageio
from skimage import img_as_ubyte
import os, sys
import yaml
from argparse import ArgumentParser
import numpy as np
from skimage.transform import resize
import torch

import time
import random
import pandas as pd
import collections
import itertools
from scipy.spatial import ConvexHull
import scipy.io as io
import json
import cv2
import math
import torch.nn.functional as F


from arithmetic.value_encoder import *
from arithmetic.value_decoder import *

from GFVC.utils import *
from GFVC.CFTE_utils import *
from GFVC.FOMM_utils import *
from GFVC.FV2V_utils import *
from GFVC.Decoder_utils import *
from GFVC.Translator.Paramtranscoder import *





    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--original_seq", default='./testing_sequence/001_256x256.rgb', type=str, help="path to the input testing sequence")
    parser.add_argument("--encoding_frames", default=250, help="the number of encoding frames")
    parser.add_argument("--seq_width", default=256, help="the width of encoding frames")
    parser.add_argument("--seq_height", default=256, help="the height of encoding frames")
    parser.add_argument("--quantization_factor", default=256, type=int, help="the quantization factor for the residual conversion from float-type to int-type")
    parser.add_argument("--Iframe_QP", default=42, help="the quantization parameters for encoding the Intra frame")
    parser.add_argument("--Iframe_format", default='YUV420', type=str,help="the quantization parameters for encoding the Intra frame")    
    parser.add_argument("--Encoder_type", default='CFTE', type=str,help="encoder type of GFVC", choices=['FV2V','FOMM','CFTE'])
    opt = parser.parse_args()

    encoder_type = opt.Encoder_type
    if encoder_type == 'CFTE':
        CFTE2FV2V_checkpoint_path='./GFVC/Translator/checkpoint/CFTE2FV2V_checkpoint.pth.tar'      
       
        trans_enc = load_CFTE_Enc_translator(CFTE2FV2V_checkpoint_path, cpu=False)
        trans_dec = load_FV2V_Dec_translator(CFTE2FV2V_checkpoint_path, cpu=False)
        CFTE_config_path='./GFVC/CFTE/checkpoint/CFTE-256.yaml'
        CFTE_checkpoint_path='./GFVC/CFTE/checkpoint/CFTE-checkpoint.pth.tar'         
        CFTE_Analysis_Model, CFTE_Synthesis_Model = load_CFTE_checkpoints(CFTE_config_path, CFTE_checkpoint_path, cpu=False)
        print('--CFTE Model Loaded--')

    if encoder_type == 'FOMM':
        FOMM2FV2V_checkpoint_path='./GFVC/Translator/checkpoint/FOMM2FV2V_checkpoint.pth.tar'      
      
        trans_enc = load_FOMM_Enc_translator(FOMM2FV2V_checkpoint_path, cpu=False)
        trans_dec = load_FV2V_Dec_translator(FOMM2FV2V_checkpoint_path, cpu=False)
        ## FOMM
        FOMM_config_path='./GFVC/FOMM/checkpoint/FOMM-256.yaml'
        FOMM_checkpoint_path='./GFVC/FOMM/checkpoint/FOMM-checkpoint.pth.tar'         
        FOMM_Analysis_Model, _ = load_FOMM_checkpoints(FOMM_config_path, FOMM_checkpoint_path, cpu=False)
        print('--FOMM Model Loaded--')
   
        
    
    
    frames=int(opt.encoding_frames)
    width=opt.seq_width
    height=opt.seq_width
    Qstep=opt.quantization_factor
    QP=opt.Iframe_QP
    Iframe_format=opt.Iframe_format    
    seq = os.path.splitext(os.path.split(opt.original_seq)[-1])[0]
    

    ## FV2V
    FV2V_config_path='./GFVC/FV2V/checkpoint/FV2V-256.yaml'
    FV2V_checkpoint_path='./GFVC/FV2V/checkpoint/FV2V-checkpoint.pth.tar'         
    FV2V_Analysis_Model_Detector, FV2V_Analysis_Model_Estimator, FV2V_Synthesis_Model = load_FV2V_checkpoints(FV2V_config_path, FV2V_checkpoint_path, cpu=False)
    print('--FV2V Model Loaded--')
    modeldir = 'FV2V' 
    model_dirname='./experiment/'+modeldir+"/"+'Iframe_'+str(Iframe_format)   
    model_dirname_enc ='./experiment/'+encoder_type+"/"+'Iframe_'+str(Iframe_format)  
      
############################################                   
            
    driving_kp = model_dirname_enc+'/kp/'+seq+'_QP'+str(QP)+'/'

    dir_dec=model_dirname+'/'+encoder_type+'2'+modeldir+'_dec/'

    os.makedirs(dir_dec,exist_ok=True)     # the real decoded video  
    decode_seq=dir_dec+seq+'_QP'+str(QP)+'.rgb'

    dir_enc = model_dirname_enc+'/enc/'+seq+'_QP'+str(QP)+'/'
    os.makedirs(dir_enc,exist_ok=True)     # the frames to be compressed by vtm       
            
    dir_bit=dir_dec+'resultBit/'
    os.makedirs(dir_bit,exist_ok=True)         
    
    f_dec=open(decode_seq,'w') 



    seq_kp_integer=[]               #  the quantilized compact feature list of the whole sequence

    start=time.time() 
    generate_time = 0

    sum_bits = 0


    for frame_idx in range(0, frames):            
        frame_idx_str = str(frame_idx).zfill(4)   
        if frame_idx in [0]:      # I-frame                      
          
            if Iframe_format=='YUV420':
                os.system("./vtm/decode.sh "+dir_enc+'frame'+frame_idx_str)
                bin_file=dir_enc+'frame'+frame_idx_str+'.bin'
                bits=os.path.getsize(bin_file)*8
                sum_bits += bits

                #  read the rec frame (yuv420) and convert to rgb444
                rec_ref_yuv=yuv420_to_rgb444(dir_enc+'frame'+frame_idx_str+'_rec.yuv', width, height, 0, 1, False, False) 
                img_rec = rec_ref_yuv[frame_idx]
                img_rec = img_rec[:,:,::-1].transpose(2, 0, 1)    # HxWx3
                img_rec.tofile(f_dec)                         
                img_rec = resize(img_rec, (3, height, width))    # normlize to 0-1                                      

            elif Iframe_format=='RGB444':
                os.system("./vtm/decode_rgb444.sh "+dir_enc+'frame'+frame_idx_str)
                bin_file=dir_enc+'frame'+frame_idx_str+'.bin'
                bits=os.path.getsize(bin_file)*8
                sum_bits += bits

                f_temp=open(dir_enc+'frame'+frame_idx_str+'_dec.rgb','rb')
                img_rec=np.fromfile(f_temp,np.uint8,3*height*width).reshape((3,height,width))   # 3xHxW RGB         
                img_rec.tofile(f_dec) 
                img_rec = resize(img_rec, (3, height, width))    # normlize to 0-1      
                
            with torch.no_grad(): 
                reference = torch.tensor(img_rec[np.newaxis].astype(np.float32))
                reference = reference.cuda()    # require GPU
                kp_canonical = FV2V_Analysis_Model_Detector(reference)  ####reference 
                he_source = FV2V_Analysis_Model_Estimator(reference)  
                kp_reference = keypoint_transformation_source(kp_canonical, he_source, estimate_jacobian=False,
                                                           free_view=False, yaw=0, pitch=0, roll=0) ###### I frame                         
                if encoder_type == 'CFTE':
                    seq_kp_integer, _ = cfte_receiving_key_frame(CFTE_Analysis_Model, seq_kp_integer,reference)   
                elif encoder_type == 'FOMM':
                    seq_kp_integer, _, _ = fomm_receiving_key_frame(FOMM_Analysis_Model, seq_kp_integer,reference)
                elif encoder_type == 'FV2V':
                    seq_kp_integer, _, _, _ = fv2v_receiving_key_frame(FV2V_Analysis_Model_Estimator, seq_kp_integer,reference)

        else:
            frame_index=str(frame_idx).zfill(4)
            bin_save=driving_kp+'/frame'+frame_index+'.bin'            
            kp_dec = final_decoder_expgolomb(bin_save)

            ## decoding residual
            kp_difference = data_convert_inverse_expgolomb(kp_dec)
            ## inverse quanzation
            kp_difference_dec=[i/Qstep for i in kp_difference]
            kp_difference_dec= eval('[%s]'%repr(kp_difference_dec).replace('[', '').replace(']', ''))  

            kp_previous=seq_kp_integer[frame_idx-1]
            kp_previous= eval('[%s]'%repr(kp_previous).replace('[', '').replace(']', '').replace("'", ""))   
            
            if encoder_type == 'CFTE':
                seq_kp_integer, kp_current_value = cfte_receiving_inter_frame(listformat_adptive_CFTE,kp_previous,kp_difference_dec,seq_kp_integer)
                kp_current_exp, kp_current_mat, kp_current_t = trans_dec(trans_enc(kp_current_value))
            elif encoder_type == 'FOMM':
                seq_kp_integer, kp_current_value, kp_current_jocobi = fomm_receiving_inter_frame(listformat_kp_jocobi_FOMM,kp_previous,kp_difference_dec,seq_kp_integer)
                kp_current_exp, kp_current_mat, kp_current_t = trans_dec(trans_enc(kp_current_value, kp_current_jocobi))
            elif encoder_type == 'FV2V':   
                seq_kp_integer, kp_current_mat, kp_current_t, kp_current_exp = fv2v_receiving_inter_frame(listformat_kp_mat_exp_FV2V,kp_previous,kp_difference_dec,seq_kp_integer)
               
            
            
            dict={}                  
            dict['rot_mat']=kp_current_mat 
      
            dict['t']=kp_current_t    
            dict['exp']=kp_current_exp  

            kp_current_matrix=dict  

            kp_current = keypoint_transformation(kp_canonical, kp_current_matrix, estimate_jacobian=False, 
                                                 free_view=False, yaw=0, pitch=0, roll=0)    

            # generated frame
            generate_start = time.time()    

            prediction = make_FV2V_prediction(reference, kp_reference, kp_current, FV2V_Synthesis_Model) #######################

            generate_end = time.time()
            generate_time += generate_end - generate_start                        

            pre=(prediction*255).astype(np.uint8)  
            pre.tofile(f_dec)                              

            frame_index=str(frame_idx).zfill(4)
            bin_save=driving_kp+'/frame'+frame_index+'.bin'
            bits=os.path.getsize(bin_save)*8
            sum_bits += bits

    f_dec.close()     
    end=time.time()                    


    print(seq+'_QP'+str(QP)+'.rgb',"success. Total time is %.4fs. Model inference time is %.4fs. Total bits are %d" %(end-start,generate_time,sum_bits))
    
    totalResult=np.zeros((1,3))
    totalResult[0][0]=sum_bits   
    totalResult[0][1]=end-start   
    totalResult[0][2]=generate_time   
    
    np.savetxt(dir_bit+seq+'_QP'+str(QP)+'.txt', totalResult, fmt = '%.5f')            



                    
                    
                    
                    
                    
    
    