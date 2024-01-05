import os
import sys
import json
import torch 
from GFVC.FV2V_utils import *

def cfte_receiving_key_frame(CFTE_Analysis_Model,seq_kp_integer,reference):
    kp_reference = CFTE_Analysis_Model(reference) 
    kp_value = kp_reference['value']
    kp_value_list = kp_value.tolist()
    kp_value_list = str(kp_value_list)
    kp_value_list = "".join(kp_value_list.split())

    kp_value_frame=json.loads(kp_value_list)
    kp_value_frame= eval('[%s]'%repr(kp_value_frame).replace('[', '').replace(']', ''))
    seq_kp_integer.append(kp_value_frame)   
    return seq_kp_integer, kp_value

def fomm_receiving_key_frame(FOMM_Analysis_Model,seq_kp_integer,reference):
    kp_reference_fomm = FOMM_Analysis_Model(reference)          
    ####
    kp_value = kp_reference_fomm['value']
    kp_value_list = kp_value.tolist()
    kp_value_list = str(kp_value_list)
    kp_value_list = "".join(kp_value_list.split())

    kp_jacobian=kp_reference_fomm['jacobian'] 
    kp_jacobian_list=kp_jacobian.tolist()
    kp_jacobian_list=str(kp_jacobian_list)
    kp_jacobian_list="".join(kp_jacobian_list.split())  


    kp_value_frame=json.loads(kp_value_list)###20
    kp_value_frame= eval('[%s]'%repr(kp_value_frame).replace('[', '').replace(']', ''))
    kp_jacobian_frame=json.loads(kp_jacobian_list)  ###40
    kp_jacobian_frame= eval('[%s]'%repr(kp_jacobian_frame).replace('[', '').replace(']', ''))
    kp_integer=kp_value_frame+kp_jacobian_frame ###20+40 
    kp_integer=str(kp_integer)

    seq_kp_integer.append(kp_integer)  
    return seq_kp_integer, kp_value, kp_jacobian

def fv2v_receiving_key_frame(FV2V_Analysis_Model_Estimator,seq_kp_integer,reference):
    kp_cur = FV2V_Analysis_Model_Estimator(reference)  
    ####
    ###  yaw+pttch+roll-->rot mat
    yaw=kp_cur['yaw']
    pitch=kp_cur['pitch']
    roll=kp_cur['roll']
    yaw = headpose_pred_to_degree(yaw)
    pitch = headpose_pred_to_degree(pitch)
    roll = headpose_pred_to_degree(roll)
    kp_rot = get_rotation_matrix(yaw, pitch, roll)            
    kp_rot_list=kp_rot.tolist()
    kp_rot_list=str(kp_rot_list)
    kp_rot_list="".join(kp_rot_list.split())                               

    kp_t=kp_cur['t'] 
    kp_t_list=kp_t.tolist()
    kp_t_list=str(kp_t_list)
    kp_t_list="".join(kp_t_list.split())  

    kp_exp=kp_cur['exp'] 
    kp_exp_list=kp_exp.tolist()
    kp_exp_list=str(kp_exp_list)
    kp_exp_list="".join(kp_exp_list.split())                    

    rot_frame=json.loads(kp_rot_list)###torch.Size([1, 3, 3])
    rot_frame= eval('[%s]'%repr(rot_frame).replace('[', '').replace(']', ''))
    t_frame=json.loads(kp_t_list)  ###torch.Size([1, 3])
    t_frame= eval('[%s]'%repr(t_frame).replace('[', '').replace(']', ''))
    exp_frame=json.loads(kp_exp_list)  ###torch.Size([1, 45])
    exp_frame= eval('[%s]'%repr(exp_frame).replace('[', '').replace(']', ''))
    kp_integer=rot_frame+t_frame+exp_frame ###9+3+45=57
    kp_integer=str(kp_integer)

    seq_kp_integer.append(kp_integer)

    return seq_kp_integer, kp_exp, kp_rot, kp_t



def cfte_receiving_inter_frame(listformat_adptive_CFTE,kp_previous,kp_difference_dec,seq_kp_integer):
    kp_integer=listformat_adptive_CFTE(kp_previous, kp_difference_dec, 1,4)  #####                        
    seq_kp_integer.append(kp_integer)
    kp_integer=json.loads(str(kp_integer))
    kp_current_value=torch.Tensor(kp_integer).to('cuda:0')          
    return seq_kp_integer, kp_current_value


def fomm_receiving_inter_frame(listformat_kp_jocobi_FOMM,kp_previous,kp_difference_dec,seq_kp_integer):
    kp_integer,kp_value,kp_jocobi=listformat_kp_jocobi_FOMM(kp_previous, kp_difference_dec) #######
    seq_kp_integer.append(kp_integer)                    

    # dict={}
    kp_value=json.loads(kp_value)
    kp_current_value=torch.Tensor(kp_value).to('cuda:0')          
    kp_jocobi=json.loads(kp_jocobi)
    kp_current_jocobi=torch.Tensor(kp_jocobi).to('cuda:0')  
    return seq_kp_integer, kp_current_value, kp_current_jocobi

def fv2v_receiving_inter_frame(listformat_kp_mat_exp_FV2V,kp_previous,kp_difference_dec,seq_kp_integer):
    kp_integer,kp_mat_value,kp_t_value,kp_exp_value=listformat_kp_mat_exp_FV2V(kp_previous, kp_difference_dec) #######
    seq_kp_integer.append(kp_integer)
    dict={}                  
    kp_mat_value=json.loads(kp_mat_value)
    kp_current_mat=torch.Tensor(kp_mat_value).to('cuda:0')          
    kp_t_value=json.loads(kp_t_value)
    kp_current_t=torch.Tensor(kp_t_value).to('cuda:0')          
    kp_exp_value=json.loads(kp_exp_value)
    kp_current_exp=torch.Tensor(kp_exp_value).to('cuda:0')          
    return seq_kp_integer, kp_current_mat, kp_current_t, kp_current_exp





