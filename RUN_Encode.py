# +
import os
import sys


seqlist=['001','002','003','004','005','006','007','008','009','010','011','012','013','014','015']
qplist=[ "22", "32", "42", "52"]

    
Sequence_dir='/home/ysz/datasets/testing_sequence_30/'  ###You should download the testing sequence and modify the dir.

height=256
width=256

Mode='Encoder'           ## "Encoder" OR 'Decoder'   ###You need to define whether to encode or decode a sequence.
Iframe_format='YUV420'   ## 'YUV420'  OR 'RGB444' ###You need to define what color format to use for encoding the first frame.
Model = 'CFTE'

if Model=='FV2V':
    quantization_factor=256
if Model=='CFTE':
    quantization_factor=4
if Model=='FOMM':
    quantization_factor=64

testingdata_name = 'CFVQA':
if testingdata_name=='CFVQA':
    frames=125
if testingdata_name=='VOXCELEB':
    frames=250

for qp in qplist:
    for seq in seqlist:
        original_seq=Sequence_dir+testingdata_name+'_'+str(seq)+'_'+str(width)+'x'+str(height)+'_25_8bit_444.rgb'
        if Mode == 'Encoder':
            cmd = "bash ./RUN_Encoder.sh "+Model+" "+Mode+" "+original_seq+" "+str(frames)+" "+str(quantization_factor)+" "+str(qp)+" "+str(Iframe_format)
        if Mode == 'Decoder':
            encoder_type = 'CFTE'
            cmd = "bash ./RUN_Decoder.sh "+Model+" "+Mode+" "+original_seq+" "+str(frames)+" "+str(quantization_factor)+" "+str(qp)+" "+str(Iframe_format)+" "+encoder_type
        os.system(cmd)  
        
        print(Model+"_"+Mode+"_"+seq+"_"+qp+" Finished")
