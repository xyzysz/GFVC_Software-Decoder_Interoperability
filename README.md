# GFVC_Software-Decoder_Interoperability

+ The code is adapted from [GFVC_Software](https://github.com/Berlin0610/GFVC_Software/tree/main). The GFVC checkpoints and overall testing dataset can be found in their page.
+ With three leading GFVC works, i.e., [FOMM](https://github.com/AliaksandrSiarohin/first-order-model), [CFTE](https://github.com/Berlin0610/CFTE_DCC2022) and [FV2V](https://github.com/zhanglonghao1992/One-Shot_Free-View_Neural_Talking_Head_Synthesis) integrated in GFVC_Software, we use face parameter translator to enable decodeing interoperability for fixed decoder.

## Encoding/Decoding Porcesses
The platform details can be described as follows,
-	The pretrained analysis/synthesis models and codes of the three representative GFVC algorithms are encapsulated in the `GFVC` folder. 
-	The corresponding interfaced functions regarding the encoder and decoder are defined in `CFTE_Encoder.py`, `CFTE_Decoder.py`, `FOMM_Encoder.py`, `FOMM_Decoder.py`, `FV2V_Encoder.py` and `FV2V_Decoder.py`.
-	**The face parameter translators are defined in `Parametertranscoder.py` for translation between each of three different types of features extracted by corresponding GFVC models.**
-	The `arthmetic` and `vtm` folders include the packaged tools regarding the context-adaptive arithmetic coder for feature parameter encoding and the latest VVC software VTM 22.2 for base picture encoding.
-	The shell file (i.e., `RUN.sh` ) and batch execution code (i.e., `RUN.py` ) are provided to complete the encoding and decoding processes.

# BibTeX
```
@inproceedings{yin2024,
            title={Enabling Translatability of Generative Face Video Coding: A Unified Face Feature Transcoding Framework},
            author={Shanzhi, Yin and Bolin, Chen and Shiqi, Wang and Yan, Ye},
            journal={Data Compression Conference (DCC)},
            year={2024}
          }
```
