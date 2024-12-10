# Prediction of PD-L1 tumor positive score in lung squamous cell carcinoma with H&E staining images and deep learning

Qiushi Wang, Xixiang Deng, Pan Huang, Qiang Ma, Lianhua Zhao, Yangyang Feng, Yiying Wang, Yuan Zhao, Yan Chen, Peng Zhong, Peng He, Mingrui Ma, Peng Feng and Hualiang Xiao


***
# Introduction
Background: Detecting programmed death ligand 1 (PD-L1) expression based on immunohistochemical (IHC) staining is an important guide for the treatment of lung cancer with immune checkpoint inhibitors. However, this method has problems such as high staining costs, tumor heterogeneity, and subjective differences among pathologists. Therefore, the application of deep learning models to segment and quantitatively predict PD-L1 expression in digital sections of Hematoxylin and eosin (H&E) stained lung squamous cell carcinoma is of great significance. Methods: We constructed a dataset comprising H&E-stained digital sections of lung squamous cell carcinoma and used a Transformer Unet (TransUnet) deep learning network with an encoder-decoder design to segment PD-L1 negative and positive regions and quantitatively predict the tumor cell positive score (TPS). Results: The results showed that the dice similarity coefficient (DSC) and intersection overunion(IoU) of deep learning for PD-L1 expression segmentation of H&E-stained digital slides of lung squamous cell carcinoma were 80% and 72%, respectively, which were better than the other seven cutting-edge segmentation models. The root mean square error (RMSE) of quantitative prediction TPS was 26.8, and the intra-group correlation coefficients with the gold standard was 0.92 (95% CI: 0.90–0.93), which was better than the consistency between the results of five pathologists and the gold standard. Conclusion: The deep learning model is capable of segmenting and quantitatively predicting PD-L1 expression in H&E-stained digital sections of lung squamous cell carcinoma, which has significant implications for the application and guidance of immune checkpoint inhibitor treatments.


---
![image](https://github.com/Aster-wang/TransUnet_for_Predicting_PD_L1_from_HE/blob/main/1.PNG)
Fig.1. Overview of our work 


# My Datasets link
https://drive.google.com/drive/folders/1TUHCGv414OEexQvvuOA289MbRDRLPRmi?usp=drive_link
