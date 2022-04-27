# MR-VAE
“一种基于MR-VAR的低照度图像增强方法”[1]  
Low-illumination Image Enhancement Based on MR-VAE
![images](https://github.com/csxuwu/MR-VAE/blob/master/images/MR-VAE.png)

## Abstract
针对低照度图像多重失真特点（低亮度、多噪声和模糊等），本文基于变分自编码器提出了一种多重构变分自编码器 （Multiple Reconstruction-Variational AutoEncoder,MR-VAE），逐步增强、从粗到细地生成高质量低照度增强图像。MR-VAE 由特征概率分布捕获、全局重构和细节重构三个模块构成，核心思想是将全局特征与局部特征分阶段重建、将多重失真问题逐步解决，全局重构模块构建图像全局特征，提高全局亮度，得到较粗糙的图像；细节重构模块权衡去噪与去模糊，生成细节更逼真、噪声更少与局部亮度更合适的图像；此外，本文定义了一个多项损失函数替代𝑙2损失，以引导网络生成高质量图像。实验结果表明，多重构与多项损失函数的设计提高了网络生成复杂图像、处理多重失真的低照度图像性能，且提高了生成图像的质量、信噪比和视觉特性。  
**关键词** 低照度图像增强；多重构；多项损失；多重失真；变分自编码；残差网络

According to our investigation, low-illumination images have multiple distortion characteristics (including low light, high noise, blur, etc.). In order to better enhance the low-illumination image, we propose a Multi-Reconstruction Variational AutoEncoder(MR-VAE) based on the Variational AutoEncoder(VAE) to gradually enhance the image and generate high-quality low-illumination enhanced image from coarse to fine. MR-VAE is mainly composed of three modules: Feature Probability Distribution Capture (FPDC), Global Reconstruction(GR),  and Detail Reconstruction(DR). Its key idea is to generate the global and local features of the image in stages, and solve the problem of multiple distortion step by step. Finally, MR-VAE can capture low-illumination to normal illumination, noisy to noiseless, fuzzy to clear composite nonlinear mapping. The FPDC module is mainly used to capture hidden variables that cover the entire image feature, which is equivalent to encoding the entire image. The GR module is mainly used to capture the nonlinear mapping from low-illumination to normal illumination and uses the hidden variables to generate global features of the image step by step (global features include: scene, color distribution, illumination characteristics), and finally obtain appropriate brightness enhancement, rough quality image. The DR module is mainly used to capture composite nonlinear maps from noisy to noiseless and clear to fuzzy, finding a good balance between removing noise and retaining detailed information, helping the network to generate high-quality images which detail more realistic, less noisy, and more suitable for local brightness. More importantly, we redefine a multi-loss function to replace the L2 loss function, which improves the image quality criteria and guides the network to generate high-quality images. This multi-loss function consists of "hidden variable content loss", "global reconstruction loss" and "detail reconstruction loss". Hidden variable content loss helps the FPDC module capture hidden variables that better reflect the probability distribution of the essential content of the image. Global reconstruction loss is used to help the GR module generate large image features and overall brightness. Detail reconstruction loss helps the DR to generate detailed features between rough and high-resolution images and helps DR to be more robust to noise. In addition, based on the MS COCO dataset, we have created pairs of lowlight images for training our network. Gamma correction is used to adjust the brightness, including four different brightness levels. Gaussian noise is used to simulate actual noise, the noise level, and brightness level meet a certain relationship. Gaussian blur is used to simulate the blur of the image to produce a composite image that is very close to the actual low-illumination image. In the experimental stage, We did a complete experiment, including the ablation experiment of the network structure and the comparison experiment with other methods. The ablation experiment is mainly to verify the depth of the network structure and the role of the multi-loss function. The comparison experiment includes the traditional method and the deep learning methods. The experimental results show that the multi-reconstruction and multi-loss function can help the network to generate complex images and improve the network's enhanced performance for low-intensity images with multiple distortions. Indicating that our method can better enhance low-illumination images and has a better ability to remove noise. 

**Keywords**: Low-illumination Image Enhancement; Multiple reconstruction; Multi-loss function;Multiple distortion; Variational AutoEncoder; Residual network

## Codes
### Train
run "main_for_mrvae_v2_1.py"
### Test
run "eval_for_mrvae_v2_1.py" in the eval.

## Citation
[1]江泽涛,伍旭,张少钦.一种基于MR-VAE的低照度图像增强方法[J].计算机学报,2020,43(07):1328-1339.


