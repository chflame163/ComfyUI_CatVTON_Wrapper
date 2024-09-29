# ComfyUI_CatVTON_Wrapper

CatVTON warpper for ComfyUI. fixed the cropping issue of images with different proportions input.    

Original repo: [https://github.com/Zheng-Chong/CatVTON](https://github.com/Zheng-Chong/CatVTON)    

It is recommend use 6G or higher VRAM NVidia GPU device to run.

![image](workflow/catvton_example.jpg)     

### Install Plugin
Open the cmd window in the plugin directory of ComfyUI, like ComfyUI\custom_nodes，type:
```
git clone https://github.com/chflame163/ComfyUI_CatVTON_Wrapper.git
```

### Install Dependency Packages
Open the cmd window in the ComfyUI_CatVTON_Wrapper plugin directory like ComfyUI\custom_ Nodes\ComfyUI_CatVTON_Wrapper and enter the following command,    

For ComfyUI official portable package, type:
```
..\..\..\python_embeded\python.exe -s -m pip install -r requirements.txt
```
For Aki ComfyUI package, type:
```
..\..\python\python.exe -s -m pip install -r requirements.txt
```
### Download Models
Download all model files from [BaiduNetdisk](https://pan.baidu.com/s/1xbL5evB6iiav3QozsL0fJQ?pwd=06nv) or [GoogleDrive](https://drive.google.com/drive/folders/1TJNNql7UfDPVgHJuItDDjowycN5jpC5o?usp=sharing) and copy to ```ComtyUI/models/CatVTON``` directory. 

### Example Workflow
The example workflow files is in the workflow directory.

### Node Options
![image](image/catvton_wrapper_node.jpg)      
* image: The input image.
* mask: The mask of input image, clothing within the mask range will be repaint.
* refer_image: The reference image of repaint clothing.
* mask_grow: The size of the mask expansion. If the repaint clothing style is incorrect, try adjusting this value.
* mixed_precision: fp32, fp16, bf16 can be selected, the model will run under the selected data type.
* seed: The random seed of sampler.
* steps: The number of steps of sampler.
* cfg: The cfg strength of the diffusion model.

### Statement
Please refer to the original project license to authorization agreement.

