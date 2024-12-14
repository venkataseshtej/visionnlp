Introduction

This repo is the modified version of the original repo to make the model trainable based on my requirements.
GIT: A Generative Image-to-text Transformer for Vision and Language.

The Original repo for GIT. - https://github.com/microsoft/GenerativeImage2Text

#Installation: 

Install azfuse. The tool is used to automatically download the data. The configuration of AzFuse has already been in this repo.

Download the source code by doing below:

git clone https://github.com/harshalstomp/GenerativeImage2Text.git
cd GenerativeImage2Text
Install the package

pip install -r requirements.txt
python setup.py build develop
- If `prefix` is empty, it is effectively the captioning task.
- If `prefix` is a question, it is effectively the visual question answering task.
- Use a list for `image_path` if it is for video. The example here is 6 identical images, only
  for a demo purpose. It should be different image frames from a video.
- `model_name` here can be the following. 
Performance details can be found in the reference paper.

  | model_name          | Information                                   | Performance             |
  |---------------------|---------------------------------------------- | ----------------------- |
  | GIT_BASE            | pretrained on 4M images                       |                         |
  | GIT_BASE_COCO       | fine-tuned on COCO                            | CIDEr: 131.4            |
  | GIT_BASE_TEXTCAPS   | fine-tuned on TextCaps for captioning         | CIDEr: 64.9             |
  | GIT_BASE_VQAv2      | fine-tuned on VQAv2                           | test-dev: 72.72         |
  | GIT_BASE_TEXTVQA    | fine-tuned on TextVQA                         | val/acc: 18.81          |
  | GIT_BASE_VATEX      | fine-tuned on VATEX for captioning            | public/test/CIDEr: 60.0 |
  | GIT_BASE_MSRVTT_QA  | fine-tuned on MSRVTT for question answering   | acc: 41.0               |
  | GIT_LARGE           | pretrained on 14M images                      |                         |
  | GIT_LARGE_COCO      | fine-tuned on COCO                            | CIDEr: 138.5            |
  | GIT_LARGE_TEXTCAPS  | fine-tuned on TextCaps for captioning         | CIDEr: 106.3            |
  | GIT_LARGE_VQAv2     | fine-tuned on VQAv2                           | test-dev: 75.51         |
  | GIT_LARGE_TEXTVQA   | fine-tuned on TextVQA                         | val/acc: 37.47          |
  | GIT_LARGE_VATEX     | fine-tuned on VATEX for captioning            | public/test/CIDEr: 72.5 |
  | GIT_LARGE_MSRVTT_QA | fine-tuned on MSRVTT for question answering   | acc: 42.7               |

Inference on a TSV file, which is a collection of multiple images.

Data format (for information only)
image TSV: Each row has two columns. The first is the image key; the second is base64-encoded jpg or png bit string.

caption or question tsv: Each row has two columns. The first is the image key; the second is a list of dictionaries in the json format. For caption TSV, the dictionary should contain at least the field of 'caption'. For the question answering TSV, it should contain at least question_id and question.

inference on COCO Karpathy test.
# Inference.

# base
AZFUSE_TSV_USE_FUSE=1 python -m generativeimage2text.inference -p "{'type': 'test_git_inference_single_tsv', \
      'image_tsv': 'data/coco_caption/test.img.tsv', \
      'model_name': 'GIT_BASE_COCO', \
      'question_tsv': null, \
      'out_tsv': 'inference/GIT_BASE_COCO/coco.tsv', \
}"
# GIT_LARGE_COCO. If there are 8 GPUs, it can parallel by mpirun -n 8
AZFUSE_TSV_USE_FUSE=1 mpirun -n 8 python -m generativeimage2text.inference -p "{'type': 'test_git_inference_single_tsv', \
      'image_tsv': 'data/coco_caption/test.img.tsv', \
      'model_name': 'GIT_LARGE_COCO', \
      'question_tsv': null, \
      'out_tsv': 'inference/GIT_LARGE_COCO/coco.tsv', \
}"
#Calculate the evaluation metric

# base
AZFUSE_TSV_USE_FUSE=1 python -m generativeimage2text.inference -p "{'type': 'evaluate_on_coco_caption', \
      'res_file': 'inference/GIT_BASE_COCO/coco.tsv', \
      'label_file': 'data/coco_caption/test.caption.tsv', \
}"
The CIDEr score should be 131.35 for GIT_BASE_COCO and 138.45 for GIT_LARGE_COCO. If you get lower score (e.g. 126 for the base model), the reason could be the misalignment of the environment, e.g. pytorch version.

# Steps to Train the model:

The repo shows the key code path of constructing the network input with transformations and forward/backward. The code can be plugged into any trainer easily. Here is the example for the base model.

Train for Image Captioning

Run all the cells in git_train.ipynb.

The git_train file has all the steps to run from installing the required files, to downloading the dataset, to training the model and running the evaluation scripts as well.
Download the dataset, extract them in the dataset directory, and change the paths in the configs.
Run the inference after training the model and then run the evaluation code to get the scores of the model.

python lightning_trainer.py

#Citation
Please consider to cite the following reference if it helps.

@article{wang2022git,
  title={GIT: A Generative Image-to-text Transformer for Vision and Language},
  author={Wang, Jianfeng and Yang, Zhengyuan and Hu, Xiaowei and Li, Linjie and Lin, Kevin and Gan, Zhe and Liu, Zicheng and Liu, Ce and Wang, Lijuan},
  journal={arXiv preprint arXiv:2205.14100},
  year={2022}
}
Acknowledgement
Part of the code is based on transformers, clip, maskrcnn-benchmark, oscar, virtex.

Contributing
This project welcomes contributions and suggestions. Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the Microsoft Open Source Code of Conduct. For more information see the Code of Conduct FAQ or contact opencode@microsoft.com with any additional questions or comments.

Trademarks
This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow Microsoft's Trademark & Brand Guidelines. Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party's policies.
