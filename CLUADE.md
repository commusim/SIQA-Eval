此项目用于SIQA的开源处理：
应该包括以下几个部分:
1. 评测工具的一键部署，一键测试。
2. 论文中SIQA-S Baseline传统方法的部署
3. 相关Readme.md说明，以及各类工具的运用

你可以查阅的资料有：
huggingface数据集：https://huggingface.co/datasets/SIQA/TrainSet
huggingface上微调的模型：https://huggingface.co/commusim-hf/SIQA-Finetune/tree/main
之前比赛配套的评测工具： C:\Code\SIQA\Evaluate-Pipeline
比赛相关的资料：C:\Code\SIQA\SIQA_Website
SIQA-S所用基线相关的复现代码：C:\Users\commusim\Desktop\SIQA-Baseline（注意，此部分分为模型权重和调用代码等等，代码已经经过魔改适配）



此项目应该分为以下结构：
/data： 用于存放下载的数据
/model: 用于存放下载的模型以及SIQA-Baseline相关的模型
/eval：  分别对SIQA-U和SIQA-S进行评估。
/outputs: 存放评估过程中输出的结果文件
/results： 存放最终的评测结果
