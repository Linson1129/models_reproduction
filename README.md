项目概述
本项目基于MMSA(Multi-Modal Sentiment Analysis)框架，在CH-SIMS中文多模态情感数据集上对多种主流模型进行了系统的复现、对比和优化实验。

实验模型
我们共复现和对比了6个主流多模态情感分析模型：

模型	        全称	                                    核心创新
TFN	        Tensor Fusion Network	                    张量融合策略捕捉模态间交互
LMF	        Low-rank Multimodal Fusion	                低秩分解降低计算复杂度
MFN	        Memory Fusion Network	                    跨模态记忆网络建模时序依赖
MULT	    Multimodal Transformer	                    跨模态注意力机制
MISA	    Multimodal Interactive Sentiment Analysis	模态公共/私有空间分离
BERT-MAG	BERT Multimodal Adaptation Gate	            在BERT中注入多模态门控

数据集
CH-SIMS             中文多模态情感数据集
属性	            数值
版本	            unaligned_39
模态	            文本(T)、音频(A)、视频(V)
特征维度            [768, 33, 709]
序列长度            [39, 400, 55]
训练样本            1368
分类数	            3 (负类/中性/正类)
语言	            中文

实验设计

1. 多模型对比实验
所有模型在相同数据集、相同评估指标下进行公平对比
每个模型使用相同的训练/验证/测试划分

2. 随机种子敏感性分析
每个模型至少测试2个随机种子(1111, 1112)
计算均值±标准差，评估模型稳定性

3. 模型优化迭代
对TFN、MFN等模型进行了参数调优
对比初始版本与优化版本的性能提升

实验结果
排名    	模型	 F1分数	    二分类	  三分类
1	    MULT	     80.79%	   80.80%	 74.03%
2	    TFN     	 80.46%	   80.46%	 72.92%
3	    MFN	         80.09%	   80.03%	 71.52%
4	    MISA	     77.34%	   77.32%	 68.23%
5	    BERT-MAG	 75.82%	   75.82%	 69.44%
6	    LMF	         72.45%	   72.60%	 62.83%

主要发现
1. 模型性能梯队
第一梯队：MULT、TFN(opt)、MFN(opt) → F1 > 80%
第二梯队：MISA、BERT-MAG → F1 75-78%
第三梯队：LMF、TFN(base) → F1 < 73%

2. 优化效果显著
TFN经过优化后，F1从72.83%提升至80.46%（+7.63%）
MFN经过优化后，F1从78.79%提升至80.09%（+1.30%）

3. 中性类识别挑战
所有模型在中性类上的召回率（68-73%）均低于正负类（75-81%）
中性情感特征模糊是共同难点

4. 种子敏感性
MULT、TFN对随机种子较敏感（标准差0.5-0.8）
BERT-MAG最稳定（标准差<0.2）

配置文件
为每个模型生成标准JSON配置：
configs
    |——lmf_sims.json
    |——tfn_sims.json
    |——mfn_sims.json
    |——mult_sims.json
    |——misa_sims.json
    |——bert_mag_sims.json

快速开始
# 训练示例
python train.py --config configs/mult_sims.json

# 结果查看
python view_results.py