# 🧮 Handwritten Mathematical Formula Recognition

基于 **Seq2Seq（Encoder-Decoder + Attention）** 的手写数学公式识别项目，将手写公式图像识别为 LaTeX 序列。

本项目参考模型为 [ABM](https://github.com/XH-B/ABM)（DenseNet 编码器 + GRU 解码器 + 注意力机制），支持 L2R / R2L / 双向解码。

## ✨ 功能特性

- 📐 端到端公式识别：图像 → LaTeX 序列
- 🔄 支持从左到右（L2R）、从右到左（R2L）及双向解码策略
- 🧠 DenseNet 编码器 + GRU 解码器 + Attention（`encoder.py` / `decoder.py` / `encoder_decoder.py`）
- 📝 词表驱动解码，支持 PAD/EOS 等特殊标记
- 📊 训练/验证脚本 + WER 评估（`train.py` / `compute-wer.py`）
- 🌐 提供 Flask 推理 API（`api_server.py`）
- 🚀 支持 Render 云端部署（`render.yaml`）

## 📁 项目结构

```
Handwritten-Mathematical-Formula-Recognition-main/
├── encoder.py            # DenseNet 编码器
├── decoder.py            # GRU 解码器
├── encoder_decoder.py    # 整体 Encoder-Decoder 模型
├── train.py              # 训练入口（python train.py <result_path>）
├── train.sh / test.sh    # 训练/测试脚本
├── test.py               # 测试脚本
├── compute-wer.py        # WER 词错误率评估
├── gen_pkl copy.py       # 数据预处理/打包脚本【待确认用途】
├── api_server.py         # Flask 推理 API
├── analize_data.py       # 数据分析脚本
├── package/              # 工具包（data_loader.py / utils.py）
├── render.yaml           # Render 部署配置
└── pdf/                  # SURF 海报与证书（docs 性质）
```

## 🚀 快速开始

### 环境依赖

- Python ≥ 3.6
- PyTorch ≥ 1.7
- CUDA（GPU 训练推荐，也支持 CPU 推理）

```bash
pip install -r requirements.txt   
```

### 训练

```bash
# result_path 为输出目录（训练脚本通过命令行参数传入）
python train.py ./result_path
```

> 模型关键超参在 `train.py` 内配置（growthRate=24、D=684、n/m=256、K=113、dim_attention=512 等）。

### 推理 / API

```bash
python api_server.py   # 启动 Flask 服务（默认 5000 端口）
# POST /recognize，body 传 {"image": "图片URL"}
```

### 数据集

- 项目面向 **CROHME** 类手写公式数据集（`data/train/images`、`data/val/images` 目录结构约定见 `train.py`）。

## 🏆 项目背景

- 负责使用 seq2seq 方法进行手写数学公式识别。
- 参考论文/代码：[ABM](https://github.com/XH-B/ABM)
  

## 📄 许可

未指定开源许可（默认保留所有权利）。
