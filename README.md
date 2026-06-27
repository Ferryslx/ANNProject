# ANNProject

基于 PyTorch 的人工神经网络（ANN）实现手机价格分类预测。

## 项目概述

使用多层神经网络对手机进行价格区间分类预测（4个价格等级），涵盖完整的深度学习流程：数据预处理、模型构建、训练、评估与保存。

## 项目结构

```
├── data/
│   └── 手机价格预测.csv    # 数据集（20维特征，4类价格标签）
├── src/
│   └── ANNProject.py       # 主程序：神经网络定义、训练与测试
├── utils/
│   └── log.py              # 日志记录器封装
├── model/
│   └── phone.pth           # 训练好的模型参数
├── log/
│   └── ANNModel*.log        # 训练日志
└── README.md
```

## 神经网络架构

| 层 | 输入 → 输出 | 激活函数 |
|---|---|---|
| 全连接层 1 | 20 → 128 | ReLU |
| 全连接层 2 | 128 → 256 | ReLU |
| 全连接层 3 | 256 → 512 | ReLU |
| 全连接层 4 | 512 → 128 | ReLU |
| 输出层 | 128 → 4 | Softmax（在损失函数中） |

## 优化策略

1. **优化器**: SGD → Adam（自适应学习率）
2. **学习率**: 0.001 → 0.0001
3. **数据标准化**: StandardScaler 归一化
4. **网络深度**: 增加至 5 层全连接网络
5. **训练轮数**: 100 epochs

## 依赖

- Python 3.x
- PyTorch
- scikit-learn
- pandas / numpy
- matplotlib
- torchsummary

## 使用

```bash
# 训练模型
python src/ANNProject.py

# 或使用数据预处理工具
python utils/common.py
```

训练完成后模型参数保存至 `model/phone.pth`，训练日志输出至 `log/` 目录。

## 结果

模型在测试集上输出分类准确率，详细训练过程和准确率记录在日志文件中。
