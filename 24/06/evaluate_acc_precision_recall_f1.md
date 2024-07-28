# 深度学习二分类分类评估详细解析与代码实战

深度学习二分类的实战代码：使用 Trainer API 微调模型. https://huggingface.co/learn/nlp-course/zh-CN/chapter3/3

> 如果你刚接触 自然语言处理，huggingface 是你绕不过去的坎。但是目前它已经被墙了，相信读者的实力，自行解决吧。

设置代理，如果不设置的话，那么huggingface的包无法下载；

```python
import os
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
```

在探讨二分类问题时，经常会遇到四种基本的分类结果，它们根据样例的真实类别与分类器的预测类别来定义。以下是对这些分类结果的详细解释：

这四个定义均由两个字母组成，它们各自代表了不同的含义。

> 第一个字母（True/False）用于表示算法预测的正确性，而第二个字母（Positive/Negative）则用于表示算法预测的结果。

- **第1个字母**（True/False）：描述的是分类器是否预测正确。True表示分类器判断正确，而False则表示分类器判断错误。
- **第2个字母**（Positive/Negative）：表示的是分类器的预测结果。Positive代表分类器预测为正例，而Negative则代表分类器预测为负例。

1. **真正例（True Positive，TP）**：当样例的真实类别为正例时，如果分类器也预测其为正例，那么我们就称这个样例为真正例。简而言之，真实情况与预测结果均为正例。
2. **假正例（False Positive，FP）**：有时，分类器可能会将真实类别为负例的样例错误地预测为正例。这种情况下，我们称该样例为假正例。它代表了分类器的“过度自信”或“误报”现象。
3. **假负例（False Negative，FN）**：与假正例相反，假负例指的是真实类别为正例的样例被分类器错误地预测为负例。这种情况下的“遗漏”或“漏报”是分类器性能评估中需要重点关注的问题。
4. **真负例（True Negative，TN）**：当样例的真实类别和预测类别均为负例时，我们称其为真负例。这意味着分类器正确地识别了负例。



## 数据准备

做深度学习的同学应该都默认装了 torch，跳过 torch的安装

```python
!pip install evaluate
```

### 导包

```python
import torch
import random
import evaluate
```


随机生成二分类的预测数据 pred 和 label；

```python
label = torch.tensor([random.choice([0, 1]) for i in range(20)])
pred = torch.tensor([random.choice([0, 1, label[i]]) for i in range(20)])
sum(label == pred)
```

下述是随机生成的 label 和 pred

```python
# label
tensor([0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0])

# pred
tensor([0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0])
```

使用 `random.choice([0, 1, label[i]]` 是为了提高 pred 的 准确率； 因为 `label[i]` 是真实的 label；

下述的是计算TP、TN、FP、FN的值：

> Tips: 
>
> pred : 与**第2个字母**（Positive/Negative）保持一致，
>
> label: 根据第一个字母是否预测正确，再判断填什么

```python
TP = sum((label == 1) & (pred == 1))
TN = sum((label == 0) & (pred == 0))
FP = sum((label == 0) & (pred == 1))
FN = sum((label == 1) & (pred == 0))
```


| 标签 | Value |
| ---- | ----- |
| TP   | 6     |
| TN   | 8     |
| FP   | 2     |
| FN   | 4     |



## 准确率 Accuracy

准确率（Accuracy）: 分母通常指的是所有样本的数量，即包括真正例（True Positives, TP）、假正例（False Positives, FP）、假负例（False Negatives, FN）和真负例（True Negatives, TN）的总和。而分子中的第一个字母为“T”（True），意味着我们计算的是算法预测正确的样本数量，即TP和TN的总和。

然而，准确率作为一个评价指标存在一个显著的缺陷，那就是它对数据样本的均衡性非常敏感。当数据集中的正负样本数量存在严重不均衡时，准确率往往不能准确地反映模型的性能优劣。

例如，假设有一个测试集，其中包含90%正样本和仅10%负样本。若模型将所有样本都预测为正样本，那么它的准确率将轻松达到90%。从准确率这一指标来看，模型似乎表现得非常好。但实际上，这个模型对于负样本的预测能力几乎为零。

因此，在处理样本不均衡的问题时，需要采用其他更合适的评价指标，如精确度（Precision）、召回率（Recall）、F1分数（F1 Score）等，来更全面地评估模型的性能。这些指标能够更准确地反映模型在各类样本上的预测能力，从而帮助我们做出更准确的决策。

精准率的公式如下：
$$
Accuracy = \frac{TP + TN}{TP + TN + FP +FN} = \frac{TP + TN}{所有样本数}
$$

```python
accuracy = evaluate.load("accuracy")
accuracy.compute(
        predictions=pred, 
        references=label
    )
```

Output:

```python
{'accuracy': 0.7}
```

下述三种方法都可以用来计算 `accuracy`:

```python
print(
    (TP + TN) / (TP + TN + FP +FN),
    (TP + TN) / len(label),
    sum((label == pred)) / 20
)
```

Output:

```python
tensor(0.7000) tensor(0.7000) tensor(0.7000)
```



使用公式计算出来的与通过`evaluate`库，算出来的结果一致，都是 0.7。

## precision 精准率

$$
Precision = \frac{TP}{TP + FP}
$$



```python
precision = evaluate.load("precision")
precision.compute(
        predictions=pred, 
        references=label
    )
```

Output:

```python
{'precision': 0.75}
```

```python
TP / (TP + FP)
```



### recall 召回率

$$
Recall = \frac{TP}{TP + FN}
$$



```python
recall = evaluate.load("recall")
recall.compute(
        predictions=pred, 
        references=label
    )
```
Output:

```python
{'recall': 0.6}
```




```python
TP / (TP + FN)
```



## F1

```python
f1 = evaluate.load("f1")
f1.compute(
        predictions=pred, 
        references=label
    )
```

Output:

```python
{'f1': 0.6666666666666666}
```


$$
F1 = \frac{2 \times {Precision} \times {Recall}}{{Precision} + {Recall}}
$$


```python
2 * 0.7500 * 0.6000 / (0.7500 + 0.6000)
```

Output:

```python
0.6666666666666665
```



> 希望这篇文章，通过代码实战，能够帮到你加深印象与理解！概念说再多遍，不如代码实现一遍。

## 参考资料

* 如何在python代码中使用代理下载Hungging face模型. https://www.jianshu.com/p/209528bed023 
* [机器学习] 二分类模型评估指标---精确率Precision、召回率Recall、ROC|AUC. https://blog.csdn.net/zwqjoy/article/details/78793162
* 使用 Trainer API 微调模型. https://huggingface.co/learn/nlp-course/zh-CN/chapter3/3
* Huggingface Evaluate 文档. https://huggingface.co/docs/evaluate/index





