---
title: TensorFlow学习笔记(2)
date: 2017-07-21 17:55:15
tags: [TensorFlow, 深度学习]
---

![](https://pbs.twimg.com/media/DAJBiP5VwAAk1iK.jpg)

# TensorFlow学习笔记

> 借助 ***迁移学习（Transfer Learning）*** 知识，使用**Inception v3**模型对我们已有的样本图片进行再训练，得到指定图片样本集下的分类模型

## 1.简单介绍

该笔记将介绍如果借助TensorFlow中给出的例子，很快解决我们实际工作中亟待解决的图像识别或者分类问题（Image Recognition）。其实TensorFlow官方文档已经给出了非常详细的说明（参考资料2，3），只是缺少了使用再次训练的模型（Retrained Model），进行分类测试和对模型评估的操作过程（官方文档有说明，但是针对第一次接触TensorFlow的小白用户，还是不够详细）。首次接触需要了解的背景知识：

- [ImageNet](http://www.image-net.org/)：Inception-v3模型依赖的样本集合
- "Model"前辈：[QuocNet(2012)](http://static.googleusercontent.com/media/research.google.com/en//archive/unsupervised_icml2012.pdf)，[AlexNet](http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf)，[Inception(GoogLeNet)(2014)](https://arxiv.org/abs/1409.4842)，[BN-Inception-v2](https://arxiv.org/abs/1502.03167)
- [Inception-v3](https://arxiv.org/abs/1512.00567)：具体和前辈们有什么差别，......，Google讲的比我好
- [Inception-v4](https://arxiv.org/abs/1602.07261)：v3还没有研究清楚，v4又于2016年2月横空出世

## 2.获取样本集

1.下载样本集，样本集描述：**”An archive of creative-commons licensed flower photos“**

```Python
curl -O http://download.tensorflow.org/example_images/flower_photos.tgz
tar xzf flower_photos.tgz
```

> **注意**：个人电脑2015MBP（I5 CPU）在这样的一个样本集上，花费了大约20+的时间，请耐心等待。。。

2.为了尽快见到结果，我们可以对训练集瘦身，以下操作可以减少训练集的70%


```python
ls flower_photos/roses | wc -l
rm flower_photos/*/[3-9]*
ls flower_photos/roses | wc -l
```
## 3.训练“盗梦空间（Inception）”

> Inception这个词实在不知道该怎么翻译，本意“初始，开始”，也是电影《盗梦空间》的原名

1.下载训练代码

```python
curl -O https://raw.githubusercontent.com/tensorflow/tensorflow/r1.1/tensorflow/examples/image_retraining/retrain.py
```

> **As noted in the introduction, Inception is a huge image classification model with millions of parameters that can differentiate a large number of kinds of images. We're only training the final layer of that network, so training will end in a reasonable amount of time.**

2.运行tensorboard，以便随时监控训练进程

```python
tensorboard --logdir training_summaries &
```

3.开始运行训练脚本

```python
python retrain.py \
  --bottleneck_dir=bottlenecks \
  --how_many_training_steps=500 \
  --model_dir=inception \
  --summaries_dir=training_summaries/basic \
  --output_graph=retrained_graph.pb \
  --output_labels=retrained_labels.txt \
  --image_dir=flower_photos
```

4.模型训练的第一阶段：分析样本图片，计算各个图片之间的**Bottleneck**值


![](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/img/1395d9d6ff9d3e43.png)

Inception v3模型是由许多层堆叠在一起组合而成的，如上图TensorBoard中所有曾现的一样，具体内容可以参考资料4。这些层都是训练过的（pre-trained），已经具备了发现和汇总有效信息的价值意义，进而能够实现多数图片的类别划分。我们训练过程的意义在于实现最终分类模型中的最后一层（**final_training_ops**如下图所示）。

![](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/img/84a6154ed64fd0fb.png)

上图中，左侧标识`softmax`的节点是原始模型的输出层。而右侧的这些节点是在训练过程中添加的，其余的节点是训练结束后由训练脚本自动探测添加的。

A "Bottleneck", is an informal term we often use for the layer just before the final output layer that actually does the classification. As, near the output, the representation is much more compact than in the main body of the network.

Every image is reused multiple times during training. Calculating the layers behind the bottleneck for each image takes a significant amount of time. Since these lower layers of the network are not being modified their outputs can be cached and reused.

So the script is running the constant part of the network, everything below the node labeled `Bottlene...` above, and caching the results.

The command you ran saves these files to the `bottlenecks/` directory. If you rerun the script, they'll be reused, so you don't have to wait for this part again.


5.模型训练等待的过程中：**Training**


Once the script finishes generating all the bottleneck files, the actual training of the final layer of the network begins.

The training operates efficiently by feeding the cached value for each image into the Bottleneck layer. The true label for each image is also fed into the node labeled `GroundTruth.` Just these two inputs are enough to calculate the classification probabilities, training updates, and the various performance metrics.

As it trains you'll see a series of step outputs, each one showing training accuracy, validation accuracy, and the cross entropy:

- The **training accuracy** shows the percentage of the images used in the current training batch that were labeled with the correct class.
- **Validation accuracy**: The validation accuracy is the precision (percentage of correctly-labelled images) on a randomly-selected group of images from a different set.
- **Cross entropy** is a loss function that gives a glimpse into how well the learning process is progressing (lower numbers are better here).

The figures below show an example of the progress of the model's accuracy and cross entropy as it trains. If your model has finished generating the bottleneck files you can check your model's progress by [opening TensorBoard](http://0.0.0.0:6006/), and clicking on the figure's name to show them. TensorBoard may print out warnings to your command line. These can safely be ignored.

![](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/img/bc758910e1c6eee7.png)

![](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/img/10b16d75eff9b4da.png)

![](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/img/ee5d91da57a84831.png)

A true measure of the performance of the network is to measure its performance on a data set that is not in the training data. This performance is measured using the validation accuracy. If the training accuracy is high but the validation accuracy remains low, that means the network is overfitting, and the network is memorizing particular features in the training images that don't help it classify images more generally.

The training's objective is to make the cross entropy as small as possible, so you can tell if the learning is working by keeping an eye on whether the loss keeps trending downwards, ignoring the short-term noise.

By default, this script runs 4,000 training steps. Each step chooses 10 images at random from the training set, finds their bottlenecks from the cache, and feeds them into the final layer to get predictions. Those predictions are then compared against the actual labels to update the final layer's weights through a back-propagation process.

As the process continues, you should see the reported accuracy improve. After all the training steps are complete, the script runs a final test accuracy evaluation on a set of images that are kept separate from the training and validation pictures. This test evaluation provides the best estimate of how the trained model will perform on the classification task.

You should see an accuracy value of between 85% and 99%, though the exact value will vary from run to run since there's randomness in the training process. (If you are only training on two classes, you should expect higher accuracy.) This number value indicates the percentage of the images in the test set that are given the correct label after the model is fully trained.

## 4.使用Retrained Model分类

模型训练结束后，将要使用的两个文件分别是`tf_files/retrained_graph.pb`(A version of the Inception v3 network with a final layer retrained to your categories)，以及包含类别标识的文本文件`tf_files/retrained_labels.txt`。

***label_image.py***

```python
import os, sys

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# change this as you see fit
image_path = sys.argv[1]

# Read in the image_data
image_data = tf.gfile.FastGFile(image_path, 'rb').read()

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("retrained_labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    predictions = sess.run(softmax_tensor, \
             {'DecodeJpeg/contents:0': image_data})

    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        print('%s (score = %.5f)' % (human_string, score))
```

label_image.py的使用方法：

```python
python label_image.py flower_photos/roses/test.jpg
```

类似的结果输出如下：

```python
daisy (score = 0.99071)
sunflowers (score = 0.00595)
dandelion (score = 0.00252)
roses (score = 0.00049)
tulips (score = 0.00032)
```

## 5.参数调整（Try other Hyperparameters）

`retrain.py`支持多种参数设置，以便我们对训练结果进行调优。

获得帮助：

```python
python retrain.py -h
```

输出如下：

```python
usage: retrain.py [-h] [--image_dir IMAGE_DIR] [--output_graph OUTPUT_GRAPH]
                  [--intermediate_output_graphs_dir INTERMEDIATE_OUTPUT_GRAPHS_DIR]
                  [--intermediate_store_frequency INTERMEDIATE_STORE_FREQUENCY]
                  [--output_labels OUTPUT_LABELS]
                  [--summaries_dir SUMMARIES_DIR]
                  [--how_many_training_steps HOW_MANY_TRAINING_STEPS]
                  [--learning_rate LEARNING_RATE]
                  [--testing_percentage TESTING_PERCENTAGE]
                  [--validation_percentage VALIDATION_PERCENTAGE]
                  [--eval_step_interval EVAL_STEP_INTERVAL]
                  [--train_batch_size TRAIN_BATCH_SIZE]
                  [--test_batch_size TEST_BATCH_SIZE]
                  [--validation_batch_size VALIDATION_BATCH_SIZE]
                  [--print_misclassified_test_images] [--model_dir MODEL_DIR]
                  [--bottleneck_dir BOTTLENECK_DIR]
                  [--final_tensor_name FINAL_TENSOR_NAME] [--flip_left_right]
                  [--random_crop RANDOM_CROP] [--random_scale RANDOM_SCALE]
                  [--random_brightness RANDOM_BRIGHTNESS]
                  [--architecture ARCHITECTURE]
optional arguments:
  -h, --help            show this help message and exit
  --image_dir IMAGE_DIR
                        Path to folders of labeled images.
...
  --learning_rate LEARNING_RATE
                        How large a learning rate to use when training.
...
```

以`--learning_rate`参数为例，该参数可以控制对最后一层训练时的更新量级。默认值是0.01。频率值越小（如0.005），模型训练学习的时间就会越长，但是结果也会越好，有助于提高整体准确率。相对越高的值（如1.0），训练学习的时间就会越短，但结果就会很差。

所以多次调整参数，对比模型测试数据将帮助我们获得比较好的分类模型。

> 借助TensorBoard的可视化展示，可以更有效地帮助我们完成参数调优的过程
>
> `--summaries_dir` 参数选项用来控制tensorboard中的显示名称
>
> 上文我们使用的是`--summaries_dir=training_summaries/basic`
>
> TensorBoard 监控的是 `training_summaries` 文件夹下的内容，因此任何子目录的存在都不会影响TensorBoard的正常工作。
>
> 参数调整实例：
>
> `--summaries_dir=training_summaries/LR_0.5`
> `--learning_rate=0.5`

## 6.改变样本训练集

> 换一个你真正关心的样本集，获得一个能给你产生实际价值的分类器

![](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/img/9444bbae4d5d9ab1.png)

参考上文中提到的样本数据集，只需要将你自己的图片按照不同类别，保存在以类别命名的文件加下即可。

然后在运行训练脚本时可以参考如下命令：

```python
python retrain.py --image_dir=flower_photos
```

通过参数`--image_dir`选择样本集。

## 7.参考资料

1.***迁移学习（Transfer Learning）***：[DeCAF: A Deep Convolutional Activation Feature for Generic Visual Recognition](https://arxiv.org/pdf/1310.1531v1.pdf)
2.[TensorFlow Tutorials--Image Recognition](https://www.tensorflow.org/tutorials/image_recognition)
3.[How to Retrain Inception's Final Layer for New Categories](https://www.tensorflow.org/tutorials/image_retraining)
4.[Going Deeper with Convolutions](http://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf)
