---
title: TensorFlow学习笔记(1)
date: 2017-07-16 17:58:47
tags: [TensorFlow, 深度学习]
---

![](https://www.wired.com/wp-content/uploads/2015/11/Google-TensorFlow-AI-F.jpg)

# TensorFlow学习笔记

## 1.关于深度学习

确定需要了解的一些基本概念：

- 有监督学习，无监督学习，分类，聚类，***回归***
- [神经元模型](http://www.ruanyifeng.com/blog/2017/07/neural-network.html)，多层感知器，***BP算法***
- 目标函数（损失函数），***激活函数***，***梯度下降法***
- ***全连接网络***、***卷积神经网络***、***递归神经网络***
- 训练集，测试集，交叉验证，欠拟合，过拟合
- 数据规范化

## 2.深度学习主流框架

- TensorFlow: ***商业*** ***MacOS or Linux*** ***Google Brain***
  ***Google research and production needs***
  ***前任：DistBelif***
- PyTorch: ***商业*** ***Facebook*** ***Lua-based Torch Framework*** ***Pythonic***
- Theano: ***学术*** ***Any OS***  ***Python库*** ***与NumPy紧密集成***
- Keras: ***基于Theano和TensorFlow的深度学习库***

[针对Google的TenserFlow和Facebook的PyTorch的比较](https://medium.com/@dubovikov.kirill/pytorch-vs-tensorflow-spotting-the-difference-25c75777377b)：

***TensorFlow***:

- 更适合生产环境的模型开发


- 开发需要在手机平台上部署的模型
- 丰富的社区支持和更为全面的文档说明
- 丰富的学习资源, 如[MOOC](https://www.udacity.com/course/deep-learning--ud730)
- 可视化工具Tensorboard
- 大规模分布式的模型训练

***PyTouch***

- 相对更友好的开发和调试环境
- *Love all things Pythonic*

[对比深度学习十大框架：TensorFlow最流行但并不是最好](https://zhuanlan.zhihu.com/p/24687814)

## 3.TensorFlow环境搭建

**操作系统**：MacOS

**依赖工具**：Virtualenv，Python3.6

**参考链接**：

- [Installing TensorFlow on Mac OS X](https://www.tensorflow.org/install/install_mac)
- [配合Vagrant在Mac上搭建一个干净的TensorFlow环境](https://juejin.im/post/58a85f7975c4cd340fa497bd)

***TensorFlow在Mac上编译安装方法*** [参考链接](https://www.tensorflow.org/install/install_sources)

虽然Mac下使用 ***pip install tensorflow*** 就可以安装tensorflow，但没有CPU和GPU加速，而且会出现一堆如下警告:

```python
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE3 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
```

所以决定自己编译源码安装（记住一句话：”没有强迫症的程序员不是好运维“），无GPU加速。

1. clone源码

   ***git clone https://github.com/tensorflow/tensorflow***

2. 安装依赖

   - bazel ***brew install bazel***
   - TensorFlow Python依赖: ***six*** ***numpy*** ***wheel***

3. 配置安装

   按下面命令配置源码，我的mac pro没有NVIDA的显卡，不需要配置GPU加速，所以一路回车就可以。配置后会自动下载依赖的gz，由于大家都懂的网络原因，会出现各种Timeout，不要怕，上VPN就能解决问题，这个下载过程很慢，耐心等待。

   ***./configure***

4. 编译源码

   这是关键步骤，因为mac没有GPU，只能优化CPU，采用-march=native参数会根据本机CPU特性进行编译（mac pro支持SSE4.2）。这个过程也会下载各种依赖，编译过程大概需要1-2小时，耐心等待。

   ***bazel build --config=opt --copt=-march=native //tensorflow/tools/pip_package:build_pip_package***

5. 生成whl包并安装

   ***bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg***
   ***pip install /tmp/tensorflow_pkg/tensorflow-1.2.1-py2-none-any.whl***

6. TensorFlow测试

   ```python
   # Python
   import tensorflow as tf
   hello = tf.constant('Hello, TensorFlow!')
   sess = tf.Session()
   print(sess.run(hello))
   ```
