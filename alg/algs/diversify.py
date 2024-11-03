# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cdist

from alg.modelopera import get_fea
from network import Adver_network, common_network
from alg.algs.base import Algorithm
from loss.common_loss import Entropylogits


class Diversify(Algorithm):  # Diversify 类是一个算法类，继承自 Algorithm 基类，具体实现了一个用于多域或多任务学习的深度学习算法。它通过构建多个特征提取器和分类器，以及对抗性网络，来实现对复杂数据分布的学习和泛化。以下是对 Diversify 类及其各个方法的详细解释：

    def __init__(self, args):  # 初始化方法：定义 __init__ 方法，初始化 Diversify 类的实例。  args：包含配置参数的对象，用于设置算法中的各种组件和超参数。

        super(Diversify, self).__init__(args)  # 调用父类初始化：通过 super() 调用父类 Algorithm 的初始化方法，确保继承的属性和方法被正确初始化。
        self.featurizer = get_fea(args)  #  初始化各类网络组件 作用：初始化特征提取器 featurizer。原因：特征提取器用于将输入数据映射到特征空间，提取有用的特征用于后续分类或对抗任务。
        self.dbottleneck = common_network.feat_bottleneck(  # 作用：初始化 dbottleneck 层，用于对特征进行瓶颈处理，减少维度或改变特征空间。 self.featurizer.in_features：特征提取器的输出维度。
            self.featurizer.in_features, args.bottleneck, args.layer)  # args.bottleneck：瓶颈层的输出维度。 # args.layer：瓶颈层的类型或结构配置。 # 原因：瓶颈层用于压缩或转换特征，减少特征冗余，提高训练效率。
        self.ddiscriminator = Adver_network.Discriminator(  # 作用：初始化 ddiscriminator 对抗性判别器，用于对瓶颈特征进行分类或区分不同域。
            args.bottleneck, args.dis_hidden, args.num_classes)  # args.bottleneck：瓶颈层输出的特征维度。 args.dis_hidden：判别器隐藏层的大小。 args.num_classes：分类的类别数量。  # 原因：对抗性判别器用于通过对抗训练，使得特征具有域不变性。

        self.bottleneck = common_network.feat_bottleneck(  # 作用：初始化主瓶颈层 bottleneck 和分类器 classifier。
            self.featurizer.in_features, args.bottleneck, args.layer)
        self.classifier = common_network.feat_classifier(  # 原因：这些组件用于将特征映射到类别概率空间，完成分类任务。
            args.num_classes, args.bottleneck, args.classifier)

        self.abottleneck = common_network.feat_bottleneck(  # 作用：初始化另一个瓶颈层 abottleneck 和分类器 aclassifier，以及域分类器 dclassifier。
            self.featurizer.in_features, args.bottleneck, args.layer)

        self.aclassifier = common_network.feat_classifier(  # 原因：这些组件用于处理额外的任务或域相关的特征，支持多任务学习或多域区分。
            args.num_classes*args.latent_domain_num, args.bottleneck, args.classifier)
        self.dclassifier = common_network.feat_classifier(
            args.latent_domain_num, args.bottleneck, args.classifier)
        self.discriminator = Adver_network.Discriminator(  # 作用：初始化对抗性判别器 discriminator，用于区分不同的潜在域。
            args.bottleneck, args.dis_hidden, args.latent_domain_num)  # 原因：通过对抗训练，使特征具有域不变性，提升模型的泛化能力。
        self.args = args  # 保存参数：将 args 参数对象保存为类属性，便于在类的其他方法中使用。

    def update_d(self, minibatch, opt):  # 方法 update_d：更新对抗性判别器
        all_x1 = minibatch[0].cuda().float()
        all_d1 = minibatch[1].cuda().long()
        all_c1 = minibatch[4].cuda().long()
        z1 = self.dbottleneck(self.featurizer(all_x1))  # 作用：处理一个小批次的数据，提取特征并通过瓶颈层进行转换。 原因：通过对抗性训练，优化特征使其具有域不变性。
        disc_in1 = Adver_network.ReverseLayerF.apply(z1, self.args.alpha1)  # 对抗训练：ReverseLayerF.apply：使用反向传播层翻转梯度，促使对抗性训练。
        disc_out1 = self.ddiscriminator(disc_in1)  # disc_loss：计算对抗损失，鼓励特征分布不变性。
        disc_loss = F.cross_entropy(disc_out1, all_d1, reduction='mean')  # # disc_loss：计算对抗损失，鼓励特征分布不变性。
        cd1 = self.dclassifier(z1)   # 分类与优化：结合分类器输出和对抗损失，计算总损失并更新参数。
        ent_loss = Entropylogits(cd1)*self.args.lam + \
            F.cross_entropy(cd1, all_c1)  # 调用一个自定义的损失函数 Entropylogits，计算输出 cd1 的熵损失。
        loss = ent_loss+disc_loss  # 分类器loss和区域辨别器损失
        opt.zero_grad()
        loss.backward()
        opt.step()  # 分类与优化：结合分类器输出和对抗损失，计算总损失并更新参数。
        return {'total': loss.item(), 'dis': disc_loss.item(), 'ent': ent_loss.item()}  # 返回损失：返回各部分损失的值，用于监控训练过程。

    def set_dlabel(self, loader):  # 方法 set_dlabel：设置域标签
        self.dbottleneck.eval()  #  dbottleneck（瓶颈层）、dclassifier（域分类器）和 featurizer（特征提取器）
        self.dclassifier.eval()  # 在评估模式下，模型的参数保持不变，不会进行梯度更新。同时，BatchNorm 和 Dropout 等层在 eval 模式下行为不同，更适合进行推断和评估。
        self.featurizer.eval()  # 作用：设置模型为评估模式，不更新参数。 原因：在评估模式下，确保模型不执行梯度更新。

        start_test = True  # 评估阶段：不进行梯度计算，逐批次提取特征并预测。  原因：计算并收集特征，用于后续的域标签设置。
        with torch.no_grad():  # 作用: 禁用梯度计算（torch.no_grad()），防止计算图的构建和内存的消耗；初始化数据加载器 iter_test 以便逐批次提取数据。  ## 原因: 禁用梯度计算是因为在评估模式下不需要计算梯度，同时节省内存和加快推断速度。
            iter_test = iter(loader)  #
            for _ in range(len(loader)):
                data = next(iter_test)
                inputs = data[0]
                inputs = inputs.cuda().float()
                index = data[-1]
                feas = self.dbottleneck(self.featurizer(inputs))  # # 作用: 使用 featurizer 提取输入 inputs 的特征，然后通过 dbottleneck 进行特征降维或变换，最后通过 dclassifier 预测域标签。  ## 原因: 通过提取特征并进行分类，可以获得数据的特征表示和初步的域标签分布信息。
                outputs = self.dclassifier(feas)  # # 评估阶段：不进行梯度计算，逐批次提取特征并预测。  原因：计算并收集特征，用于后续的域标签设置。
                if start_test:  # 合并特征和输出：将所有批次的特征和输出合并，形成完整的数据集特征和预测结果。 原因：为了后续的距离计算和标签重新分配，需要将所有数据合并处理。  ### 5. 累积特征和输出
                    all_fea = feas.float().cpu()
                    all_output = outputs.float().cpu()
                    all_index = index
                    start_test = False
                else:
                    all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                    all_output = torch.cat(
                        (all_output, outputs.float().cpu()), 0)
                    all_index = np.hstack((all_index, index))  # 合并特征和输出：将所有批次的特征和输出合并，形成完整的数据集特征和预测结果。原因：为了后续的距离计算和标签重新分配，需要将所有数据合并处理。
        all_output = nn.Softmax(dim=1)(all_output)  # 归一化特征：将特征标准化为单位长度，便于后续计算。 原因：归一化有助于减少不同特征尺度的影响，提升聚类效果。

        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
        all_fea = all_fea.float().cpu().numpy()  # # 归一化特征：将特征标准化为单位长度，便于后续计算。 原因：归一化有助于减少不同特征尺度的影响，提升聚类效果。

        K = all_output.size(1)  # 初步聚类：通过余弦距离计算数据到初始聚类中心的距离，并分配初始标签。 原因：通过聚类为每个样本分配潜在的域标签。
        aff = all_output.float().cpu().numpy()
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = cdist(all_fea, initc, 'cosine')
        pred_label = dd.argmin(axis=1)    # 初步聚类：通过余弦距离计算数据到初始聚类中心的距离，并分配初始标签。 原因：通过聚类为每个样本分配潜在的域标签。

        for _ in range(1):  # 多次迭代聚类：对聚类结果进行迭代调整，使得聚类中心更加精确。 原因：多次迭代能改善聚类效果，使标签分配更加稳定。
            aff = np.eye(K)[pred_label]
            initc = aff.transpose().dot(all_fea)
            initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
            dd = cdist(all_fea, initc, 'cosine')
            pred_label = dd.argmin(axis=1)  #   # 多次迭代聚类：对聚类结果进行迭代调整，使得聚类中心更加精确。 原因：多次迭代能改善聚类效果，使标签分配更加稳定。  # 多次迭代聚类：对聚类结果进行迭代调整，使得聚类中心更加精确。 原因：多次迭代能改善聚类效果，使标签分配更加稳定。

        loader.dataset.set_labels_by_index(pred_label, all_index, 'pdlabel')  # 设置新标签：将聚类得到的新标签分配回数据集，并恢复模型为训练模式。 原因：将域标签更新到数据集中，以便后续训练使用。
        print(Counter(pred_label))
        self.dbottleneck.train()
        self.dclassifier.train()
        self.featurizer.train()  # 设置新标签：将聚类得到的新标签分配回数据集，并恢复模型为训练模式。   原因：将域标签更新到数据集中，以便后续训练使用。

    def update(self, data, opt):  # 方法 update：更新模型
        all_x = data[0].cuda().float()  # 作用：处理一个小批次的数据，通过特征提取、瓶颈转换和分类进行训练，更新模型参数。 # 原因：通过联合分类和对抗损失优化，使模型具备分类能力并且具有域不变性。
        all_y = data[1].cuda().long()
        all_z = self.bottleneck(self.featurizer(all_x))

        disc_input = all_z
        disc_input = Adver_network.ReverseLayerF.apply(
            disc_input, self.args.alpha)
        disc_out = self.discriminator(disc_input)
        disc_labels = data[4].cuda().long()

        disc_loss = F.cross_entropy(disc_out, disc_labels)
        all_preds = self.classifier(all_z)
        classifier_loss = F.cross_entropy(all_preds, all_y)
        loss = classifier_loss+disc_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        return {'total': loss.item(), 'class': classifier_loss.item(), 'dis': disc_loss.item()}  # # 作用：处理一个小批次的数据，通过特征提取、瓶颈转换和分类进行训练，更新模型参数。 # 原因：通过联合分类和对抗损失优化，使模型具备分类能力并且具有域不变性。

    def update_a(self, minibatches, opt):  # 方法 update_a：更新对抗分类器
        all_x = minibatches[0].cuda().float()  # 作用：处理一个小批次的数据，通过特征提取和对抗分类器进行训练，更新模型参数。 原因：通过对抗分类器使特征对任务和域具有区分能力。  ## 作用：将输入数据 minibatches[0] 移动到 GPU，并将数据类型转换为浮点数。  ## 原因：在 GPU 上进行计算以加速训练过程，同时确保数据类型兼容后续的神经网络运算。
        all_c = minibatches[1].cuda().long()  ## 作用：将类别标签 minibatches[1] 移动到 GPU，并转换为长整型（整数类型）。  ## 原因：标签需要在 GPU 上进行交叉熵损失计算，因此必须是长整型。
        all_d = minibatches[4].cuda().long()  ## 作用：将域标签 minibatches[4] 移动到 GPU，并转换为长整型。 ## 原因：域标签用于对抗性学习中，以区分不同的域。
        all_y = all_d*self.args.num_classes+all_c  ## 作用：计算组合标签 all_y，将类别标签 all_c 和域标签 all_d 结合，得到一个唯一的标签索引。  ## 原因：通过这种方式，每个域-类别组合都有一个独特的标签，便于对抗分类器进行多任务学习，使其能够同时学习域和类别之间的区分。
        all_z = self.abottleneck(self.featurizer(all_x))  ## 作用：对输入数据进行特征提取，先通过 self.featurizer 提取初级特征，再通过 self.abottleneck 进行特征变换或降维。  ## 原因：将输入数据转换为抽象的特征表示，为后续的对抗分类提供更有意义的输入。
        all_preds = self.aclassifier(all_z)  ## 作用：将提取的特征 all_z 通过 aclassifier 学习任务和域的联合分布，使特征具有对类别和域的细粒度区分能力。”
        classifier_loss = F.cross_entropy(all_preds, all_y)  ## 作用：计算分类器的交叉熵损失 classifier_loss，将预测结果 all_preds 与组合标签 all_y 进行对比。 ## 原因：交叉熵损失用于衡量预测结果与真实标签之间的差异，优化该损失可以提升对抗分类器的分类性能。
        loss = classifier_loss  # 作用：将分类损失赋值给 loss 变量，用于后续的反向传播和优化。 ## 原因：简化代码逻辑，如果需要增加其他损失项，可以在这里进行叠加。
        opt.zero_grad()  # 作用：将优化器中的梯度缓存清零，以防止累积之前的梯度。 ## 原因：每次进行反向传播前都需要清除旧的梯度，否则梯度会在每次 backward 时累加，导致参数更新错误。
        loss.backward()  # 作用：计算损失的梯度，通过反向传播将梯度传递到模型的所有可学习参数上。 ## 原因：更新参数需要知道损失相对于各个参数的梯度，因此必须进行反向传播。
        opt.step()  ## 作用：使用优化器 opt 通过计算的梯度更新模型参数。  ## 原因：根据计算得到的梯度调整模型参数，使得损失最小化，从而提高分类器的性能。
        return {'class': classifier_loss.item()}  #  # 作用：处理一个小批次的数据，通过特征提取和对抗分类器进行训练，更新模型参数。 原因：通过对抗分类器使特征对任务和域具有区分能力。  # 作用：返回当前批次的分类损失 classifier_loss，并将其转换为 Python 标量。  ## 原因：返回损失值便于在训练过程中打印和监控，确保对抗分类器的学习进展。
        # 总结
        # update_a 方法：用于训练对抗分类器，通过特征提取和对抗分类器的联合学习，使得模型能同时学习任务和域的区分。它通过组合标签的方式有效地对抗不同域的数据，使特征对任务和域都具有区分能力。反向传播和优化步骤更新了分类器的参数，使其逐步优化，减少预测与真实标签之间的差异。
    def predict(self, x):  # 方法 predict 和 predict1：预测
        return self.classifier(self.bottleneck(self.featurizer(x)))  # 作用：通过特征提取、瓶颈转换和分类器进行预测。 ## 原因：输出分类结果，用于评估模型的性能。

    def predict1(self, x):
        return self.ddiscriminator(self.dbottleneck(self.featurizer(x)))  # 作用：通过特征提取、瓶颈转换和对抗判别器进行预测。  原因：用于域相关的预测任务。


#  Diversify 类通过多种特征提取、瓶颈处理、分类和对抗性判别器，结合对抗训练和聚类优化，实现了对多域、多任务数据的处理和学习。
#  它通过多次更新和优化过程，使得模型既具备分类能力，又能处理不同域之间的分布差异，提高了模型的泛化能力。
#