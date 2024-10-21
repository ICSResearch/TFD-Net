import torch
import torch.nn as nn
import torch.nn.functional as F


class DeviationLoss(nn.Module):

    def __init__(self):
        super().__init__()


    def forward(self, y_pred, y_true):
        confidence_margin = 2.
        # confidence_margin = 0.5
        # print(y_pred,y_true)
        # size=5000 is the setting of l in algorithm 1 in the paper


        # contain_one = torch.nonzero(y_true>0)
        # if contain_one.shape[0] > 1:
        #     # contain_one = 0
        #     print(contain_one.shape[0])
        #     for i in range(len(contain_one)):
        #         print(y_pred[contain_one[i][0]])

        # ref = torch.abs(torch.normal(mean=0., std=torch.full([5000], 1.))).cuda()
        ref = torch.normal(mean=0.1, std=torch.full([5000], 2.).cuda())
        dev = (y_pred - torch.mean(ref)) / torch.std(ref)
        # dev = (y_pred - torch.mean(ref)) / torch.std(ref)
        # ????????????????????????????????????????
        inlier_loss = torch.pow(torch.tanh(dev), 8)
        # ??????????
        outlier_loss = torch.pow(torch.tanh((confidence_margin - dev)), 2)
        # outlier_loss = torch.abs(torch.tanh((confidence_margin - dev)))
        # return torch.mean((1-alpha)*(1 - y_true) * inlier_loss ** self.gamma + alpha*y_true * outlier_loss ** self.gamma)
        # return torch.mean(y_pred ** self.gamma * (1 - y_true) * inlier_loss + (1-y_pred) ** self.gamma * y_true * outlier_loss)
        loss = (1 - y_true) * inlier_loss + y_true * outlier_loss
        # loss_normal = F.normalize(loss,p=1,dim=1)
        return torch.mean(loss)


# 这个损失函数可以用于二分类问题的深度异常检测。其基本思路是通过标准化模型的输出，
# 将正常样本的得分分布映射到一个均值为0，方差为1的标准正态分布上，再利用一个置信度阈值对异常样本进行分类。
# 在损失函数的计算中，使用了正常样本得分的均值和标准差来标准化模型输出，
# 然后使用两个不同的损失项来惩罚正常样本和异常样本，以便在分类中更好地区分它们。
# 具体而言，对于一个输入样本，将其输出值进行标准化，得到样本的偏差值（dev），
# 然后使用tanh函数将其映射到(-1,1)之间。对于正常样本，使用偏差值的平方乘以一个较大的gamma作为损失项；
# 对于异常样本，使用置信度阈值（confidence_margin）和偏差值之间的差的平方作为损失项。
# 最终的损失函数是两个损失项的加权和，权重是一个二元变量y_true，表示输入样本是否为异常样本。
# 这个损失函数的优点是简单易懂，易于实现，并且对于大多数深度异常检测问题具有很好的效果。
# 但是，对于一些特定的异常检测问题，可能需要根据实际情况进行调整。

class TransDLossBinary(nn.Module):

    def __init__(self, gamma=2 ,c_margin=1.4):
        super().__init__()
        self.gamma = gamma
        self.c_margin = c_margin

    def forward(self, y_pred, y_true):
        confidence_margin = self.c_margin
        gamma = self.gamma
        # 离散正态分布中抽取随机数

        # 查看异常数据评分
        # contain_one = torch.nonzero(y_true>0)
        # if contain_one.shape[0] > 1:
        #     # contain_one = 0
        #     print(contain_one.shape[0])
        #     for i in range(len(contain_one)):
        #         print(y_pred[contain_one[i][0]])

        # 较宽的分布能够更好地适应数据的变化和噪声，因此模型的训练效果可能会更好。
        ref = torch.normal(mean=0.1, std=torch.full([5000], 2.).cuda())
        # std_ref = torch.std(ref)
        # mean_ref = torch.mean(ref)
        dev = (y_pred - torch.mean(ref)) / torch.std(ref)
        # 默认正常点分布都在定义的混合高斯区间之中
        inlier_loss = torch.pow(torch.tanh(dev), 2*gamma)
        # 离群点损失
        outlier_loss = torch.pow(torch.tanh(confidence_margin - dev), 2)
        return torch.mean((1 - y_true) * inlier_loss + y_true * outlier_loss)

class DeviationEmbedF1Loss(nn.Module):
    def __init__(self,gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, y_pred, y_true):
        confidence_margin = 1.2
        # print(y_pred,y_true)
        # size=5000 is the setting of l in algorithm 1 in the paper
        # 离散正态分布中抽取随机数

        # 查看异常数据评分
        # contain_one = torch.nonzero(y_true>0)
        # if contain_one.shape[0] > 1:
        #     # contain_one = 0
        #     print(contain_one.shape[0])
        #     for i in range(len(contain_one)):
        #         print(y_pred[contain_one[i][0]])

        # 标准化
        # ref = torch.abs(torch.normal(mean=0.1, std=torch.full([5000], 0.7))).cuda()
        ref = torch.normal(mean=0.1, std=torch.full([5000], 1.).cuda())
        dev = (y_pred - torch.mean(ref)) / torch.std(ref)
        # dev = (y_pred - torch.mean(ref)) / torch.std(ref)
        # 默认正常点分布都在定义的混合高斯区间之中
        inlier_loss = torch.pow(torch.tanh(dev), self.gamma*2)
        # 离群点损失
        outlier_loss = torch.pow(torch.tanh((confidence_margin - dev)), 2)
        loss = (1 - y_true) * inlier_loss + y_true * outlier_loss
        # loss_normal = F.normalize(loss,p=1,dim=1)

        f1_score = self.f1_score_loss(y_true, y_pred)

        # return torch.mean(loss)

        return torch.mean(0.5 * loss+0.5 * f1_score)
    def f1_score_loss(self, y_true, y_pred):
        alpha = 0.75
        pt = y_pred
        loss = - alpha * (1 - pt) ** self.gamma * y_true * torch.log(pt) - \
               (1 - alpha) * pt ** self.gamma * (1 - y_true) * torch.log(1 - pt)
        # if self.reduction == 'elementwise_mean':
        loss = torch.mean(loss)
        # elif self.reduction == 'sum':
        #     loss = torch.sum(loss)
        return loss


class BCEFocalLoss(torch.nn.Module):
    """
    二分类的Focalloss alpha 固定
    """
    def __init__(self, gamma=2, alpha=0.75, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, _input, target):
        # pt = torch.sigmoid(_input)
        pt = _input
        alpha = self.alpha
        loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
               (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

class BCEFocalLossFew(torch.nn.Module):
    """
    二分类的Focalloss alpha 固定
    """
    def __init__(self, gamma=4, alpha=0.8, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, _input, target):
        # pt = torch.sigmoid(_input)
        pt = _input
        alpha = self.alpha
        # loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
        #        (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        loss = - alpha * (1 - pt) ** 1 * target * torch.log(pt) - \
               (1 - alpha) * pt ** 8 * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduce=True):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss



# 构造损失
def build_criterion(criterion, gamma=2,c_margin=1.4):
    if criterion == "deviation":
        # 自定义偏差损失
        print("Loss : Deviation")
        return DeviationLoss()
    elif criterion == "TransDLossBinary":
        # 自定义偏差损失
        print("Loss : Transformer deviation Binary")
        return TransDLossBinary(gamma,c_margin)
    elif criterion == "BCE":
        print("Loss : Binary Cross Entropy")
        return torch.nn.BCEWithLogitsLoss()
    elif criterion == "BCEfocal":
        print("Loss : BCEfocal")
        return BCEFocalLoss()
    elif criterion == "deviEmbedfocal":
        # 偏差损失+focal loss
        print("Loss : deviation Embedding focal loss")
        return DeviationEmbedF1Loss(gamma)
    elif criterion == "BCEFew":
        print("Loss : Binary Cross Entropy Few")
        return BCEFocalLossFew()
    else:
        raise NotImplementedError