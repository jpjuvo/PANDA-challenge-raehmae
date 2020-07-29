import fastai
from fastai.vision import *
import torch
import torch.nn as nn
from torch.autograd import Variable

def get_default_loss(model_name, data=None, is_ordinal=False):
    if model_name in ["multihead_tilecat", "multihead_tilecat_attention"]:
        n_tasks = len(data.mt_lengths)
        return MultiTaskLoss(data, 
                             loss_weights=[1.] + [(1./n_tasks) for _ in range(n_tasks - 1)] # weight the first task as much as the all the rest summed
                             )
    elif model_name in ["iafoss_regr", "conv_branch_regr"]:
        return MSELossFlatSmooth()
    else:
        if is_ordinal:
            return nn.BCEWithLogitsLoss()
        return nn.CrossEntropyLoss()

class MultiTaskLoss(nn.Module):
    def __init__(self, data, override_losses = [], loss_weights = []):
        super().__init__()
        self.mt_lengths, self.mt_types = data.mt_lengths, data.mt_types
        self.override_losses = override_losses
        self.loss_weights = loss_weights
        
    def forward(self, inputs, *targets, **kwargs):
        loss_size = targets[0].shape[0] if kwargs.get('reduction') == 'none' else 1
        losses = torch.zeros([loss_size]).cuda()
        
        for i in range(len(inputs)):
            input = inputs[i]
            target = targets[i]

            loss_weight = self.loss_weights[i] if len(self.loss_weights) > i else 1.0
            
            if len(self.override_losses) > i:
                losses += loss_weight*self.override_losses[i](**kwargs)(input, target).cuda()
            else:
                if self.mt_types[i] == CategoryList:
                    if i==0: #<- CCE for the isup categorization
                        losses += loss_weight*CrossEntropyFlat(**kwargs)(input, target).cuda()
                    else: #<- focal loss for cancer categorization
                        losses += loss_weight*FocalLoss(gamma=2, alpha=0.25, **kwargs)(input, target).cuda()
                elif issubclass(self.mt_types[i], FloatList):
                    losses += loss_weight*MSELossFlat(**kwargs)(input, target).cuda()
        
        return losses.sum()


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)
            
        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()



def gaussian(ins, mean=0, stddev=0.2):
    noise = Variable(ins.data.new(ins.size()).normal_(mean, stddev)).cuda()
    return ins + noise

class MSELossFlatSmooth(nn.MSELoss): 
    def forward(self, input:Tensor, target:Tensor) -> Rank0Tensor:
        return super().forward(input.view(-1), gaussian(target.float()).view(-1))