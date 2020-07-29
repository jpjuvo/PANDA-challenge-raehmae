import fastai
from fastai.vision import *
import torch
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score

def get_default_metrics(model_name, data=None, is_ordinal=False):
    """ Returns a list of metrics and the name of the metric to monitor when saving the best model. """
    # multitask
    if model_name in ["multihead_tilecat", "multihead_tilecat_attention"]:
        metricsList = mt_metrics_generator(data.mt_lengths,
                                           override_metrics=[partial(cohen_kappa_score,weights='quadratic', labels=[0,1,2,3,4,5])],
                                           override_names=['kappa_score'],
                                           is_numpy_metric_list=[True],
                                           is_argmax_list = [True])
        return metricsList, 'kappa_score'
    
    # regression models
    elif model_name in ["iafoss_regr", "conv_branch_regr"]:
        return [KappaOptimizer()], 'quad_kappa'
    
    else: # regular
        if is_ordinal:
            return [OrdinalKappaScore(weights="quadratic")], 'ordinal_kappa_score'
        return [KappaScore(weights='quadratic')], 'kappa_score'


class MultitaskAverageMetric(AverageMetric):
    def __init__(self, func, name=None):
        super().__init__(func)
        self.name = name # subclass uses this attribute in the __repr__ method.

def _mt_parametrable_metric(inputs, *targets, func, i=0, is_numpy=False, is_argmax=True):
    input = inputs[i]
    target = targets[i]
    if is_argmax:
        input = input.argmax(1)
    if is_numpy:
        input = input.cpu().numpy()
        target = target.cpu().numpy()
    return torch.tensor(func(input, target))

def _format_metric_name(field_name, metric_func):
    return f"{field_name}"

def mt_metrics_generator(mt_lengths, override_metrics = [], override_names = [], is_numpy_metric_list = [], is_argmax_list=[]):
    metrics = []
    for i, _ in enumerate(mt_lengths):
        metric_func = override_metrics[i] if len(override_metrics) > i else accuracy
        is_argmax = is_argmax_list[i] if len(is_argmax_list) > i else False
        is_numpy_metric = is_numpy_metric_list[i] if len(is_numpy_metric_list) > i else False
        name = override_names[i] if len(override_names) > i else f'acc_{i}'
        if metric_func:
            partial_metric = partial(_mt_parametrable_metric, i=i, func=metric_func, is_numpy=is_numpy_metric, is_argmax=is_argmax)
            metrics.append(MultitaskAverageMetric(partial_metric, _format_metric_name(name,metric_func)))
    return metrics


class KappaOptimizer(Callback):
    ''' Calculate the quadratic weighted kappa score on epochs end for 
        the validation data as a whole (i.e. not as an average of batches) '''
        
    def __init__(self):
        super().__init__()
        self.coef = [0.5, 1.5, 2.5, 3.5, 4.5]
        self.name = 'quad_kappa'
        
    def on_epoch_begin(self, **kwargs):
        self.targs, self.preds = torch.FloatTensor([]), torch.FloatTensor([])
        
    def on_batch_end(self, last_output:torch.FloatTensor, last_target:torch.FloatTensor, **kwargs):
        last_output = self.predict(last_output)
        self.preds = torch.cat((self.preds, last_output.cpu()))
        self.targs = torch.cat((self.targs, last_target.cpu()))
    
    def on_epoch_end(self, last_metrics, **kwargs):
        return add_metrics(last_metrics, self.quad_kappa(self.preds, self.targs))
    

    def predict(self, preds):
        return self._predict(self.coef, preds)

    @classmethod
    def _predict(cls, coef, preds):
        if type(preds).__name__ == 'Tensor':
            y_hat = preds.clone().view(-1)
        else:
            y_hat = torch.FloatTensor(preds).view(-1)

        for i,pred in enumerate(y_hat):
            if   pred < coef[0]: y_hat[i] = 0
            elif pred < coef[1]: y_hat[i] = 1
            elif pred < coef[2]: y_hat[i] = 2
            elif pred < coef[3]: y_hat[i] = 3
            elif pred < coef[4]: y_hat[i] = 4
            else:                y_hat[i] = 5
        return y_hat.float()
    
    
    def quad_kappa(self, preds, y):
        return self._quad_kappa(self.coef, preds, y)

    
    @classmethod
    def _quad_kappa(cls, coef, preds, y, model_n_out=6):
        y_hat = cls._predict(coef, preds)
        
        def __checkNaN(val):
            return 1.0 if np.isnan(val) else val
        try:
            return __checkNaN(cohen_kappa_score(y, y_hat, 
                                                labels = list(range(model_n_out)), 
                                                weights='quadratic'))
        except:
            return __checkNaN(cohen_kappa_score(y.cpu(), y_hat.cpu(), 
                                                labels = list(range(model_n_out)), 
                                                weights='quadratic'))
        
    def fit(self, preds, y):
        ''' maximize quad_kappa '''
        neg_kappa = lambda coef: -self._quad_kappa(coef, preds, y)
        opt_res = sp.optimize.minimize(neg_kappa, x0=self.coef, method='nelder-mead',
                                       options={'maxiter':100, 'fatol':1e-20, 'xatol':1e-20})
        self.coef = opt_res.x

        
    def forward(self, preds, y):
        ''' the pytorch loss function '''
        return torch.tensor(self.quad_kappa(preds, y))


class ConfusionMatrix(fastai.callback.Callback):
    "Computes the confusion matrix."

    def on_train_begin(self, **kwargs):
        self.n_classes = 0

    def on_epoch_begin(self, **kwargs):
        self.cm = None

    def _get_preds(self, arr):
        #mask = arr == 0
        #return np.clip(np.where(mask.any(1), mask.argmax(1), 6) - 1, 0, 5)
        return np.sum(arr, 1)

    def on_batch_end(self, last_output:Tensor, last_target:Tensor, **kwargs):
        preds = torch.tensor(self._get_preds((torch.sigmoid(last_output) > 0.5).cpu().numpy()))
        
        targs = torch.tensor(self._get_preds(last_target.cpu().numpy()))

        if self.n_classes == 0:
            self.n_classes = last_output.shape[-1]
            self.x = torch.arange(0, self.n_classes)
        
        cm = ((preds==self.x[:, None]) & (targs==self.x[:, None, None])).sum(dim=2, dtype=torch.float32)
        if self.cm is None: self.cm =  cm
        else:               self.cm += cm

    def on_epoch_end(self, **kwargs):
        self.metric = self.cm
        

@dataclass
class OrdinalKappaScore(ConfusionMatrix):
    "Compute the rate of agreement (Cohens Kappa)."
    weights:Optional[str]=None      # None, `linear`, or `quadratic`

    def on_epoch_end(self, last_metrics, **kwargs):
        sum0 = self.cm.sum(dim=0)
        sum1 = self.cm.sum(dim=1)
        expected = torch.einsum('i,j->ij', (sum0, sum1)) / sum0.sum()
        if self.weights is None:
            w = torch.ones((self.n_classes, self.n_classes))
            w[self.x, self.x] = 0
        elif self.weights == "linear" or self.weights == "quadratic":
            w = torch.zeros((self.n_classes, self.n_classes))
            w += torch.arange(self.n_classes, dtype=torch.float)
            w = torch.abs(w - torch.t(w)) if self.weights == "linear" else (w - torch.t(w)) ** 2
        else: raise ValueError('Unknown weights. Expected None, "linear", or "quadratic".')
        k = torch.sum(w * self.cm) / torch.sum(w * expected)
        return add_metrics(last_metrics, 1-k)
    