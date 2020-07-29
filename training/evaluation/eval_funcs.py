from model.model_config import ModelConfig
from model.model_utils import load_models_from_dir
from train.metrics_broker import *
from data.sampler import FoldSampler
import matplotlib.pyplot as plt
import os
import fastai
from fastai.vision import *
from fastai.callbacks import *
from sklearn.metrics import cohen_kappa_score
from train.radam import *
from sklearn.metrics import confusion_matrix
import cv2


# inspired by https://www.kaggle.com/tanlikesmath/intro-aptos-diabetic-retinopathy-eda-starter
from sklearn.metrics import cohen_kappa_score
class KappaOptimizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.coef = [0.5, 1.5, 2.5, 3.5, 4.5]
        # define score function:
        self.func = self.quad_kappa
    
    
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
        return y_hat.int()
    
    
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

kappa_score = KappaOptimizer()

def regrPreds2cat(preds, classes=[0,1,2,3,4,5]):
    coef = [(classes[i] + classes[i+1])/2 for i in range(len(classes)) if i < len(classes) - 1]
    if type(preds).__name__ == 'Tensor':
        y_hat = preds.clone().view(-1)
    else:
        y_hat = torch.FloatTensor(preds).view(-1)
    for i,pred in enumerate(y_hat):
        tmp = -1
        for j, c in enumerate(coef):
            if pred < c and tmp < 0:
                tmp = classes[j]
        if tmp < 0:
            tmp = classes[-1]
        y_hat[i] = tmp
    return y_hat.int().numpy()

def ordinalRegs2cat(arr, classes=[0,1,2,3,4,5]):
    #mask = arr == 0
    #return np.clip(np.where(mask.any(1), mask.argmax(1), len(classes)) - 1, classes[0], classes[-1])
    return np.sum(arr,1)

def plot_confusion_matrix_scipy(preds, targets, normalize:bool=False, title:str='Confusion matrix', cmap:Any="Blues", slice_size:int=1,
                              norm_dec:int=2, plot_txt:bool=True, classes=[0,1,2,3,4,5], **kwargs)->Optional[plt.Figure]:
        "Plot the confusion matrix, with `title` and using `cmap`."
        
        # This function is mainly copied from the sklearn docs
        cm = confusion_matrix(targets, preds)#, normalize='true' if normalize else None)
        if normalize: cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fig = plt.figure(**kwargs)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        tick_marks = classes#np.arange(classes)
        plt.xticks(tick_marks, classes, rotation=90)
        plt.yticks(tick_marks, classes, rotation=0)

        if plot_txt:
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                coeff = f'{cm[i, j]:.{norm_dec}f}' if normalize else f'{cm[i, j]}'
                plt.text(j, i, coeff, horizontalalignment="center", verticalalignment="center", color="white" if cm[i, j] > thresh else "black")

        ax = fig.gca()
        ax.set_ylim(len(classes)-.5,-.5)
                           
        plt.tight_layout()
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.grid(False)
        return cm

def plot_samples(data, losses, preds, image_ids, ds_rate=5, k=5):
    f, ax = plt.subplots(2*k+1,1, figsize=(10,(k+1)*2))
    
    def _faiIm2npy(fai_im):
        np_im = fai_im.to_one().data.numpy().swapaxes(0,2).swapaxes(0,1)
        return cv2.resize(np_im, (np_im.shape[1]//ds_rate, np_im.shape[0]//ds_rate))
    
    def _setAx(ax,title):
        ax.set_title(title)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
    
    # top losses
    top_losses, top_loss_indices = losses.topk(k=k,largest=True,sorted=True)
    for i, loss, ind in zip(range(k), top_losses, top_loss_indices):
        fai_im, cl = data.valid_ds.__getitem__(ind)
        pred = preds[ind]
        image_id = image_ids[ind]
        ax[i].imshow(_faiIm2npy(fai_im))
        _setAx(ax[i],"{3}|loss:{0:.2f}|gt_cl:{1}|pred_cl:{2}".format(loss.numpy(),str(cl),int(pred),image_id))
    
    ax[k].axis('off')
    ax[k//2].set_ylabel('Top losses', fontsize=16)
    
    min_losses, min_loss_indices = losses.topk(k=k,largest=False,sorted=True)
    for i, loss, ind in zip(range(k+1,2*k+1), min_losses, min_loss_indices):
        fai_im, cl = data.valid_ds.__getitem__(ind)
        pred = preds[ind]
        image_id = image_ids[ind]
        ax[i].imshow(_faiIm2npy(fai_im))
        _setAx(ax[i],"{3}|loss:{0:.2f}|gt_cl:{1}|pred_cl:{2}".format(loss.numpy(),str(cl),int(pred),image_id))
        
    ax[k + 1 + k//2].set_ylabel('Min losses', fontsize=16)

def evaluate_model_dir(model_dir, sampler=None, TRAIN=None, LABELS=None, **kwargs):
    """
    Evaluates CV models in out-of-fold fashion and saves some stats to the model dir

    Provide either sampler or TRAIN and LABELS.
    model_dir: directory containing models
    sampler (FoldSampler): optional data sampler instance
    TRAIN: optional training images folder
    LABELS: optional train.csv path
    """
    # load config
    config = ModelConfig.fromDir(model_dir)
    # load models
    models = load_models_from_dir(model_dir)
    model_name = config.getField('model_name')
    regr = "regr" in model_name
    
    n_folds = len(models)
    sz = config.getField('sz')
    mean = torch.tensor(np.array(config.getField('mean')).astype(np.float32))
    std = torch.tensor(np.array(config.getField('std')).astype(np.float32))
    N = config.getField('N')
    is_ordinal = config.getMetaField('is_ordinal')

    if sampler is None:
        assert (TRAIN is not None and LABELS is not None), "Either sampler or TRAIN + LABELS must be provided"
        
        sampler = FoldSampler(TRAIN,
                            LABELS,
                            mean,
                            std,
                            N,
                            tfms=[], 
                            sz=sz, 
                            bs=1, 
                            n_folds=n_folds,
                            is_ordinal=is_ordinal,
                            model_name=model_name)
    
    # evaluate out of fold
    val_qwks = []
    karolinska_preds = []
    karolinska_targets = []
    radboud_preds = []
    radboud_targets = []
    all_preds = []
    all_targets = []
    score_dict = {}
    for fold, model in zip(range(n_folds), models):
        data = sampler.get_data(fold)
        default_metrics, monitor_metric = get_default_metrics(model_name, data=data, is_ordinal=is_ordinal)
        learn = Learner(data,
                        model,
                        metrics=default_metrics,
                        opt_func=Over9000
                        ).to_fp16()
        learn.create_opt(1e-3, 0.9)

        # calculate data provider specific scores
        preds, targets, losses = learn.get_preds(with_loss=True)
        targets = targets.numpy()
        if is_ordinal:
            targets = ordinalRegs2cat(targets)
            losses = torch.sum(losses.view(preds.shape[0],preds.shape[1]),axis=1)

        if not regr:
            if is_ordinal:
                preds = ordinalRegs2cat((preds > 0.5).numpy())
            else:
                preds = np.argmax(preds.numpy(), axis=1)
        else:
            # convert to categories
            preds = regrPreds2cat(preds)

        all_preds += list(preds)
        all_targets += list(targets)

        # fold qwk
        val_qwk = cohen_kappa_score(preds, targets, weights="quadratic")
        val_qwks.append(val_qwk)
        score_dict[f'{fold}_qwk'] = str(val_qwk)
        
        # get 'karolinska' 'radboud' labels
        data_providers = [sampler.df[sampler.df.image_id==os.path.basename(_id)].data_provider.values[0] for _id in data.valid_ds.items]
        for pred, target, provider in zip(preds,targets,data_providers):
            if provider == "karolinska":
                karolinska_preds.append(pred)
                karolinska_targets.append(target)
            else:
                radboud_preds.append(pred)
                radboud_targets.append(target)

        # plot top and min losses
        plot_samples(data, losses, preds, sampler.df[sampler.df.split==fold].image_id.values)
        plt.savefig(os.path.join(model_dir, "losses_fold-{0}.png".format(fold)), transparent=False)

        # confusion matrices
        if not regr:
            _ = plot_confusion_matrix_scipy(preds, targets, normalize=False, title='fold:{0} - qwk:{1:.3f}'.format(fold,val_qwk))
            plt.savefig(os.path.join(model_dir, "cm_fold-{0}.png".format(fold)), transparent=False)

            cm = plot_confusion_matrix_scipy(preds, targets, normalize=True, title='Norm. fold:{0} - qwk:{1:.3f}'.format(fold,val_qwk))
            plt.savefig(os.path.join(model_dir, "cm_fold-{0}-norm.png".format(fold)), transparent=False)
        else:
            _ = plot_confusion_matrix_scipy(preds, targets, normalize=False, title='fold:{0} - qwk:{1:.3f}'.format(fold,val_qwk))
            plt.savefig(os.path.join(model_dir, "cm_fold-{0}.png".format(fold)), transparent=False)

            cm = plot_confusion_matrix_scipy(preds, targets, normalize=True, title='Norm. fold:{0} - qwk:{1:.3f}'.format(fold,val_qwk))
            plt.savefig(os.path.join(model_dir, "cm_fold-{0}-norm.png".format(fold)), transparent=False)
        
        # save confusion matrix values
        np.save(os.path.join(model_dir, "cm_fold-{0}.npy".format(fold)), cm)
        
    
    cv_qwk = cohen_kappa_score(np.array(all_preds), np.array(all_targets), weights="quadratic")
    score_dict['cv_qwk'] = str(cv_qwk)
    score_dict['karolinska_qwk'] = str(cohen_kappa_score(karolinska_preds,karolinska_targets,weights="quadratic"))
    score_dict['radboud_qwk'] = str(cohen_kappa_score(radboud_preds,radboud_targets,weights="quadratic"))
    
    # save out-of-fold predictions
    np.save(os.path.join(model_dir, 'oof_preds.npy'), np.array(all_preds))
    np.save(os.path.join(model_dir, 'oof_trues.npy'), np.array(all_targets))
    
    with open(os.path.join(model_dir, 'eval.json'), 'w') as outfile:
        json.dump(score_dict, outfile, indent=4)
    
    # record for the notebook
    print(score_dict)
    plt.close('all')
