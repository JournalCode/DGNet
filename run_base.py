import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np
import random
import sys
import tqdm
import time
import argparse
import torch

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from sklearn.metrics import log_loss, roc_auc_score

sys.path.append("..")
from model.models import *

from utils.utils_de import *
from utils.earlystoping import EarlyStopping, EarlyStoppingLoss

from utils_get_data_model import get_model, CTRModelArguments, get_dataset

config = CTRModelArguments 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model,
          optimizer,
          data_loader,
          criterion,
          num_loss=1):
    model.train()
    pred = list()
    target = list()
    total_loss = 0
    for i, (user_item, label) in enumerate(tqdm.tqdm(data_loader)):
        label = label.float()
        user_item = user_item.long().cuda()
        label = label.cuda()

        model.zero_grad()
        
        if num_loss == 1:
            pred_y = torch.sigmoid(model(user_item).squeeze(1))
            loss_y = criterion(pred_y, label)
        elif num_loss==3:
            pred_y, loss_y = model.add_loss(user_item, label, criterion)
        
        elif num_loss==4:
            pred_y, loss_y = model.add_kl_loss(user_item, label, criterion)
            
        loss = loss_y 
        loss.backward()
        optimizer.step()

        pred.extend(pred_y.tolist())
        target.extend(label.tolist())
        total_loss += loss.item()

    ave_loss = total_loss / (i + 1)
    return ave_loss


def test_roc(model, data_loader):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for fields, target in tqdm.tqdm(
                data_loader, smoothing=0, mininterval=1.0):
            fields = fields.long()
            target = target.float()
            fields, target = fields.cuda(), target.cuda()
            y = torch.sigmoid(model(fields).squeeze(1))

            targets.extend(target.tolist())
            predicts.extend(y.tolist())
    return roc_auc_score(targets, predicts), log_loss(targets, predicts)


def main(dataset_name, model_name, epoch, embed_dim, learning_rate,
         batch_size, weight_decay, save_dir, path, hint, num_loss=1):
    # path = r"/root/autodl-tmp/CTR/torch_FICTR/data/"
    field_dims, trainLoader, validLoader, testLoader = get_dataset(dataset_name, batch_size=batch_size)
    print(field_dims)
    time_fix = time.strftime("%m%d%H%M%S", time.localtime())
    print(time_fix)
    for K in [embed_dim]:
        paths = os.path.join(save_dir, str(K))
        if not os.path.exists(paths):
            os.makedirs(paths)
        with open(paths + f"/{model_name}_{K}_{batch_size}_{time_fix}_{num_loss}.p",
                  "a+") as fout:
            fout.write("dataset_name：{}\tBatch_size:{}\tembed_dim:{}\tlearning_rate:{}\tweight_decay:{}\tnum_loss{}\n"
                       .format(dataset_name, batch_size, K, learning_rate, weight_decay,num_loss))
            print("Start train -- K : {}".format(K))
            criterion = torch.nn.BCELoss()
            
            config.embed_dim = embed_dim
            config.batch_size = batch_size
            config.field_dims = field_dims
            # config.bridge_type = "concat"
            
            model = get_model(
                model_name=model_name,
                field_dims=field_dims,
                config=config).cuda()

            params = count_params(model)
            fout.write("hint:{}\n".format(hint))
            fout.write("count_params:{}\n".format(params))
            print(params)

            optimizer = torch.optim.Adam(
                params=model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay)

            # Initial EarlyStopping
            # early_stopping = EarlyStoppingLoss(patience=6, verbose=True, prefix=path)
            early_stopping = EarlyStopping(patience=6, verbose=True, prefix=path)
            scheduler_min = ReduceLROnPlateau(optimizer, 'min', verbose=True, 
                                              patience=2, factor=0.1)
            val_auc_best = 0
            auc_index_record = ""

            val_loss_best = 1000
            loss_index_record = ""

            for epoch_i in range(epoch):
                print(__file__, model_name, K, epoch_i, "/", epoch)
                print("dataset_name:{}\tmodel_name:{}\tBatch_size:{}\tembed_dim:{}\tlearning_rate:{}\tStartTime:{}\tweight_decay:{}\tnum_loss:{}\thint:{}\t"
                      .format(dataset_name, model_name, batch_size, K, learning_rate, time.strftime("%d%H%M%S", time.localtime()), weight_decay,num_loss,hint))
                start = time.time()

                train_loss = train(model, optimizer, trainLoader, criterion, num_loss=num_loss)
                val_auc, val_loss = test_roc(model, validLoader)
                test_auc, test_loss = test_roc(model, testLoader)

                scheduler_min.step(val_loss)
                print(f"epoch:{epoch_i},lr:{scheduler_min.get_last_lr()}")
                end = time.time()
                if val_loss < val_loss_best:
                    # torch.save({"state_dict": model.state_dict(), "best_auc": val_auc_best},
                    #            paths + f"/{model_name}_final_{K}_{time_fix}.pt")
                    print("save model:{}".format(test_loss))
                    torch.save(model, paths + f"/{model_name}_best_auc_{K}_{time_fix}.pkl")

                if val_auc > val_auc_best:
                    val_auc_best = val_auc
                    auc_index_record = "epoch_i:{}\t{:.6f}\t{:.6f}".format(epoch_i, test_auc, test_loss)

                if val_loss < val_loss_best:
                    val_loss_best = val_loss
                    loss_index_record = "epoch_i:{}\t{:.6f}\t{:.6f}".format(epoch_i, test_auc, test_loss)

                print(
                    "Train  K:{}\tEpoch:{}\ttrain_loss:{:.6f}\tval_loss:{:.6f}\tval_auc:{:.6f}\ttime:{:.6f}\ttest_loss:{:.6f}\ttest_auc:{:.6f}\n"
                    .format(K, epoch_i, train_loss, val_loss, val_auc, end - start, test_loss, test_auc))

                fout.write(
                    "Train  K:{}\tEpoch:{}\ttrain_loss:{:.6f}\tval_loss:{:.6f}\tval_auc:{:.6f}\ttime:{:.6f}\ttest_loss:{:.6f}\ttest_auc:{:.6f}\n"
                    .format(K, epoch_i, train_loss, val_loss, val_auc, end - start, test_loss, test_auc))

                early_stopping(val_auc)
                # early_stopping(val_loss)

                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            print("Test:{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n"
                  .format(K, val_auc, val_auc_best, val_loss, val_loss_best, test_loss, test_auc))

            fout.write("Test:{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n"
                       .format(K, val_auc, val_auc_best, val_loss, val_loss_best, test_loss, test_auc))

            fout.write("auc_best:\t{}\nloss_best:\t{}".format(auc_index_record, loss_index_record))


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='frappe', help="")
    parser.add_argument('--save_dir', default='../chkpt_dge/models/models', help="") 
    parser.add_argument('--path', default="../data/", help="")
    parser.add_argument('--model_name', default='DCNv2', help="")
    parser.add_argument('--epoch', type=int, default=2, help="")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="learning rate")
    parser.add_argument('--batch_size', type=int, default=4096, help="batch_size")
    parser.add_argument('--weight_decay', type=float, default=1e-6, help="")
    parser.add_argument('--device', default='cuda:0', help="cuda:0")
    parser.add_argument('--choice', default=0, type=int, help="choice")
    parser.add_argument('--hint', default="DGNet", help="")
    parser.add_argument('--bridge', default="att", help="")
    parser.add_argument('--repeats', default=6, type=int, help="choice")
    parser.add_argument('--num_loss', default=1, type=int, help="choice")
    parser.add_argument('--embed_dim', default=16, type=int, help="the size of feature dimension")
    args = parser.parse_args()
    print(type(args))
 
    model_names = ["DGNet_DCNv2"] * args.repeats
    print(model_names) 
    for dataset_name in ["ml1m"]:       
        if dataset_name == "ml1m":
            args.learning_rate = 1e-3
            args.weight_decay = 1e-5
            args.batch_size = 4096
        elif dataset_name == "mltag": 
            args.learning_rate = 1e-3
            args.weight_decay = 1e-5
            args.batch_size = 4096
        elif dataset_name == "frappe":
            args.learning_rate = 1e-3
            args.weight_decay = 1e-5
            args.batch_size = 4096

                     
        for embed_dim in [16]: 
            for name in model_names:
                print(args.hint)
                args.save_dir = f"./chkpt_dge/models/{name}/{dataset_name}"
                main(dataset_name=dataset_name,
                    model_name=name,
                    epoch=args.epoch,
                    learning_rate=args.learning_rate,
                    batch_size=args.batch_size,
                    weight_decay=args.weight_decay,
                    save_dir=args.save_dir,
                    path=args.path,
                    embed_dim=embed_dim,
                    hint=args.hint,
                    num_loss=args.num_loss 
                    )