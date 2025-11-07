import numpy as np
import torch.optim
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

import utils.loss
from utils.datasets import next_batch
from utils.evaluation import evaluation, get_cluster_sols
from models.baseModels import *
from utils.loss import *
from torch.nn.functional import normalize
from utils.util import target_l2
from utils.visualization import *


class RCIMVC(nn.Module):
    def __init__(self, config):
        super(RCIMVC, self).__init__()
        self._config = config
        self._input_dim1 = config['Autoencoder']['gcnEncoder1'][0]
        self._input_dim2 = config['Autoencoder']['gcnEncoder2'][0]
        self._input_dim3 = config['Autoencoder']['gcnEncoder3'][0]
        self._latent_dim = config['Autoencoder']['gcnEncoder1'][-1]
        self._n_clusters = config['n_clusters']
        self.batchsize = config['training']['batch_size']

        self.gcnEncoder1 = GraphEncoder(config['Autoencoder']['gcnEncoder1'], 'relu', True)
        self.gcnEncoder2 = GraphEncoder(config['Autoencoder']['gcnEncoder2'], 'relu', True)
        self.gcnEncoder3 = GraphEncoder(config['Autoencoder']['gcnEncoder3'], 'relu', True)

        self.instance_projector1 = InstanceProject(self._latent_dim)
        self.instance_projector2 = InstanceProject(self._latent_dim)
        self.instance_projector3 = InstanceProject(self._latent_dim)

        self.cluster = ClusterProject(self._latent_dim, self._n_clusters)
        self.fusion = AttentionLayer_3v(self._latent_dim)

    def forward(self, x1, x2, x3, adj1, adj2, adj3):
        h1 = self.gcnEncoder1(x1, adj1)
        h2 = self.gcnEncoder2(x2, adj2)
        h3 = self.gcnEncoder3(x3, adj3)
        z1 = normalize(self.instance_projector1(h1), dim=1)
        z2 = normalize(self.instance_projector2(h2), dim=1)
        z3 = normalize(self.instance_projector3(h3), dim=1)
        y1, p1 = self.cluster(h1)
        y2, p2 = self.cluster(h2)
        y3, p3 = self.cluster(h3)
        return h1, h2, h3, z1, z2, z3, y1, y2, y3, p1, p2, p3

    def eval_acc(self, z, Y_list, accumulated_metrics, logger):
        """cal acc"""
        z = z.cpu().numpy()
        y_pred, _ = get_cluster_sols(z, ClusterClass=KMeans, n_clusters=self._n_clusters, init_args={'n_init': 10})
        scores = evaluation(y_pred=y_pred, y_true=Y_list[0], accumulated_metrics=accumulated_metrics)
        logger.info("\033[2;29m" + 'trainingset_view1 ' + str(scores) + "\033[0m")

    def run_train(self, x_train, Y_list, adj, optimizer, logger, accumulated_metrics, device):
        LOSS = []
        lamb1 = 1
        lamb2 = 1
        lamb3 = 1
        lamb4 = 1
        attention = True
        if lamb2 == 0:
            lamb3 = 0
            attention = False
        epochs = self._config['training']['epoch']
        print_num = self._config['print_num']
        batch_size = self._config['training']['batch_size']
        batch_size = batch_size if x_train[0].shape[0] > batch_size else x_train[0].shape[0]

        criterion_cluster = ClusterLoss(self._n_clusters, 0.5, device).to(device)

        # train the model
        for k in range(epochs):
            h1, h2, h3, z1, z2, z3, y1, y2, y3, p1, p2, p3 = self(x_train[0], x_train[1], x_train[2], adj[0], adj[1], adj[2])
            if attention:
                h = self.fusion(h1, h2, h3)
            else:
                h = (h1 + h2 + h3) / 3

            # cluster contrastive loss
            cluster_loss = criterion_cluster(y1, y2) + criterion_cluster(y2, y3) + criterion_cluster(y1, y3)
            loss = lamb1 * cluster_loss           

            # high confidence loss
            y, _ = self.cluster(h)
            y_max = torch.maximum(y1, y2)
            y_max = torch.maximum(y_max, y3)
            y_max = torch.maximum(y_max, y)
            y_max = target_l2(y_max)
            y = torch.where(y < EPS, torch.tensor([EPS], device=y.device), y)
            hc_loss = F.kl_div(y.log(), y_max.detach(), reduction='batchmean')
            loss += lamb3 * hc_loss

            # Try to add memo_loss
            q_memo = torch.cat((y1, y2, y3), dim=0)
            q_mean = (y1+y2+y3)/len(q_memo)
            memo_loss = torch.sum(q_mean * torch.log(q_mean))/self._n_clusters
            loss += lamb4 * memo_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            LOSS.append(loss.item())
            if (k + 1) % print_num == 0:  # evaluation  k == 0 or
                output = ("Epoch:{:.0f}/{:.0f}===>loss={:.4f}".format((k + 1), epochs, loss.item()))
                logger.info("\033[2;29m" + output + "\033[0m")
                with torch.no_grad():
                    h1, h2, h3, z1, z2, z3, y1, y2, y3, p1, p2, p3 = self(x_train[0], x_train[1], x_train[2], adj[0], adj[1], adj[2])
                    if attention:
                        h = self.fusion(h1, h2, h3)
                    else:
                        h = (h1 + h2 + h3) / 3
                    y, _ = self.cluster(h)
                    y = y.data.cpu().numpy().argmax(1)
                    scores = evaluation(y_pred=y, y_true=Y_list[0], accumulated_metrics=accumulated_metrics)
                    print(str(scores))
                    if lamb1 == 0:
                        self.eval_acc(h, Y_list, accumulated_metrics, logger)
        return accumulated_metrics['acc'][-1], accumulated_metrics['nmi'][-1], accumulated_metrics['ARI'][-1], h, h1, h2, y 
