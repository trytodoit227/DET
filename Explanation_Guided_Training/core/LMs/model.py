import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import rcParams
import torch.nn as nn
import numpy as np
from transformers import PreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput
# from core.utils import init_random_state
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'




def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:1" if use_cuda else "cpu")
    return device
def relu_evidence(y):
    return F.relu(y)

def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))

def softplus_evidence(y):
    return F.softplus(y)

def loglikelihood_loss(y, alpha, device=None):
    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood

def kl_divergence(alpha, num_classes, device=None):
    if not device:
        device = get_device()
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl

def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, device=None):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)

    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return A + kl_div

def edl_log_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    evidence = exp_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        edl_loss(
            torch.log, target, alpha, epoch_num, num_classes, annealing_step, device
        )
    )
    return loss,alpha



def edl_mse_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    evidence = exp_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        mse_loss(target, alpha, epoch_num, num_classes, annealing_step, device=device)
    )
    return loss
def mse_loss(y, alpha, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)
    loglikelihood = loglikelihood_loss(y, alpha, device=device)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return loglikelihood + kl_div


def plot_uncertainty_distribution(uncertainties, save_path="uncertainty_distribution.pdf"):
    """
    Plot the uncertainty distribution and save as PDF.

    Parameters:
    - uncertainties (list): A list of uncertainty values.
    - save_path (str): The file path to save the PDF.
    """
    rcParams['font.family'] = 'Times New Roman'
    rcParams['font.size'] = 20
    # Plot histogram
    plt.figure(figsize=(8, 6))
    uncertainties=uncertainties.cpu().numpy()
    plt.hist(uncertainties, bins=30, color='lightblue', edgecolor='black', alpha=0.7)
    plt.xlabel("Uncertainty", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.grid(alpha=0.3)

    # Save to PDF
    plt.savefig(save_path, format="pdf")
    plt.close()
    print(f"Uncertainty distribution saved to {save_path}")

def plot_uncertainty_distributions1(uncertainties, uncertainties1, save_path="uncertainty_distributions.pdf"):
    """
    Plot the uncertainty distributions of two datasets and save as a PDF.

    Parameters:
    - uncertainties (list): A list of uncertainty values for dataset 1.
    - uncertainties1 (list): A list of uncertainty values for dataset 2.
    - save_path (str): The file path to save the PDF.
    """
    rcParams['font.family'] = 'Times New Roman'
    rcParams['font.size'] = 20

    # Convert lists to numpy arrays
    uncertainties = np.array(uncertainties)
    uncertainties1 = np.array(uncertainties1)

    # Plot histograms
    plt.figure(figsize=(8, 6))
    plt.hist(
        uncertainties, bins=30, color='lightblue', edgecolor='black', alpha=0.7, label='Correct classifications'
    )
    plt.hist(
        uncertainties1, bins=30, color='green', edgecolor='black', alpha=0.7, label='Misclassifications'
    )
    plt.xlabel("Uncertainty", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=14)

    # Save the figure as a PDF
    plt.savefig(save_path, format="pdf", bbox_inches='tight')
    plt.close()
    print(f"Uncertainty distributions saved to {save_path}")



def edl_inference1(alpha, num_classes):
    """
    Evidential Deep Learning inference process.

    Parameters:
    - alpha (np.ndarray): The evidence vector output by the model, shape (N, K),
                          where N is the number of samples, and K is the number of classes.
    - num_classes (int): The number of classes (K).

    Returns:
    - uncertainties (np.ndarray): The model's uncertainty values for each sample, shape (N,).
    """
    # Step 1: Calculate total evidence (S) for each sample (sum over class dimension)
    S = torch.sum(alpha, dim=1)

    # Step 2: Compute uncertainty (u = K / S) for each sample
    uncertainties = num_classes / S

    return uncertainties




class BertClassifier(PreTrainedModel):
    def __init__(self, model, n_labels, dropout=0.0, seed=0, cla_bias=True, feat_shrink=''):
        super().__init__(model.config)
        self.bert_encoder = model
        self.dropout = nn.Dropout(dropout)
        self.feat_shrink = feat_shrink
        hidden_dim = model.config.hidden_size
        # self.loss_func = nn.CrossEntropyLoss(
        #     label_smoothing=0.3, reduction='mean') #raw
        self.loss_func = nn.CrossEntropyLoss(reduction='mean')

        if feat_shrink:
            self.feat_shrink_layer = nn.Linear(
                model.config.hidden_size, int(feat_shrink), bias=cla_bias)
            hidden_dim = int(feat_shrink)
        self.classifier = nn.Linear(hidden_dim, n_labels, bias=cla_bias)
        # init_random_state(seed)
        self.epoch_num = None
    def set_epoch(self, epoch_num):
        self.epoch_num = epoch_num

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                return_dict=None,
                preds=None,
                epoch_num=None,
                num_classes=None,
                annealing_step=None
                ):
        epoch_num = epoch_num if epoch_num is not None else self.epoch_num
        epoch_num = epoch_num if epoch_num is not None else 1
        outputs = self.bert_encoder(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    return_dict=return_dict,
                                    output_hidden_states=True)
        # outputs[0]=last hidden state
        emb = self.dropout(outputs['hidden_states'][-1])
        # Use CLS Emb as sentence emb.
        cls_token_emb = emb.permute(1, 0, 2)[0]
        if self.feat_shrink:
            cls_token_emb = self.feat_shrink_layer(cls_token_emb)
        logits = self.classifier(cls_token_emb)

        if labels.shape[-1] == 1:
            labels = labels.squeeze()


        # #  EDL MSE loss

        num_classes = 3 #for trump 3 ethos 2 overruling 2 values 3
        print(labels)
        labels= F.one_hot(labels, num_classes=num_classes)
        edl_loss,_ = edl_log_loss(logits, labels, epoch_num, num_classes, 10, device=logits.device)
        #
        # print('losstosee:',loss)
        print('edl_loss:',edl_loss)
        return TokenClassifierOutput(loss=edl_loss, logits=logits)


class BertClaInfModel(PreTrainedModel):
    def __init__(self, model, emb, pred, feat_shrink=''):
        super().__init__(model.config)
        self.bert_classifier = model
        self.emb, self.pred = emb, pred
        self.feat_shrink = feat_shrink
        self.loss_func = nn.CrossEntropyLoss(
            label_smoothing=0.3, reduction='mean')
        self.epoch_num = None
        self.uncertainties_tensor = None
        self.pred_label = []
        self.Label=[]
    def set_epoch(self, epoch_num):
        self.epoch_num = epoch_num

    @torch.no_grad()
    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                return_dict=None,
                node_id=None,
                epoch_num=None):

        # Extract outputs from the model
        bert_outputs = self.bert_classifier.bert_encoder(input_ids=input_ids,
                                                         attention_mask=attention_mask,
                                                         return_dict=return_dict,
                                                         output_hidden_states=True)
        # outputs[0]=last hidden state
        emb = bert_outputs['hidden_states'][-1]
        # Use CLS Emb as sentence emb.
        cls_token_emb = emb.permute(1, 0, 2)[0]
        if self.feat_shrink:
            cls_token_emb = self.bert_classifier.feat_shrink_layer(
                cls_token_emb)
        logits = self.bert_classifier.classifier(cls_token_emb)

        # Save prediction and embeddings to disk (memmap)
        batch_nodes = node_id.cpu().numpy()
        self.emb[batch_nodes] = cls_token_emb.cpu().numpy().astype(np.float16)
        self.pred[batch_nodes] = logits.cpu().numpy().astype(np.float16)

        if labels.shape[-1] == 1:
            labels = labels.squeeze()
        self.Label.append(labels.cpu())
        Label=torch.cat(self.Label, dim=0)
        Label = Label.numpy()

        # #  EDL MSE Loss
        epoch_num = epoch_num if epoch_num is not None else self.epoch_num
        epoch_num = epoch_num if epoch_num is not None else 1

        num_classes = 3
        labels= F.one_hot(labels, num_classes=num_classes)
        edl_loss,alpha = edl_log_loss(logits, labels, epoch_num, num_classes, 10, device=logits.device)
        S = torch.sum(alpha, dim=1, keepdim=True)
        probabilities = alpha / S
        pred_label = torch.argmax(probabilities, dim=1)
        self.pred_label.append(pred_label.cpu())
        pred_label=torch.cat(self.pred_label, dim=0)
        pred_label = pred_label.numpy()



        # Calculate uncertainty
        uncertainties = edl_inference1(alpha, num_classes)
        print("Uncertainties for each sample:", uncertainties)

        # 拼接 uncertainties
        if self.uncertainties_tensor is None:
            self.uncertainties_tensor = torch.tensor(
                uncertainties, device=logits.device)
        else:
            self.uncertainties_tensor = torch.cat(
                (self.uncertainties_tensor, torch.tensor(uncertainties, device=logits.device)), dim=0)

        uncertainties_numpy = np.array(self.uncertainties_tensor.cpu())

        mask = uncertainties_numpy <= 0.8

        filtered_pred_label = pred_label[mask]
        filtered_label = Label[mask]

        accuracy_filtered = (np.array(filtered_pred_label) == np.array(filtered_label)).sum() / len(filtered_label)
        print(f"new Accuracy for mask: {accuracy_filtered:.4f}")



        return TokenClassifierOutput(loss=edl_loss, logits=logits)
