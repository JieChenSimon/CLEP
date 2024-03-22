import logging
import torch
import torch.nn as nn

# from __future__ import print_function

import torch
import torch.nn as nn
import logging

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        print('[INFO] the temperature is:', temperature)
        print('[INFO] the base_temperature is:', base_temperature)
        print('[INFO] the contrast_mode is:', contrast_mode)

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        stabilityfactor = 10000
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
            positive_pairs_indices = torch.where(mask == 1)
            negative_pairs_indices = torch.where(mask == 0)
            # print('the positive_pairs_indices is:', positive_pairs_indices)
            # print('the negative_pairs_indices is:', negative_pairs_indices)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))


        #输出shape
        # print('anchor_feature shape:', anchor_feature.shape)
        # print('contrast_feature shape:', contrast_feature.shape)
        #anchor_feature和contrast_feature


        # compute logits
        # print('the anchor_feature is:', anchor_feature)
        # print('the contrast_feature.T is:', contrast_feature.T)
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # print('the anchor_dot_contrast is:', anchor_dot_contrast)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        # logits_mean = torch.mean(anchor_dot_contrast, dim=1, keepdim=True)
        # print('the logits_max is:', logits_max)
        # logits = anchor_dot_contrast - logits_max.detach()
        # logits = anchor_dot_contrast除以10000
        logits = (anchor_dot_contrast - logits_max.detach()) / stabilityfactor

        # print('the logits is:', logits)

        # anchor_feature 和 contrast_feature 分别代表锚点特征和对比特征。
        # torch.matmul(anchor_feature, contrast_feature.T) 计算锚点和对比特征之间的点积，结果是一个相似度矩阵，其中每个元素表示一个锚点特征向量和一个对比特征向量之间的相似度。
        # 通过除以 self.temperature，调整相似度的尺度。温度参数用于控制相似度分数的分布，有助于学习过程的稳定性和效果。

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # print('the mask is:', mask)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        # print('the logits_mask is:', logits_mask)
        # logits_mask 创建了一个新的掩码，用于遮蔽掉自身对比的情况（即锚点与其自身的对比）。
        # torch.scatter 用于在指定位置将掩码设置为 0，从而确保每个样本不会与自己进行对比
        mask = mask * logits_mask
        # print('the mask after mask * logits_mask:', mask)

        # compute log_prob
        logits_tmp = torch.exp(logits)
        exp_logits = logits_tmp * logits_mask

        # print('the exp_logits is:', exp_logits)

        expsum = torch.log(exp_logits.sum(1, keepdim=True))

        log_prob = logits - expsum
        # print('the log_prob is:', log_prob)

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        # print('[INFO] loss is:', loss)
        # print('anchor_count is:', anchor_count)
        # print('batch_size is:', batch_size)
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class TripletLoss(nn.Module):

    def __init__(self, reduce = 'mean'):
        """
        If reduce == False, we calculate sample loss, instead of batch loss.
        """
        super(TripletLoss, self).__init__()
        self.reduce = reduce

    def forward(self, features, labels = None, margin = 10.0,
                weight = None, split = None):
        """
        Triplet loss for model.

        Args:
            features: hidden vector of shape [bsz, feature_dim]. e.g., (512, 128)
            labels: ground truth of shape [bsz].
            weight: sample weights to adjust the sample loss values
            split: whether it is in test data
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        
        # print('[INFO] batch size equal features.shape[0] is:', features.shape[0])
        batch_size = features.shape[0]

        pass_size = batch_size // 3
        """
        three shares of pass_size
        1) training data sample
        2) positive samples
        3) negative samples
        """
        anchor = features[:pass_size]
        positive = features[pass_size:pass_size*2]
        # negative = features[pass_size*2:]
        negative = features[pass_size*2:pass_size*2+pass_size]
        positive_losses = torch.maximum(torch.tensor(1e-10), torch.linalg.norm(anchor - positive, ord = 2, dim = 1))
        a = torch.tensor(0)
        # print('[INFO] torch.tensor(0) is:', a)
        # print("[INFO] Size of anchor along dimension 0:", anchor.size(0))
        # print("[INFO] Size of negative along dimension 0:", negative.size(0))
        # print("[INFO] Size of positive along dimension 0:", positive.size(0))
        b = margin - torch.linalg.norm(anchor - negative, ord=2, dim=1)
        
        # print('[INFO] margin - torch.linalg.norm(anchor - negative, ord=2, dim=1) is:', b)
        negative_losses = torch.maximum(torch.tensor(0), margin - torch.linalg.norm(anchor - negative, ord = 2, dim = 1))

        if weight is not None:
            anchor_weight = weight[:pass_size]
            positive_weight = weight[pass_size:pass_size*2]
            # negative_weight = weight[pass_size*2:]
            negative_weight = weight[pass_size*2:pass_size*2+pass_size]
            positive_losses = positive_losses * anchor_weight * positive_weight
            negative_losses = negative_losses * positive_weight * negative_weight
        
        loss = positive_losses + negative_losses

        if self.reduce == 'mean':
            loss = loss.mean()

        return loss

class TripletMSELoss(nn.Module):
    def __init__(self, reduce = 'mean'):
        super(TripletMSELoss, self).__init__()
        # reduce: whether use 'mean' reduction or keep sample loss
        self.reduce = reduce

    def forward(self, cae_lambda,
            x, x_prime,
            features, labels = None,
            margin = 10.0,
            weight = None,
            split = None):
        """
        Args:
            cae_lambda: scale the CAE loss
            x: input to the Autoencoder
            x_prime: decoded x' from Autoencoder
            features: hidden vector of shape [bsz, feature_dim].
            labels: ground truth of shape [bsz].
            weight: sample weights to adjust the sample loss values
            split: whether it is in test data
        Returns:
            A loss scalar.
        """
        Triplet = TripletLoss(reduce = self.reduce)
        supcon_loss = Triplet(features, labels = labels, margin = margin, weight = weight, split = split)

        mse_loss = torch.nn.functional.mse_loss(x, x_prime, reduction = self.reduce)
        
        loss = cae_lambda * supcon_loss + mse_loss
        
        del Triplet
        torch.cuda.empty_cache()

        return loss, supcon_loss, mse_loss

class HiDistanceXentLoss(nn.Module):
    def __init__(self, reduce = 'mean', sample_reduce = 'mean'):
        super(HiDistanceXentLoss, self).__init__()
        # reduce: whether use 'mean' reduction or keep sample loss
        self.reduce = reduce
        self.sample_reduce = sample_reduce

    def forward(self, xent_lambda,
            y_bin_pred, y_bin_batch,
            features, labels = None,
            margin = 10.0,
            weight = None,
            split = None):
        """
        Args:
            xent_lambda: scale the binary xent loss
            y_bin_pred: predicted MLP output
            y_bin_batch: binary one-hot encoded y
            features: hidden vector of shape [bsz, feature_dim].
            labels: ground truth of shape [bsz].
            margin: margin for HiDistanceLoss.
            weight: sample weights to adjust the sample loss values
            split: whether it is in test data, so we ignore these entries
        Returns:
            A loss scalar.
        """
        Dist = HiDistanceLoss(reduce = self.reduce, sample_reduce = self.sample_reduce)
        # try not giving any weight to HiDistanceLoss
        supcon_loss = Dist(features, y_bin_batch, labels = labels, margin = margin, weight = None, split = split)
        
        xent_bin_loss = torch.nn.functional.binary_cross_entropy(y_bin_pred[:, 1], y_bin_batch[:, 1],
                                                        reduction = self.reduce, weight = weight)
        
        if self.reduce == 'mean':
            xent_bin_loss = xent_bin_loss.mean()

        loss = supcon_loss + xent_lambda * xent_bin_loss
        
        del Dist
        torch.cuda.empty_cache()

        return loss, supcon_loss, xent_bin_loss
    

num_04_batch_list = []
num_05_batch_list = []
num_06_batch_list = []
num_07_batch_list = []
num_08_batch_list = []

class HiDistanceLoss(nn.Module):
    def __init__(self, reduce = 'mean', sample_reduce='mean', versions=None):
        """
        If reduce == False, we calculate sample loss, instead of batch loss.
        """
        super(HiDistanceLoss, self).__init__()
        self.reduce = reduce
        self.sample_reduce = sample_reduce
        self.versions = versions

    def forward(self, features, labels = None, versions=None, margin = 10.0,
                weight = None, split = None):
        """
        Pair distance loss.

        Args:
            features: hidden vector of shape [bsz, feature_dim]. e.g., (512, 128)
            binary_cat_labels: one-hot binary labels.
            labels: ground truth of shape [bsz].
            margin: margin for dissimilar distance.
            weight: sample weights to adjust the sample loss values
            split: whether it is in test data, so we ignore entries for these
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if labels == None:
            raise ValueError('Need to define labels in DistanceLoss')

        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        
        # similar masks

        # 创建二元标签掩码:
        # mask_{i,j}=1 if sample j has the same class as sample i.
        # binary_labels = binary_cat_labels[:, 1].view(-1, 1)
        binary_labels = labels.view(-1, 1).float()
        versions = versions.unsqueeze(0)  # 将versions变为1xN
        #检查一下里面的版本中0.4，0.5，0.6，0.7，0.8的个数
        # num_04 = (versions == 0.4).sum().item()
        # num_05 = (versions == 0.5).sum().item()
        # num_06 = (versions == 0.6).sum().item()
        # num_07 = (versions == 0.7).sum().item()
        # num_08 = (versions == 0.8).sum().item()
        # num_04_batch_list.append(num_04)
        # num_05_batch_list.append(num_05)
        # num_06_batch_list.append(num_06)
        # num_07_batch_list.append(num_07)
        # print('the number of 0.4:', num_04)
        # print('the number of 0.5:', num_05)
        # print('the number of 0.6:', num_06)
        # print('the number of 0.7:', num_07)
        # print('the number of 0.8:', num_08)
        
        # binary_mask = torch.eq(binary_labels, binary_labels.T).float().to(device)
        binary_mask = torch.eq(labels, labels.T).float().to(device)
        version_mask = torch.eq(versions, versions.T).float().to(device)
        version_diff_mask = torch.logical_not(version_mask).float().to(device)



        # 1.both benign samples[similar samples]
        flawless_labels = torch.logical_not(labels).float().to(device)
        both_flawless_mask = torch.matmul(flawless_labels, flawless_labels.T)
        num_ones_flawless = (both_flawless_mask == 1).sum().item()
        # print('the number of both flawless:', num_ones_flawless)


        #2. both vulnerable samples for different version [similar samples]
        both_vulnerable_diff_version_mask = binary_mask * version_diff_mask
        

        #same vulnerable sampler for all versions [similar samples]
        both_vulnerable_mask = torch.matmul(binary_labels, binary_labels.T)
   
        ##3. [highly similar samples]both vulnerable samples for same version
        both_vulnerable_same_version_mask = binary_mask * version_mask
        num_ones_vulnerable_same_version = (both_vulnerable_same_version_mask == 1).sum().item()
        # num_ones = (same_vulnerable_mask == 1).sum().item()
        # num_zeros = (same_vulnerable_mask == 0).sum().item()
        # print('the number of both vulnable same version:', num_ones_vulnerable_same_version)
        # print('same_vulnerable_mask:', same_vulnerable_mask)

 

        
        # logging.debug("=== new batch ===")
        # pseudo loss
        if self.reduce == 'none':
            tmp = other_mal_mask
            other_mal_mask = same_vulnerable_mask
            same_vulnerable_mask = tmp
            # debug
            # split_index = torch.nonzero(split, as_tuple=True)[0]
            # logging.debug(f'split_index, {split_index}')
        # logging.debug(f'binary_labels {binary_labels}')
        # logging.debug(f'binary_mask {binary_mask}')
        # logging.debug(f'labels {labels}')
        # logging.debug(f'multi_mask {multi_mask}')
        # logging.debug(f'other_mal_mask = binary_mask - multi_mask {other_mal_mask}')
        # logging.debug(f'flawless_labels {flawless_labels}')
        # logging.debug(f'same_flawless_mask {same_flawless_mask}')
        # logging.debug(f'same_vulnerable_mask = multi_mask - same_flawless_mask {same_vulnerable_mask}')
        
        # 4. dissimilar mask. vulnerable vs benign flawless labels
        one_vul_one_flawless_mask = torch.logical_not(binary_mask).float().to(device)

        # mask-out self-contrast cases
        diag_mask = torch.logical_not(torch.eye(batch_size)).float().to(device)



        # similar mask
        # binary_mask = binary_mask * diag_mask
        # other_mal_mask = other_mal_mask * diag_mask
        # same_flawless_mask = same_flawless_mask * diag_mask
        # same_vulnerable_mask = same_vulnerable_mask * diag_mask

        both_flawless_mask = both_flawless_mask * diag_mask
        both_vulnerable_mask = both_vulnerable_mask * diag_mask
        both_vulnerable_diff_version_mask = both_vulnerable_diff_version_mask * diag_mask
        both_vulnerable_same_version_mask = both_vulnerable_same_version_mask * diag_mask
        one_vul_one_flawless_mask = one_vul_one_flawless_mask * diag_mask



        



        # # adjust the masks based on test indices
        # if split is not None:
        #     split_index = torch.nonzero(split, as_tuple=True)[0]
        #     # instance-level loss, paired with training samples, pseudo loss
        #     # logging.debug(f'split_index, {split_index}')
        #     binary_negate_mask[:, split_index] = 0
        #     # multi_negate_mask[:, split_index] = 0
        #     binary_mask[:, split_index] = 0
        #     other_mal_mask[:, split_index] = 0
        #     same_flawless_mask[:, split_index] = 0
        #     same_vulnerable_mask[:, split_index] = 0

        # reference: https://github.com/Lightning-AI/metrics/blob/master/src/torchmetrics/functional/pairwise/euclidean.py
        # not taking the sqrt for numerical stability
        # features #torch.Size([32, 1, 320, 768])
        # features #torch.Size([32, 1, 320, 768])

        x = features.view(features.size(0), -1)
        y = features.view(features.size(0), -1)
        
        # x_norm = x.norm(dim=1, keepdim=True) #torch.Size([32, 1, 320, 768]) #每个点的向量的模的平方 构成的矩阵
        # y_norm = y.norm(dim=1).T #torch.Size([768, 320, 32])
        # x_norm_squared = x_norm * x_norm
        # y_norm_squared = y_norm * y_norm
        # x_y_transpose = x.mm(y.T)*2
        # distance_matrix = x_norm_squared + y_norm_squared - x_y_transpose
        pdist = nn.PairwiseDistance(p=2)
        distance_matrix = pdist(x,y)
        
        # distance_matrix = x_norm * x_norm + y_norm * y_norm - 2 * x.mm(y.T)
        distance_matrix = torch.maximum(torch.tensor(1e-10), distance_matrix)
        # logging.debug(f'distance_matrix {distance_matrix}')
        # #logging.debug(f'torch.isnan(distance_matrix).any() {torch.isnan(distance_matrix).any()}')
        # logging.debug(f'same_flawless_mask {same_flawless_mask}')
        # logging.debug(f'other_mal_mask {other_mal_mask}')
        # logging.debug(f'same_vulnerable_mask {same_vulnerable_mask}')
        # logging.debug(f'binary_negate_mask {binary_negate_mask}')
        
        # four types of pairs
        # 1. ben, ben. same_flawless_mask
        # 2. mal, mal from different families. other_mal_mask
        # 3. mal, mal from same families. same_vulnerable_mask
        # 4. ben, mal. binary_negate_mask

        # default is to compute mean for these values per sample
        if self.sample_reduce == 'mean' or self.sample_reduce == None:
            if weight == None:
                sum_both_flawless = torch.maximum(
                                    torch.sum(both_flawless_mask * distance_matrix, dim=1) - \
                                            both_flawless_mask.sum(1) * torch.tensor(margin),
                                    torch.tensor(0))
                
                sum_both_vul_diff_version = torch.maximum(
                                    torch.sum(both_vulnerable_diff_version_mask * distance_matrix, dim=1) - \
                                            both_vulnerable_diff_version_mask.sum(1) * torch.tensor(margin),
                                    torch.tensor(0))
                
                sum_both_vul_same_version = torch.sum(both_vulnerable_same_version_mask * distance_matrix, dim=1)

                num_ones_vul_flaw = one_vul_one_flawless_mask.sum(1)
                distance_matrix_one_vul_one_flaw = one_vul_one_flawless_mask * distance_matrix
                sum_one_vul_one_flawless = torch.maximum(num_ones_vul_flaw * torch.tensor(2 * margin) - torch.sum(distance_matrix_one_vul_one_flaw, dim=1), torch.tensor(0))  
                
                sum_one_vul_one_flawless = torch.maximum(
                                    one_vul_one_flawless_mask.sum(1) * torch.tensor(2 * margin) - \
                                            torch.sum(one_vul_one_flawless_mask * distance_matrix,
                                                    dim=1),
                                    torch.tensor(0))
                # logging.debug(f'sum_same_ben {sum_same_ben}, same_flawless_mask.sum(1) {same_flawless_mask.sum(1)}')
                # logging.debug(f'sum_other_mal {sum_other_mal}, other_mal_mask.sum(1) {other_mal_mask.sum(1)}')
                # logging.debug(f'sum_same_mal_fam {sum_same_mal_fam}, same_vulnerable_mask.sum(1) {same_vulnerable_mask.sum(1)}')
                # logging.debug(f'sum_bin_neg {sum_bin_neg}, binary_negate_mask.sum(1) {binary_negate_mask.sum(1)}')
            # weighted loss
            else:
                weight_matrix = torch.matmul(weight.view(-1, 1), weight.view(1, -1)).to(device)
                sum_both_flawless = torch.maximum(
                                    torch.sum(both_flawless_mask * distance_matrix * weight_matrix, dim=1) - \
                                            both_flawless_mask.sum(1) * torch.tensor(margin),
                                    torch.tensor(0))
                sum_both_vul_diff_version = torch.maximum(
                                    torch.sum(both_vulnerable_diff_version_mask * distance_matrix * weight_matrix, dim=1) - \
                                            both_vulnerable_diff_version_mask.sum(1) * torch.tensor(margin),
                                    torch.tensor(0))
                
                sum_both_vul_same_version = torch.sum(both_vulnerable_same_version_mask * distance_matrix * weight_matrix, dim=1)

                weight_prime = torch.div(1.0, weight)
                weight_matrix_prime = torch.matmul(weight_prime.view(-1, 1), weight_prime.view(1, -1)).to(device)
                sum_one_vul_one_flawless = torch.maximum(
                                    one_vul_one_flawless_mask.sum(1) * torch.tensor(2 * margin) - \
                                            torch.sum(one_vul_one_flawless_mask * distance_matrix * weight_matrix_prime,
                                                    dim=1),
                                    torch.tensor(0))
            # 计算各个部分的损失
            # num_ones_both_flawless = both_flawless_mask.sum(1)
            # loss_both_flawless = sum_both_flawless / torch.maximum(num_ones_both_flawless, torch.tensor(1))
            # num_ones_vul_diff_version = both_vulnerable_diff_version_mask.sum(1)
            # loss_both_vul_diff_version = sum_both_vul_diff_version / torch.maximum(num_ones_vul_diff_version, torch.tensor(1))
            # num_ones_vulnerable_same_version = both_vulnerable_same_version_mask.sum(1)
            # loss_both_vul_same_version = sum_both_vul_same_version / torch.maximum(num_ones_vulnerable_same_version, torch.tensor(1))

            # num_ones_vul_flaw = one_vul_one_flawless_mask.sum(1)
            # loss_one_vul_one_flawless = sum_one_vul_one_flawless / torch.maximum(num_ones_vul_flaw, torch.tensor(1))
            # print('loss_both_flawless:', loss_both_flawless)
            # print('loss_both_vul_diff_version:', loss_both_vul_diff_version)
            # print('loss_both_vul_same_version:', loss_both_vul_same_version)
            # print('loss_one_vul_one_flawless:', loss_one_vul_one_flawless)
            # # 将所有损失相加得到总损失
            # loss = loss_both_flawless + loss_both_vul_diff_version + loss_both_vul_same_version + loss_one_vul_one_flawless 
            # print('loss:', loss)   
            loss = sum_both_flawless / torch.maximum(both_flawless_mask.sum(1), torch.tensor(1)) + \
                    sum_both_vul_diff_version / torch.maximum(both_vulnerable_diff_version_mask.sum(1), torch.tensor(1)) + \
                    sum_both_vul_same_version / torch.maximum(both_vulnerable_same_version_mask.sum(1), torch.tensor(1)) + \
                    sum_one_vul_one_flawless / torch.maximum(one_vul_one_flawless_mask.sum(1), torch.tensor(1))
            # print('loss:', loss)
        elif self.sample_reduce == 'max':
            max_both_flawless = torch.maximum(
                                torch.amax(both_flawless_mask * distance_matrix, 1) - \
                                        torch.tensor(margin),
                                torch.tensor(0))
            max_both_vul_diff_version = torch.maximum(
                                torch.amax(both_vulnerable_diff_version_mask * distance_matrix, 1) - \
                                        torch.tensor(margin),
                                torch.tensor(0))
            max_both_vul_same_version = torch.amax(both_vulnerable_same_version_mask * distance_matrix, 1)
            max_one_vul_one_flawless = torch.maximum(
                                torch.tensor(2 * margin) - \
                                        torch.amin(one_vul_one_flawless_mask * distance_matrix, 1),
                                torch.tensor(0))
            loss = max_both_flawless + max_both_vul_diff_version + max_both_vul_same_version + max_one_vul_one_flawless
        else:
            raise Exception(f'sample_reduce = {self.sample_reduce} not implemented yet.')

        if self.reduce == 'mean':
            loss = loss.mean()
        
        return loss

class HiDistanceXentLoss(nn.Module):
    def __init__(self, reduce = 'mean', sample_reduce = 'mean'):
        super(HiDistanceXentLoss, self).__init__()
        # reduce: whether use 'mean' reduction or keep sample loss
        self.reduce = reduce
        self.sample_reduce = sample_reduce
        


    def forward(self, xent_lambda,
            features, labels = None,
            versions=None,
            margin = 10.0,
            weight = None,
            split = None):
        """
        Args:
            xent_lambda: scale the binary xent loss
            y_bin_pred: predicted MLP output
            y_bin_batch: binary one-hot encoded y
            features: hidden vector of shape [bsz, feature_dim].
            labels: ground truth of shape [bsz].
            margin: margin for HiDistanceLoss.
            weight: sample weights to adjust the sample loss values
            split: whether it is in test data, so we ignore these entries
        Returns:
            A loss scalar.
        """
        Dist = HiDistanceLoss(reduce = self.reduce, sample_reduce = self.sample_reduce)
        # try not giving any weight to HiDistanceLoss
        supcon_loss = Dist(features, labels = labels, versions = versions, margin = margin, weight = None, split = split)
        
        # xent_bin_loss = torch.nn.functional.binary_cross_entropy(y_bin_pred[:, 1], y_bin_batch[:, 1],
        #                                                 reduction = self.reduce, weight = weight)
        
        # if self.reduce == 'mean':
        #     xent_bin_loss = xent_bin_loss.mean()

        # loss = supcon_loss + xent_lambda * xent_bin_loss
        
        del Dist
        torch.cuda.empty_cache()

        return supcon_loss