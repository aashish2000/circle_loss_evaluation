import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torchvision import transforms
from sklearn import metrics

class ImageReader(Dataset):

    def __init__(self, data_path, data_name, data_type, crop_type):
        if crop_type == 'cropped' and data_name not in ['car', 'cub']:
            raise NotImplementedError('cropped data only works for car or cub dataset')

        data_dict = torch.load('{}/{}/{}_data_dicts.pth'.format(data_path, data_name, crop_type))[data_type]
        self.class_to_idx = dict(zip(sorted(data_dict), range(len(data_dict))))
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        if data_type == 'train':
            self.transform = transforms.Compose([transforms.Resize((252, 252)), transforms.RandomCrop(224),
                                                 transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        else:
            self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
        self.images, self.labels = [], []
        for label, image_list in data_dict.items():
            self.images += image_list
            self.labels += [self.class_to_idx[label]] * len(image_list)

    def __getitem__(self, index):
        path, target = self.images[index], self.labels[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.images)

def precision(feature_vectors, feature_labels, rank, gallery_vectors=None, gallery_labels=None):
    num_features = len(feature_labels)
    feature_labels = torch.tensor(feature_labels, device=feature_vectors.device)
    gallery_vectors = feature_vectors if gallery_vectors is None else gallery_vectors

    dist_matrix = torch.cdist(feature_vectors.unsqueeze(0), gallery_vectors.unsqueeze(0)).squeeze(0)

    if gallery_labels is None:
        dist_matrix.fill_diagonal_(float('inf'))
        gallery_labels = feature_labels
    else:
        gallery_labels = torch.tensor(gallery_labels, device=feature_vectors.device)

    idx = dist_matrix.topk(k=rank[-1], dim=-1, largest=False)[1]
    acc_list = []
    prec_list = []
    for r in rank:
        y_pred = (gallery_labels[idx[:, r-1:r]]).cpu().numpy()
        y_true = feature_labels.unsqueeze(dim=-1).cpu().numpy()

        #acc_list.append(metrics.recall_score(y_true, y_pred, average='weighted'))
        prec_list.append(metrics.precision_score(y_true, y_pred, average='weighted'))
        #print(metrics.precision_score(y_true, y_pred, average='None'))
        #correct = (gallery_labels[idx[:, 0:r]] == feature_labels.unsqueeze(dim=-1)).any(dim=-1).float()
        #acc_list.append((torch.sum(correct) / num_features).item())
    return prec_list

def recall(feature_vectors, feature_labels, rank, gallery_vectors=None, gallery_labels=None):
    num_features = len(feature_labels)
    feature_labels = torch.tensor(feature_labels, device=feature_vectors.device)
    gallery_vectors = feature_vectors if gallery_vectors is None else gallery_vectors

    dist_matrix = torch.cdist(feature_vectors.unsqueeze(0), gallery_vectors.unsqueeze(0)).squeeze(0)

    if gallery_labels is None:
        dist_matrix.fill_diagonal_(float('inf'))
        gallery_labels = feature_labels
    else:
        gallery_labels = torch.tensor(gallery_labels, device=feature_vectors.device)

    idx = dist_matrix.topk(k=rank[-1], dim=-1, largest=False)[1]
    acc_list = []
    prec_list = []
    for r in rank:
        y_pred = (gallery_labels[idx[:, r-1:r]]).cpu().numpy()
        y_true = feature_labels.unsqueeze(dim=-1).cpu().numpy()

        correct_1 = gallery_labels[idx[:, 0:r]] == feature_labels.unsqueeze(dim=-1)).any(dim=-1)).float()
        correct_2 = gallery_labels[idx[:, r-1:r]] == feature_labels.unsqueeze(dim=-1)).any(dim=-1)).float()
        print(correct_1)
        print(correct_2)
        acc_list.append(metrics.recall_score(y_true, y_pred, average='weighted'))
        #prec_list.append(metrics.precision_score(y_true, y_pred, average='weighted'))
        #print(metrics.precision_score(y_true, y_pred, average='None'))
        #correct = (gallery_labels[idx[:, 0:r]] == feature_labels.unsqueeze(dim=-1)).any(dim=-1).float()
        #acc_list.append((torch.sum(correct) / num_features).item())
    return acc_list


class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, smoothing=0.1, temperature=1.0):
        super().__init__()
        self.smoothing = smoothing
        self.temperature = temperature

    def forward(self, x, target):
        log_probs = F.log_softmax(x / self.temperature, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(dim=-1)).squeeze(dim=-1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin=1.0, gamma = 80):
        super().__init__()
        self.margin = margin
        self.gamma = gamma 
        self.soft_plus = nn.Softplus()

    @staticmethod
    def get_anchor_positive_triplet_mask(target):
        mask = torch.eq(target.unsqueeze(0), target.unsqueeze(1))
        mask.fill_diagonal_(False)
        return mask

    @staticmethod
    def get_anchor_negative_triplet_mask(target):
        labels_equal = torch.eq(target.unsqueeze(0), target.unsqueeze(1))
        mask = ~ labels_equal
        return mask

    def forward(self, x, target):
        pairwise_dist = torch.cdist(x.unsqueeze(0), x.unsqueeze(0)).squeeze(0)

        mask_anchor_positive = self.get_anchor_positive_triplet_mask(target)
        anchor_positive_dist = mask_anchor_positive.float() * pairwise_dist
        hardest_positive_dist = anchor_positive_dist.max(1, True)[0]

        mask_anchor_negative = self.get_anchor_negative_triplet_mask(target)
        # make positive and anchor to be exclusive through maximizing the dist
        max_anchor_negative_dist = pairwise_dist.max(1, True)[0]
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative.float())
        hardest_negative_dist = anchor_negative_dist.min(1, True)[0]


        # loss = F.relu(torch.logsumexp(self.gamma*(hardest_negative_dist-hardest_positive_dist+self.margin), dim=0))
        # return loss


# Circle Loss
        sp = anchor_positive_dist
        sn = anchor_negative_dist
        ap = torch.clamp_min(-anchor_positive_dist + 1 + self.margin, min=0.)
        an = torch.clamp_min(anchor_negative_dist + self.margin, min=0.)

        delta_p = 1 - self.margin
        delta_n = self.margin

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))
        # print(loss.shape)
        return loss.mean()
# Circle Loss End 
        # loss = torch.log(1+torch.sum(torch.exp(self.gamma*(hardest_negative_dist - hardest_positive_dist + self.margin))))
        # return loss

# Original Triplet Loss: Uncomment next two lines for Triplet Loss '''
        # loss = (F.relu(hardest_positive_dist - hardest_negative_dist + self.margin))
        # return loss.mean()


class MPerClassSampler(Sampler):
    def __init__(self, labels, batch_size, m=4):
        self.labels = np.array(labels)
        self.labels_unique = np.unique(labels)
        self.batch_size = batch_size
        self.m = m
        assert batch_size % m == 0, 'batch size must be divided by m'

    def __len__(self):
        return len(self.labels) // self.batch_size

    def __iter__(self):
        for _ in range(self.__len__()):
            labels_in_batch = set()
            inds = np.array([], dtype=np.int)

            while inds.shape[0] < self.batch_size:
                sample_label = np.random.choice(self.labels_unique)
                if sample_label in labels_in_batch:
                    continue

                labels_in_batch.add(sample_label)
                sample_label_ids = np.argwhere(np.in1d(self.labels, sample_label)).reshape(-1)
                subsample = np.random.permutation(sample_label_ids)[:self.m]
                inds = np.append(inds, subsample)

            inds = inds[:self.batch_size]
            inds = np.random.permutation(inds)
            yield list(inds)
