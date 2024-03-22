#proposed sampler for cross version
# 实现假设每个类别的每个版本中至少有m个样本，或者允许重复采样以达到这个数量


import numpy as np
import collections
from torch.utils.data.sampler import Sampler
from pytorch_metric_learning.utils  import common_functions as c_f
class MPerClassVersionSampler(Sampler):
    #  类的构造函数，用于初始化采样器的状态
    def __init__(self, labels, versions, m, batch_size=None, length_before_new_iter=100000):
        self
        self.labels = np.array(labels)
        self.versions = np.array(versions)
        self.version_type_num = len(np.unique(versions)) # 版本的种类数

        self.labels_to_indices = c_f.get_labels_to_indices(labels)
        self.labels_idx = list(self.labels_to_indices.keys())#每个类别对应的索引列表


        self.sample_per_class_per_version = int(m)
        self.batch_size = int(batch_size) if batch_size is not None else None
        self.indices = self._build_indices()#indices里面的类别和版本索引是正确的
        self.unique_labels = np.unique(self.labels)

        self.sample_per_class = self.sample_per_class_per_version * self.version_type_num # 每个类别中采样的样本数量

        #表示在不考虑批次大小限制的情况下，单次遍历所有选中标签和版本所需处理的样本总数。
        self.length_of_single_pass = self.sample_per_class_per_version * self.version_type_num * len(self.labels_idx)

        # print('self.length_of_single_pass: ', self.length_of_single_pass)
        self.list_size = length_before_new_iter
        self._adjust_list_size()
        # print('self.list_size init: ', self.list_size)


    def _build_indices(self):
        # 创建一个嵌套的字典结构，用于组织数据索引。外层键是标签，内层键是版本，值是对应的索引列表。
        indices = collections.defaultdict(lambda: collections.defaultdict(list))
        for idx, (label, version) in enumerate(zip(self.labels, self.versions)):
            indices[label][version].append(idx)
        return indices

    def _adjust_list_size(self):
        #检查批次大小是否未指定：
        if self.batch_size is None:
            #如果self.length_of_single_pass小于self.list_size，说明单次遍历的长度不足以覆盖整个列表。为了解决这个问题，代码通过减少self.list_size来使其能够被self.length_of_single_pass整除，
            if self.length_of_single_pass < self.list_size:
                self.list_size -= (self.list_size) % (self.length_of_single_pass)
        else:
            print('batch_size is specified, asserting and adjusting list_size')
            #第一个断言assert self.list_size >= self.batch_size确保整个列表足够大，至少能够容纳一个批次的数据。
            assert self.list_size >= self.batch_size
            # 第二个断言assert (self.length_of_single_pass >= self.batch_size)确保单次遍历的长度至少等于一个批次的大小，这是为了保证在每次遍历中都有足够的数据可供处理。
            assert (
                self.length_of_single_pass >= self.batch_size
            ), "m * (number of unique labels) must be >= batch_size"

            # 第三个断言assert (self.batch_size % self.sample_per_class_per_version) == 0确保每个批次的大小可以被self.sample_per_class_per_version整除，这可能是为了确保每个批次中可以均匀地包含每个类别的样本，从而保持数据的均衡性。这里的self.sample_per_class_per_version可能表示每个类别在每个版本中的样本数量。
            assert (
                self.batch_size % self.sample_per_class_per_version
            ) == 0, "self.sample_per_class_per_version must divide batch_size without any remainder"

            # 最后，self.list_size -= self.list_size % self.batch_size调整self.list_size使其能被self.batch_size整除。这与未指定批次大小时的逻辑相似，目的是确保列表大小与批次大小兼容，避免在数据处理时出现问题。
            # print('self.batch_size: ', self.batch_size)
            # print('self.list_size before iter: ', self.list_size)
            # self.list_size -= self.list_size % self.batch_size
            # 等价于
            remainder = self.list_size % self.batch_size
            self.list_size = self.list_size - remainder

        # print('self.list_size after iter: ', self.list_size)
        
       

    def __len__(self):
        return self.list_size // (self.batch_size if self.batch_size is not None else 1)

    def __iter__(self):
        # 生成采样索引的迭代器。它首先打乱标签，然后根据batch_size和m_per_class的值选择相应数量的标签。对于每个选中的标签和版本，它会根据是否需要重复采样来选择m_per_class个样本。最后，它会将这些索引打乱并返回迭代器。
        idx_list = []
        num_iters = self.calculate_num_iters()
        version_samples_info = collections.defaultdict(lambda: collections.defaultdict(list))
        i= 0
        for _ in range(num_iters):
            i += 1
            # print(f"Iteration {i}/{num_iters}")
            c_f.NUMPY_RANDOM.shuffle(self.unique_labels)
            curr_label_set = self.unique_labels[: self.batch_size // self.sample_per_class_per_version] if self.batch_size else self.unique_labels


            for label in curr_label_set:
                for version, idxs in self.indices[label].items():
                    sampled_indices = np.random.choice(idxs, self.sample_per_class_per_version, replace=len(idxs) < self.sample_per_class_per_version)
                    idx_list.extend(sampled_indices)
                    version_samples_info[label][version].extend(sampled_indices)
            if self.batch_size and len(idx_list) > self.batch_size:
                break  # Adjust if more samples are collected than batch_size
        # print('idx_list: ', idx_list)
        # np.random.shuffle(idx_list)
        # print('len(idx_list): ', len(idx_list))
        # print('self.list_size: ', self.list_size)
        # print('len(idx_list[:self.list_size]): ', iter(idx_list[:self.list_size]))
        # 打印每个标签的每个版本所包含的样本索引和数量
        # for label, versions in version_samples_info.items():
        #     print(f"Label {label}:")
        #     for version, indices in versions.items():
        #         print(f"  Version {version:.1f}: {len(indices)} samples, Indices: {indices}")
                
                
        return iter(idx_list[:self.list_size])
    
    def calculate_num_iters(self):
        # 这部分代码决定了计算迭代次数时应该使用的除数。如果self.batch_size为None，意味着没有指定批次大小，则使用self.length_of_single_pass作为除数；如果指定了批次大小，则使用self.batch_size作为除数。
        # self.batch_size 作为除数意味着你想要将整个数据集分割成多个批次，每个批次包含 self.batch_size 个样本。
        divisor = (
            self.length_of_single_pass if self.batch_size is None else self.batch_size
        )
        num_iters = self.list_size // divisor if divisor < self.list_size else 1
        return num_iters



import torch
from torch.utils.data.sampler import Sampler




class MPerClassSampler(Sampler):
    """
    At every iteration, this will return m samples per class. For example,
    if dataloader's batchsize is 100, and m = 5, then 20 classes with 5 samples
    each will be returned
    """

    def __init__(self, labels, m, batch_size=None, length_before_new_iter=100000):
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        self.m_per_class = int(m)
        self.batch_size = int(batch_size) if batch_size is not None else batch_size
        self.labels_to_indices = c_f.get_labels_to_indices(labels)
        self.labels = list(self.labels_to_indices.keys())
        print('labels length in MPC: ', len(self.labels))
        print('self.m_per_class: ', self.m_per_class)
        self.length_of_single_pass = self.m_per_class * len(self.labels)
        print('self.length_of_single_pass: ', self.length_of_single_pass)
        self.list_size = length_before_new_iter
        print('self.list_size before iter: ', self.list_size)

        
        if self.batch_size is None:
            if self.length_of_single_pass < self.list_size:
                self.list_size -= (self.list_size) % (self.length_of_single_pass)
        else:
            assert self.list_size >= self.batch_size
            assert (
                self.length_of_single_pass >= self.batch_size
            ), "m * (number of unique labels) must be >= batch_size"
            assert (
                self.batch_size % self.m_per_class
            ) == 0, "m_per_class must divide batch_size without any remainder"
            # self.list_size -= self.list_size % self.batch_size
            remainder = self.list_size % self.batch_size
            self.list_size = self.list_size - remainder
        print('self.list_size after iter: ', self.list_size)
    def __len__(self):
        return self.list_size

    def __iter__(self):
        idx_list = [0] * self.list_size
        i = 0
        j = 0
        num_iters = self.calculate_num_iters()
        for _ in range(num_iters):
            j += 1
            print(f"Iteration {j}/{num_iters}")

            c_f.NUMPY_RANDOM.shuffle(self.labels)
            if self.batch_size is None:
                curr_label_set = self.labels
            else:
                curr_label_set = self.labels[: self.batch_size // self.m_per_class]
            for label in curr_label_set:
                t = self.labels_to_indices[label]
                idx_list[i : i + self.m_per_class] = c_f.safe_random_choice(
                    t, size=self.m_per_class
                )
                i += self.m_per_class
        return iter(idx_list)

    def calculate_num_iters(self):
        divisor = (
            self.length_of_single_pass if self.batch_size is None else self.batch_size
        )
        return self.list_size // divisor if divisor < self.list_size else 1




# import numpy as np
# import collections
# from torch.utils.data.sampler import Sampler
# from pytorch_metric_learning.utils  import common_functions as c_f
# class MPerClassVersionSampler(Sampler):
#     def __init__(self, labels, versions, m, batch_size=None, length_before_new_iter=100000):
#         self.labels = np.array(labels)
#         self.versions = np.array(versions)
#         self.m_per_class = int(m)
#         self.batch_size = int(batch_size) if batch_size is not None else None
#         self.indices = self._build_indices()
#         self.unique_labels = np.unique(self.labels)
#         self.list_size = length_before_new_iter
#         self._adjust_list_size()

#     def _build_indices(self):
#         indices = collections.defaultdict(lambda: collections.defaultdict(list))
#         for idx, (label, version) in enumerate(zip(self.labels, self.versions)):
#             indices[label][version].append(idx)
#         return indices

#     def _adjust_list_size(self):
#         self.length_of_single_pass = sum(
#             self.m_per_class * len(versions) for label, versions in self.indices.items()
#         )
#         if self.batch_size is not None:
#             assert self.batch_size % self.m_per_class == 0, "m_per_class must divide batch_size without any remainder"
#             self.list_size = (self.list_size // self.batch_size) * self.batch_size
        
#         print(f"Adjusted list size: {self.list_size}")

#     def __len__(self):
#         return self.list_size // (self.batch_size if self.batch_size is not None else 1)

#     def __iter__(self):
#         idx_list = []
#         for _ in range(self.__len__()):
#             c_f.NUMPY_RANDOM.shuffle(self.unique_labels)
#             curr_label_set = self.unique_labels[: self.batch_size // self.m_per_class] if self.batch_size else self.unique_labels
#             for label in curr_label_set:
#                 for version, idxs in self.indices[label].items():
#                     sampled_indices = np.random.choice(idxs, self.m_per_class, replace=len(idxs) < self.m_per_class)
#                     idx_list.extend(sampled_indices)
#             if self.batch_size and len(idx_list) > self.batch_size:
#                 break  # Adjust if more samples are collected than batch_size
#         np.random.shuffle(idx_list)
#         return iter(idx_list[:self.list_size])











# from torch.utils.data.sampler import Sampler


# import collections
# import numpy as np
# import torch

# class MPerClassVersionSampler(Sampler):
#     def __init__(self, labels, versions, m, batch_size=None):
#         self.labels = np.array(labels)
#         self.versions = np.array(versions)
#         self.m = m
#         self.batch_size = batch_size
#         self.indices = self._build_indices()
#         self.unique_labels = np.unique(self.labels)

#     def _build_indices(self):
#         """构建一个嵌套字典，组织样本索引，键是类别，内层键是版本。"""
#         indices = collections.defaultdict(lambda: collections.defaultdict(list))
#         for idx, (label, version) in enumerate(zip(self.labels, self.versions)):
#             indices[label][version].append(idx)
#         return indices

#     def _sample_from_category(self, category_indices, m):
#         """从每个版本中采样m个样本，如果某个版本样本不足m个，则重复采样直到达到m个。"""
#         sampled_indices = []
#         for version, idxs in category_indices.items():
#             if len(idxs) >= m:
#                 sampled_indices.extend(np.random.choice(idxs, m, replace=False))
#             else:
#                 sampled_indices.extend(np.random.choice(idxs, m, replace=True))
#         return sampled_indices

#     def __iter__(self):
#         batch_indices = []
#         for label in self.unique_labels:
#             category_indices = self.indices[label]
#             sampled_indices = self._sample_from_category(category_indices, self.m)
#             batch_indices.extend(sampled_indices)

#         np.random.shuffle(batch_indices)
#         batch_indices = batch_indices[:self.batch_size] if self.batch_size else batch_indices
#         return iter(batch_indices)

#     def __len__(self):
#         if self.batch_size:
#             return self.batch_size
#         else:
#             return sum(self.m * len(versions) for versions in self.indices.values())

