import json
import logging
import os
import random
from typing import Tuple, List, Union

import numpy as np
import pandas as pd
import torch
from pytorch_pretrained_bert import BertTokenizer, BertAdam
from pytorch_pretrained_bert.file_utils import WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
from pytorch_pretrained_bert.optimization import WarmupLinearSchedule
from sklearn.model_selection import KFold
from torch import nn
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler)
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm


class HyperParameters(object):
    """
    用于管理模型超参数
    """

    def __init__(self, max_length: int = 128, epochs=4, batch_size=32, learning_rate=2e-5, fp16=True) -> None:
        self.max_length = max_length
        """句子的最大长度"""
        self.epochs = epochs
        """训练迭代轮数"""
        self.batch_size = batch_size
        """每个batch的样本数量"""
        self.learning_rate = learning_rate
        """学习率"""
        self.fp16 = fp16
        """是否使用fp16混合精度训练"""

    def __repr__(self) -> str:
        return self.__dict__.__repr__()


class BertForSimMatchModel(BertPreTrainedModel):
    """
    ab、ac交互并编码
    """

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(p=0.1)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)
        self.apply(self.init_bert_weights)

    def forward(self, ab, ac, labels=None):
        _, ab_pooled_output = self.bert(*ab, output_all_encoded_layers=False)
        _, ac_pooled_output = self.bert(*ac, output_all_encoded_layers=False)
        subtraction_output = ab_pooled_output - ac_pooled_output
        concated_pooled_output = self.dropout(subtraction_output)
        output = self.seq_relationship(concated_pooled_output)

        prob = torch.nn.functional.softmax(Variable(output), dim=1)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(output.view(-1, 2), labels.view(-1))
        return output, prob, loss


class TripletTextDataset(Dataset):

    def __init__(self, text_a_list, text_b_list, text_c_list, label_list=None):
        if label_list is None or len(label_list) == 0:
            label_list = [None] * len(text_a_list)
        assert all(len(label_list) == len(text_list) for text_list in [text_a_list, text_b_list, text_c_list])
        self.text_a_list = text_a_list
        self.text_b_list = text_b_list
        self.text_c_list = text_c_list
        self.label_list = [0 if label == 'B' else 1 for label in label_list]

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        text_a, text_b, text_c, label = self.text_a_list[index], self.text_b_list[index], self.text_c_list[index], \
                                        self.label_list[index]
        return text_a, text_b, text_c, label

    @classmethod
    def from_dataframe(cls, df):
        text_a_list = df['A'].tolist()
        text_b_list = df['B'].tolist()
        text_c_list = df['C'].tolist()
        if 'label' not in df:
            df['label'] = 'B'
        label_list = df['label'].tolist()
        return cls(text_a_list, text_b_list, text_c_list, label_list)

    @classmethod
    def from_dict_list(cls, data, use_augment=False):
        df = pd.DataFrame(data)
        if 'label' not in df:
            df['label'] = 'B'
        if use_augment:
            df = TripletTextDataset.augment(df)
        return cls.from_dataframe(df)

    @classmethod
    def from_jsons(cls, json_lines_file, use_augment=False):
        with open(json_lines_file, encoding='utf-8') as f:
            data = list(map(lambda line: json.loads(line), f))
        return cls.from_dict_list(data, use_augment)

    @staticmethod
    def augment(df):
        df_cp1 = df.copy()
        df_cp1['B'] = df['C']
        df_cp1['C'] = df['B']
        df_cp1['label'] = 'C'

        df_cp2 = df.copy()
        df_cp2['A'] = df['B']
        df_cp2['B'] = df['A']
        df_cp2['label'] = 'B'

        df_cp3 = df.copy()
        df_cp3['A'] = df['B']
        df_cp3['B'] = df['C']
        df_cp3['C'] = df['A']
        df_cp3['label'] = 'C'

        df_cp4 = df.copy()
        df_cp4['A'] = df['C']
        df_cp4['B'] = df['A']
        df_cp4['C'] = df['C']
        df_cp4['label'] = 'C'

        df_cp5 = df.copy()
        df_cp5['A'] = df['C']
        df_cp5['B'] = df['C']
        df_cp5['C'] = df['A']
        df_cp5['label'] = 'B'

        df = pd.concat([df, df_cp1, df_cp2, df_cp3, df_cp4, df_cp5])
        df = df.drop_duplicates()
        df = df.sample(frac=1)

        return df


def get_collator(max_len, device, tokenizer, model_class):
    def two_pair_collate_fn(batch):
        """
        获取一个mini batch的数据，将文本三元组转化成tensor。

        将ab、ac分别拼接，编码tensor

        :param batch:
        :return:
        """
        example_tensors = []
        for text_a, text_b, text_c, label in batch:
            input_example = InputExample(text_a, text_b, text_c, label)
            ab_feature, ac_feature = input_example.to_two_pair_feature(tokenizer, max_len)
            ab_tensor, ac_tensor = ab_feature.to_tensor(device), ac_feature.to_tensor(device)
            label_tensor = torch.LongTensor([label]).to(device)
            example_tensors.append((ab_tensor, ac_tensor, label_tensor))

        return default_collate(example_tensors)

    if model_class == BertForSimMatchModel:
        return two_pair_collate_fn


algorithm_map = {
    "BertForSimMatchModel": BertForSimMatchModel
}


class BertSimMatchModel(object):
    """
    基于 Bert 实现的案件相似匹配模型
    """

    def __init__(self, model, tokenizer, max_length, algorithm, device: torch.device = None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.model.to(self.device)
        self.model.eval()
        self.algorithm = algorithm
        self.model_class = algorithm_map[self.algorithm]
        self.predict_batch_size = 8

    def save(self, model_dir):
        """
        存储模型

        :param model_dir:
        :return:
        """
        # Save a trained model, configuration and tokenizer
        model_to_save = self.model.module if hasattr(self.model,
                                                     'module') else self.model  # Only save the model it-self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(model_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(model_dir, CONFIG_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        self.tokenizer.save_vocabulary(model_dir)

        with open(os.path.join(model_dir, 'param.json'), mode='w') as f:
            json.dump({'max_len': self.max_length, "algorithm": self.algorithm}, f)

    @classmethod
    def load(cls, model_dir, device=None):
        """
        加载模型。通过模型文件构造实例

        :param model_dir:
        :param device:
        :return:
        """
        with open(os.path.join(model_dir, 'param.json')) as f:
            param_dict = json.load(f)
            max_length = param_dict['max_len']
            algorithm = param_dict['algorithm']

        tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=False)
        model_class = algorithm_map[algorithm]
        model = model_class.from_pretrained(model_dir)
        return cls(model, tokenizer, max_length, algorithm, device)

    def predict(self, text_tuples: Union[List[Tuple[str, str, str]], TripletTextDataset]) -> List[Tuple[str, float]]:
        if isinstance(text_tuples, Dataset):
            data = text_tuples
        else:
            text_a_list, text_b_list, text_c_list = [list(i) for i in zip(*text_tuples)]

            data = TripletTextDataset(text_a_list, text_b_list, text_c_list, None)
        sampler = SequentialSampler(data)
        collate_fn = get_collator(self.max_length, self.device, self.tokenizer, self.model_class)
        dataloader = DataLoader(data, sampler=sampler, batch_size=8, collate_fn=collate_fn)

        final_results = []

        for batch in dataloader:
            with torch.no_grad():
                predict_results = self.model(*batch)[1].cpu().numpy()
                cata_indexes = np.argmax(predict_results, axis=1)

                for i_sample, cata_index in enumerate(cata_indexes):
                    prob = predict_results[i_sample][cata_index]
                    label = 'B' if cata_index == 0 else 'C'
                    final_results.append((str(label), float(prob)))

        return final_results


class BertModelTrainer(object):
    def __init__(self, dataset_path, bert_model_dir, param: HyperParameters, algorithm,
                 test_input_path, test_ground_truth_path) -> None:
        """

        :param dataset_path: 数据集路径。 默认当作是训练集，但当train函数采用了kfold参数时，将对该数据集进行划分并做交叉验证
        :param bert_model_dir: 预训练 bert 模型路径
        :param param: 超参数
        :param algorithm: 选择算法，默认 BertForSimMatchModel
        :param test_input_path: 固定的测试集的路径，用于快速测试模型性能
        :param test_ground_truth_path: 固定的测试集的标记
        """
        self.dataset_path = dataset_path
        self.bert_model_dir = bert_model_dir
        self.param = param
        self.test_input_path = test_input_path
        self.test_ground_truth_path = test_ground_truth_path
        self.algorithm = algorithm
        self.model_class = algorithm_map[self.algorithm]
        logger.info('算法:' + algorithm)

    def load_dataset(self, n_splits: int = 1) \
            -> List[Tuple[TripletTextDataset, TripletTextDataset, List[str]]]:
        """
        划分k折交叉验证数据集用于cv

        :param n_splits:
        :return: List[(train_data, test_data, test_labels_list)]
        """

        data = []

        if n_splits == 1:
            train_data = TripletTextDataset.from_jsons(self.dataset_path, use_augment=True)
            test_data = TripletTextDataset.from_jsons(self.test_input_path)
            with open(self.test_ground_truth_path) as f:
                test_label_list = [line.strip() for line in f.readlines()]

            data.append((train_data, test_data, test_label_list))
            return data

        raw_data_list = []
        with open(self.dataset_path, encoding='utf-8') as raw_input:
            for line in raw_input:
                raw_data_list.append(json.loads(line.strip(), encoding='utf-8'))

        kf = KFold(n_splits, shuffle=True, random_state=42)
        random.seed(42)
        for train_index, test_index in kf.split(raw_data_list):
            # 准备训练集
            train_data_list = [raw_data_list[i] for i in train_index]
            train_data = TripletTextDataset.from_dict_list(train_data_list, use_augment=True)

            # 准备测试集，打乱BC顺序
            test_data_list = [raw_data_list[i] for i in test_index]
            shuffled_test_data_list = []
            test_label_list = []
            for item in test_data_list:
                a = item['A']
                b = item['B']
                c = item['C']

                choice = int(random.getrandbits(1))
                label = 'B' if choice == 0 else 'C'
                if label == 'C':
                    item = {'A': a, 'B': c, 'C': b}

                shuffled_test_data_list.append(item)
                test_label_list.append(label)

            test_data = TripletTextDataset.from_dict_list(shuffled_test_data_list)

            data.append((train_data, test_data, test_label_list))
        return data

    def train(self, model_dir, kfold=1):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
        # n_gpu = 1
        # torch.cuda.set_device(1)
        logger.info("***** Running training *****")
        logger.info("dataset: {}".format(self.dataset_path))
        logger.info("k-fold number: {}".format(kfold))
        logger.info("device: {} n_gpu: {}".format(device, n_gpu))
        logger.info("config: {}".format(json.dumps(self.param.__dict__, indent=4, sort_keys=True)))

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(42)

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        tokenizer = BertTokenizer.from_pretrained(self.bert_model_dir, do_lower_case=True)
        data = self.load_dataset(kfold)

        all_acc_list = []
        for k, (train_data, test_data, test_label_list) in enumerate(data, start=1):
            one_fold_acc_list = []
            bert_model = self.model_class.from_pretrained(self.bert_model_dir)
            if self.param.fp16:
                bert_model.half()
            bert_model.to(device)
            if n_gpu > 1:
                bert_model = torch.nn.DataParallel(bert_model)

            num_train_optimization_steps = int(len(train_data) / self.param.batch_size) * self.param.epochs

            param_optimizer = list(bert_model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

            if self.param.fp16:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam

                optimizer = FusedAdam(optimizer_grouped_parameters,
                                      lr=self.param.learning_rate,
                                      bias_correction=False,
                                      max_grad_norm=1.0)

                loss_scale = 0
                if loss_scale == 0:
                    optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
                else:
                    optimizer = FP16_Optimizer(optimizer, static_loss_scale=loss_scale)
                warmup_linear = WarmupLinearSchedule(warmup=0.1,
                                                     t_total=num_train_optimization_steps)
            else:
                optimizer = BertAdam(optimizer_grouped_parameters,
                                     lr=self.param.learning_rate,
                                     warmup=0.1,
                                     t_total=num_train_optimization_steps)
            global_step = 0

            logger.info("***** fold {}/{} *****".format(k, kfold))
            logger.info("  Num examples = %d", len(train_data))
            logger.info("  Batch size = %d", self.param.batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps)

            train_sampler = RandomSampler(train_data)

            collate_fn = get_collator(self.param.max_length, device, tokenizer, self.model_class)

            train_dataloader = DataLoader(dataset=train_data, sampler=train_sampler, batch_size=self.param.batch_size,
                                          shuffle=False, num_workers=0, collate_fn=collate_fn, drop_last=True)
            bert_model.train()
            for epoch in range(int(self.param.epochs)):
                tr_loss = 0
                steps = tqdm(train_dataloader)
                for step, batch in enumerate(steps):
                    # if step % 200 == 0:
                    #     model = BertSimMatchModel(bert_model, tokenizer, self.param.max_length, self.algorithm)
                    #     acc, loss = self.evaluate(model, test_data, test_label_list)
                    #     logger.info(
                    #         "Epoch {}, step {}/{}, train Loss: {:.7f}, eval acc: {}, eval loss: {:.7f}".format(
                    #             epoch + 1, step, num_train_optimization_steps, tr_loss, acc, loss))
                    #     bert_model.train()

                    # define a new function to compute loss values for both output_modes
                    loss = bert_model(*batch)[2]

                    if n_gpu > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu.

                    if self.param.fp16:
                        optimizer.backward(loss)

                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = self.param.learning_rate * warmup_linear.get_lr(global_step, 0.1)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    else:
                        loss.backward()

                    tr_loss += loss.item()
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                    steps.set_description(
                        "Epoch {}/{}, Loss {:.7f}".format(epoch + 1, self.param.epochs,
                                                          loss.item()))

                model = BertSimMatchModel(bert_model, tokenizer, self.param.max_length, self.algorithm)
                acc, loss = self.evaluate(model, test_data, test_label_list)
                one_fold_acc_list.append(acc)
                logger.info(
                    "Epoch {}, train Loss: {:.7f}, eval acc: {}, eval loss: {:.7f}".format(
                        epoch + 1, tr_loss, acc, loss))
                bert_model.train()
            all_acc_list.append(one_fold_acc_list)
            model = BertSimMatchModel(bert_model, tokenizer, self.param.max_length, self.algorithm)
            model.save(model_dir)

        logger.info("***** Stats *****")
        # 计算kfold的平均的acc
        all_epoch_acc = list(zip(*all_acc_list))
        logger.info("acc for each epoch:")
        for epoch, acc in enumerate(all_epoch_acc, start=1):
            logger.info("epoch %d, mean: %.5f, std: %.5f" % (epoch, float(np.mean(acc)), float(np.std(acc))))

        logger.info("***** Training complete *****")

    @staticmethod
    def evaluate(model: BertSimMatchModel, data: TripletTextDataset, real_label_list: List[str]):
        """
        评估模型，计算acc

        :param model:
        :param data:
        :param real_label_list:
        :return:
        """
        num_padding = 0
        if isinstance(model.model, torch.nn.DataParallel):
            num_padding = model.predict_batch_size - len(data) % model.predict_batch_size
            if num_padding != 0:
                padding_data = TripletTextDataset(text_a_list=[''] * num_padding,
                                                  text_b_list=[''] * num_padding,
                                                  text_c_list=[''] * num_padding)
                data = data.__add__(padding_data)

        sampler = SequentialSampler(data)
        collate_fn = get_collator(model.max_length, model.device, model.tokenizer, model.model_class)
        dataloader = DataLoader(data, sampler=sampler, batch_size=8, collate_fn=collate_fn)

        predict_result = []
        loss_sum = 0
        for batch in dataloader:
            with torch.no_grad():
                output = model.model(*batch)
                predict_results = output[1].cpu().numpy()
                loss = output[2].mean().cpu().item()
                loss_sum += loss
                cata_indexes = np.argmax(predict_results, axis=1)

                for i_sample, cata_index in enumerate(cata_indexes):
                    prob = predict_results[i_sample][cata_index]
                    label = 'B' if cata_index == 0 else 'C'
                    predict_result.append((str(label), float(prob)))

        if num_padding != 0:
            predict_result = predict_result[:-num_padding]
        assert len(predict_result) == len(real_label_list)

        correct = 0
        for i, real_label in enumerate(real_label_list):
            try:
                predict_label = predict_result[i][0]
                if predict_label == real_label:
                    correct += 1
            except Exception as e:
                print(e)
                continue

        acc = correct / len(real_label_list)
        return acc, loss_sum


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids

    def to_tensor(self, device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.LongTensor(self.input_ids).to(device), \
               torch.LongTensor(self.segment_ids).to(device), \
               torch.LongTensor(self.input_mask).to(device)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, text_a, text_b=None, text_c=None, label=None):
        """Constructs a InputExample.

        Args:
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.label = label

    @staticmethod
    def _text_pair_to_feature(text_a, text_b, tokenizer, max_seq_length):
        tokens_a = tokenizer.tokenize(text_a)
        tokens_b = None

        if text_b:
            tokens_b = tokenizer.tokenize(text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        return input_ids, input_mask, segment_ids

    def to_two_pair_feature(self, tokenizer, max_seq_length) -> Tuple[InputFeatures, InputFeatures]:
        ab = self._text_pair_to_feature(self.text_a, self.text_b, tokenizer, max_seq_length)
        ac = self._text_pair_to_feature(self.text_a, self.text_c, tokenizer, max_seq_length)
        ab, ac = InputFeatures(*ab), InputFeatures(*ac)
        return ab, ac


def _truncate_seq_pair(tokens_a: list, tokens_b: list, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop(0)
        else:
            tokens_b.pop(0)


logger = logging.getLogger('train model')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

MODEL_DIR = 'model'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)
fh = logging.FileHandler(os.path.join(MODEL_DIR, 'train.log'), encoding='utf-8')
fh.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)

if __name__ == '__main__':
    BERT_PRETRAINED_MODEL = '/bert/pytorch_chinese_L-12_H-768_A-12'

    # TRAINING_DATASET = 'data/train/input.txt'  # for quick dev
    TRAINING_DATASET = 'data/raw/CAIL2019-SCM-big/SCM_5k.json'

    test_input_path = 'data/test/input.txt'
    test_ground_truth_path = 'data/test/ground_truth.txt'

    config = {
        "max_length": 512,
        "epochs": 2,
        "batch_size": 12,
        "learning_rate": 2e-5,
        "fp16": True
    }
    hyper_parameter = HyperParameters()
    hyper_parameter.__dict__ = config
    algorithm = 'BertForSimMatchModel'

    trainer = BertModelTrainer(TRAINING_DATASET, BERT_PRETRAINED_MODEL, hyper_parameter, algorithm, test_input_path,
                               test_ground_truth_path)
    trainer.train(MODEL_DIR, 1)
