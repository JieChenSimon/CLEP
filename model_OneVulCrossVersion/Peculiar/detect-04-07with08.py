from __future__ import absolute_import, division, print_function
import sys
import logging.config
from configparser import ConfigParser
from tree_sitter import Language, Parser
from parser import (remove_comments_and_docstrings,
                    tree_to_token_index,
                    index_to_code_token,
                    tree_to_variable_index)
from parser import DFG_solidity

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
from tqdm import tqdm, trange
import multiprocessing
from model import Model

cpu_cont = 16


dfg_function = {
    'solidity': DFG_solidity
}

# load parsers
parsers = {}
for lang in dfg_function:
    LANGUAGE = Language('/home/chen/workspace/codeproject/CL4acrossVersionSC/model_OneVulCrossVersion/Peculiar/parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    parser = [parser, dfg_function[lang]]
    parsers[lang] = parser


# initialize logger
logging.config.fileConfig("/home/chen/workspace/codeproject/CL4acrossVersionSC/model_OneVulCrossVersion/Peculiar/logging_04-07with08.cfg")
logger = logging.getLogger("root")


class RuntimeContext(object):
    """ runtime enviroment
    """

    def __init__(self):
        """ initialization
        """
        # configuration initialization
        config_parser = ConfigParser()
        config_file = self.get_config_file_name()
        config_parser.read(config_file, encoding="UTF-8")
        sections = config_parser.sections()

        file_section = sections[0]
        self.train_data_file_04_07 = config_parser.get(file_section, "train_data_file_04_07")
        self.eval_data_file_04_07 = config_parser.get(file_section, "eval_data_file_04_07")
        self.test_data_file_04_07 = config_parser.get(file_section, "test_data_file_04_07")

        self.train_data_file_08 = config_parser.get(file_section, "train_data_file_08")
        self.eval_data_file_08 = config_parser.get(file_section, "eval_data_file_08")
        self.test_data_file_08 = config_parser.get(file_section, "test_data_file_08")


        self.output_dir = config_parser.get(file_section, "output_dir")

        base_section = sections[1]
        self.model_name_or_path = config_parser.get(
            base_section, "model_name_or_path")
        self.config_name = config_parser.get(base_section, "config_name")
        self.tokenizer_name = config_parser.get(base_section, "tokenizer_name")

        parameters_section = sections[2]
        self.code_length = int(config_parser.get(
            parameters_section, "code_length"))
        self.data_flow_length = int(config_parser.get(
            parameters_section, "data_flow_length"))
        self.train_batch_size = int(config_parser.get(
            parameters_section, "train_batch_size"))
        self.eval_batch_size = int(config_parser.get(
            parameters_section, "eval_batch_size"))
        self.gradient_accumulation_steps = int(config_parser.get(
            parameters_section, "gradient_accumulation_steps"))
        self.learning_rate = float(config_parser.get(
            parameters_section, "learning_rate"))
        self.weight_decay = float(config_parser.get(
            parameters_section, "weight_decay"))
        self.adam_epsilon = float(config_parser.get(
            parameters_section, "adam_epsilon"))
        self.max_grad_norm = float(config_parser.get(
            parameters_section, "max_grad_norm"))
        self.max_steps = int(config_parser.get(
            parameters_section, "max_steps"))
        self.warmup_steps = int(config_parser.get(
            parameters_section, "warmup_steps"))
        self.seed = int(config_parser.get(parameters_section, "seed"))
        self.epochs = int(config_parser.get(parameters_section, "epochs"))
        self.n_gpu = torch.cuda.device_count()
        self.sample_size_for_other_versions = int(config_parser.get(parameters_section, "sample_size_for_other_versions"))
        # device_ids = [0, 1, 2]  # 例如，使用GPU 0 和 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cuda:" + str(device_ids[0]) if torch.cuda.is_available() else "cpu")
        self.device = device

    def get_config_file_name(self):
        """ get the configuration file name according to the command line parameters
        """
        argv = sys.argv
        config_type = "dev"  # default configuration type
        if None != argv and len(argv) > 1:
            config_type = argv[1]
        # config_file = config_type + ".cfg"
        config_file = "/home/chen/workspace/codeproject/CL4acrossVersionSC/model_OneVulCrossVersion/Peculiar/dev_04-07with08.cfg"
        logger.info("get_config_file_name() return : " + config_file)
        return config_file


# remove comments, tokenize code and extract dataflow
def extract_dataflow(code, parser, lang):
    # remove comments
    try:
        code = remove_comments_and_docstrings(code, lang)
    except:
        pass
    # obtain dataflow
    if lang == "php":
        code = "<?php"+code+"?>"
    try:
        tree = parser[0].parse(bytes(code, 'utf8'))
        root_node = tree.root_node
        tokens_index = tree_to_token_index(root_node)
        code = code.split('\n')
        code_tokens = [index_to_code_token(x, code) for x in tokens_index]
        index_to_code = {}
        for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
            index_to_code[index] = (idx, code)
        try:
            DFG, _ = parser[1](root_node, index_to_code, {})
        except:
            DFG = []
        DFG = sorted(DFG, key=lambda x: x[1])
        # identify critical node in DFG
        critical_idx = []
        for id, e in enumerate(DFG):
            if e[0] == "call" and DFG[id+1][0] == "value":
                critical_idx.append(DFG[id-1][1])
                critical_idx.append(DFG[id+2][1])
        lines = []
        for index, code in index_to_code.items():
            if code[0] in critical_idx:
                line = index[0][0]
                lines.append(line)
        lines = list(set(lines))
        for index, code in index_to_code.items():
            if index[0][0] in lines:
                critical_idx.append(code[0])
        critical_idx = list(set(critical_idx))
        max_nums = 0
        cur_nums = -1
        while cur_nums != max_nums and cur_nums != 0:
            max_nums = len(critical_idx)
            for id, e in enumerate(DFG):
                if e[1] in critical_idx:
                    critical_idx += e[-1]
                for i in e[-1]:
                    if i in critical_idx:
                        critical_idx.append(e[1])
                        break
            critical_idx = list(set(critical_idx))
            cur_nums = len(critical_idx)
        dfg = []
        for id, e in enumerate(DFG):
            if e[1] in critical_idx:
                dfg.append(e)
        dfg = sorted(dfg, key=lambda x: x[1])

        # Removing independent points
        indexs = set()
        for d in dfg:
            if len(d[-1]) != 0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG = []
        for d in dfg:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg = new_DFG
    except:
        dfg = []
    return code_tokens, dfg


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 input_tokens_1,
                 input_ids_1,
                 position_idx_1,
                 dfg_to_code_1,
                 dfg_to_dfg_1,
                 label,
                 url1

                 ):
        # The code function
        self.input_tokens_1 = input_tokens_1
        self.input_ids_1 = input_ids_1
        self.position_idx_1 = position_idx_1
        self.dfg_to_code_1 = dfg_to_code_1
        self.dfg_to_dfg_1 = dfg_to_dfg_1

        # label
        self.label = label
        self.url1 = url1


def convert_examples_to_features(item):
    # source
    url1, label, tokenizer, args, cache, url_to_code = item
    parser = parsers['solidity']

    for url in [url1]:
        if url not in cache:
            func = url_to_code[url]

            # extract data flow
            code_tokens, dfg = extract_dataflow(func, parser, 'solidity')
            code_tokens = [tokenizer.tokenize(
                '@ '+x)[1:] if idx != 0 else tokenizer.tokenize(x) for idx, x in enumerate(code_tokens)]
            ori2cur_pos = {}
            ori2cur_pos[-1] = (0, 0)
            for i in range(len(code_tokens)):
                ori2cur_pos[i] = (ori2cur_pos[i-1][1],
                                  ori2cur_pos[i-1][1]+len(code_tokens[i]))
            code_tokens = [y for x in code_tokens for y in x]

            # truncating
            code_tokens = code_tokens[:args.code_length+args.data_flow_length -
                                      3-min(len(dfg), args.data_flow_length)][:512-3]
            source_tokens = [tokenizer.cls_token] + \
                code_tokens+[tokenizer.sep_token]
            source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
            position_idx = [i+tokenizer.pad_token_id +
                            1 for i in range(len(source_tokens))]
            dfg = dfg[:args.code_length +
                      args.data_flow_length-len(source_tokens)]
            source_tokens += [x[0] for x in dfg]
            position_idx += [0 for x in dfg]
            source_ids += [tokenizer.unk_token_id for x in dfg]
            padding_length = args.code_length + \
                args.data_flow_length-len(source_ids)
            position_idx += [tokenizer.pad_token_id]*padding_length
            source_ids += [tokenizer.pad_token_id]*padding_length

            # reindex
            reverse_index = {}
            for idx, x in enumerate(dfg):
                reverse_index[x[1]] = idx
            for idx, x in enumerate(dfg):
                dfg[idx] = x[:-1]+([reverse_index[i]
                                    for i in x[-1] if i in reverse_index],)
            dfg_to_dfg = [x[-1] for x in dfg]
            dfg_to_code = [ori2cur_pos[x[1]] for x in dfg]
            length = len([tokenizer.cls_token])
            dfg_to_code = [(x[0]+length, x[1]+length) for x in dfg_to_code]
            cache[url] = source_tokens, source_ids, position_idx, dfg_to_code, dfg_to_dfg

    source_tokens_1, source_ids_1, position_idx_1, dfg_to_code_1, dfg_to_dfg_1 = cache[url1]
    return InputFeatures(source_tokens_1, source_ids_1, position_idx_1, dfg_to_code_1, dfg_to_dfg_1,

                         label, url1)


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path='train'):
        self.examples = []
        self.args = args
        index_filename = file_path

        # load index
        logger.info("Creating features from index file at %s ", index_filename)
        url_to_code = {}
        with open('/home/chen/workspace/codeproject/CL4acrossVersionSC/dataset/cross-version_dataset/cross-version_jsonl/ALL_04_08.jsonl') as f:
        # with open('/'.join(index_filename.split('/')[:-1])+'/data.jsonl') as f:
            for line in f:
                line = line.strip()
                js = json.loads(line)
                url_to_code[js['idx']] = js['contract']

        # load code function according to index
        data = []
        cache = {}
        f = open(index_filename)
        with open(index_filename) as f:
            for line in f:
                # line = line.strip()
                # url1, label = line.split('\t')
                url1, label, version = re.split(r'\s+', line.strip())
                # print(url1, label)
                if url1 not in url_to_code:
                    continue  # jump out of for
                if label == '0':
                    label = 0
                else:
                    label = 1
                data.append((url1, label, tokenizer, args, cache, url_to_code))

        # only use 10% valid data to keep best model
        if 'valid' in file_path:
            data = random.sample(data, int(len(data)*0.1))

        # convert example to input features
        self.examples = [convert_examples_to_features(
            x) for x in tqdm(data, total=len(data))]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        # calculate graph-guided masked function
        attn_mask_1 = np.zeros((self.args.code_length+self.args.data_flow_length,
                                self.args.code_length+self.args.data_flow_length), dtype=np.bool_)
        # calculate begin index of node and max length of input
        node_index = sum([i > 1 for i in self.examples[item].position_idx_1])
        max_length = sum([i != 1 for i in self.examples[item].position_idx_1])
        # sequence can attend to sequence
        attn_mask_1[:node_index, :node_index] = True
        # special tokens attend to all tokens
        for idx, i in enumerate(self.examples[item].input_ids_1):
            if i in [0, 2]:
                attn_mask_1[idx, :max_length] = True
        # nodes attend to code tokens that are identified from
        for idx, (a, b) in enumerate(self.examples[item].dfg_to_code_1):
            if a < node_index and b < node_index:
                attn_mask_1[idx+node_index, a:b] = True
                attn_mask_1[a:b, idx+node_index] = True
        # nodes attend to adjacent nodes
        for idx, nodes in enumerate(self.examples[item].dfg_to_dfg_1):
            for a in nodes:
                if a+node_index < len(self.examples[item].position_idx_1):
                    attn_mask_1[idx+node_index, a+node_index] = True

        return (torch.tensor(self.examples[item].input_ids_1),
                torch.tensor(self.examples[item].position_idx_1),
                torch.tensor(attn_mask_1),
                torch.tensor(self.examples[item].label))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer, file_version):
    """ Train the model """

    # build dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=4)

    args.max_steps = args.epochs*len(train_dataloader)
    args.save_steps = len(train_dataloader)//10
    args.warmup_steps = args.max_steps//5
    model.to(args.device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)

    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.train_batch_size//args.n_gpu)
    logger.info("  Total train batch size = %d",
                args.train_batch_size*args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)

    global_step = 0
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_f1 = 0

    model.zero_grad()

    for idx in range(args.epochs):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            (inputs_ids_1, position_idx_1, attn_mask_1,
             labels) = [x.to(args.device) for x in batch]
            model.train()
            loss, logits = model(
                inputs_ids_1, position_idx_1, attn_mask_1, labels)

            if args.n_gpu > 1:
                loss = loss.mean()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss

            avg_loss = round(train_loss/tr_num, 5)
            bar.set_description("epoch {} loss {}".format(idx, avg_loss))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                output_flag = True
                avg_loss = round(
                    np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4)

                if global_step % args.save_steps == 0:
                    results = evaluate(args, model, tokenizer, file_version,
                                       eval_when_training=True)

                    # Save model checkpoint
                    print('previous best f1:', best_f1)
                    print('current f1:', results['eval_f1'])
                    if results['eval_f1'] > best_f1:
                        best_f1 = results['eval_f1']
                        logger.info("  "+"*"*20)
                        logger.info("  Best f1:%s", round(best_f1, 4))
                        logger.info("  "+"*"*20)

                        checkpoint_prefix = 'checkpoint-best-f1'
                        output_dir = os.path.join(
                            args.output_dir, '{}'.format(checkpoint_prefix))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(
                            model, 'module') else model
                        output_dir = os.path.join(
                            output_dir, '{}'.format('model.bin'))
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info(
                            "Saving model checkpoint to %s", output_dir)


def evaluate(args, model, tokenizer, file_version, eval_when_training=False):
    # build dataloader

    print('running evaluate on file_version: ', file_version)
    logger.info("running evaluate on file_version: %s", file_version)
    eval_data_file_key = f"eval_data_file_{file_version}"
    eval_file_path = getattr(args, eval_data_file_key)
    eval_dataset = TextDataset(tokenizer, args, file_path=eval_file_path)
    # eval_dataset = TextDataset(tokenizer, args, file_path=args.eval_data_file)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    y_trues = []
    for batch in eval_dataloader:
        (inputs_ids_1, position_idx_1, attn_mask_1,
         labels) = [x.to(args.device) for x in batch]
        with torch.no_grad():
            lm_loss, logit = model(
                inputs_ids_1, position_idx_1, attn_mask_1, labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1

    # calculate scores
    logits = np.concatenate(logits, 0)

    y_trues = np.concatenate(y_trues, 0)
    best_threshold = 0.5
    best_f1 = 0

    y_scores = logits
    y_preds = logits[:, 1] > best_threshold

    #计算TP,FP,TN,FN
    TP = np.sum((y_trues==1) & (y_preds==1))
    FP = np.sum((y_trues==0) & (y_preds==1))
    TN = np.sum((y_trues==0) & (y_preds==0))
    FN = np.sum((y_trues==1) & (y_preds==0))


    from sklearn.metrics import recall_score
    recall = recall_score(y_trues, y_preds, average='macro')
    from sklearn.metrics import precision_score
    precision = precision_score(y_trues, y_preds, average='macro')
    from sklearn.metrics import f1_score
    f1 = f1_score(y_trues, y_preds, average='macro')
    macro_f1 = 2 * (precision * recall) / (precision + recall)

    #acc
    acc = (TP+TN)/(TP+TN+FP+FN)

    #计算TPR和FPR,使用tp，fp，tn，fn计算
    TPR = TP / (TP+FN)
    FPR = FP / (FP+TN)
    #使用TPR和FPR计算AUC
    from sklearn.metrics import roc_curve, roc_auc_score, auc

    #计算MCC
    from sklearn.metrics import matthews_corrcoef
    mcc = matthews_corrcoef(y_trues, y_preds)

    result = {
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "eval_macro_f1": float(macro_f1),
        "eval_acc": float(acc),
        "eval_threshold": best_threshold,
        "eval_TPR": float(TPR),
        "eval_FPR": float(FPR),
        "eval_mcc": float(mcc)
    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result


def test(args, model, tokenizer, file_version, best_threshold=0):
    # build dataloader
    print('running test on file_version: ', file_version)
    logger.info("running test on file_version: %s", file_version)

    test_data_file_key = f"test_data_file_{file_version}"
    test_file_path = getattr(args, test_data_file_key)
    # build dataloader
    eval_dataset = TextDataset(tokenizer, args, file_path=test_file_path)

    # eval_dataset = TextDataset(tokenizer, args, file_path=args.test_data_file)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    y_trues = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        (inputs_ids_1, position_idx_1, attn_mask_1,
         labels) = [x.to(args.device) for x in batch]
        with torch.no_grad():
            lm_loss, logit = model(
                inputs_ids_1, position_idx_1, attn_mask_1, labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1

    # output result
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    y_preds = logits[:, 1] > best_threshold
    y_scores = logits
    with open(os.path.join(args.output_dir, "predictions.txt"), 'w') as f:
        for example, pred in zip(eval_dataset.examples, y_preds):
            if pred:
                f.write(example.url1+'\t'+'1'+'\n')
            else:
                f.write(example.url1+'\t'+'0'+'\n')

    #计算TP,FP,TN,FN
    TP = np.sum((y_trues==1) & (y_preds==1))
    FP = np.sum((y_trues==0) & (y_preds==1))
    TN = np.sum((y_trues==0) & (y_preds==0))
    FN = np.sum((y_trues==1) & (y_preds==0))

    # calculate scores
    from sklearn.metrics import recall_score
    recall = recall_score(y_trues, y_preds, average='macro')
    from sklearn.metrics import precision_score
    precision = precision_score(y_trues, y_preds, average='macro')
    from sklearn.metrics import f1_score
    f1 = f1_score(y_trues, y_preds, average='macro')
    macro_f1 = 2 * (precision * recall) / (precision + recall)

    #acc
    acc = (TP+TN)/(TP+TN+FP+FN)

    #计算MCC
    from sklearn.metrics import matthews_corrcoef
    mcc = matthews_corrcoef(y_trues, y_preds)

    #计算TPR和FPR,使用tp，fp，tn，fn计算
    TPR = TP / (TP+FN)
    FPR = FP / (FP+TN)

    result = {
        "test_recall": float(recall),
        "test_precision": float(precision),
        "test_f1": float(f1),
        "test_macro_f1": float(macro_f1),
        "test_acc": float(acc),
        "test_threshold": best_threshold,
        "test_TPR": float(TPR),
        "test_FPR": float(FPR),
        "test_mcc": float(mcc)
    }

    logger.info("***** Test results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))
        
        # wandb.log({key: result[key]})

    return result

from torch.utils.data import Subset

def load_train_dataset_for_version(file_version, args, tokenizer, sample_size_for_other_versions):
    train_data_file_key = f"train_data_file_{file_version}"
    train_file_path = getattr(args, train_data_file_key)

    logger.info("Loading training dataset for version %s: %s", file_version, train_file_path)
    print(f'Loding training dataset for version {file_version}: {train_file_path}')
        
        # 加载指定版本的数据集
    train_dataset = TextDataset(tokenizer, args, file_path=train_file_path)

    # 对于0.4版本，加载全部数据
    if file_version == '04_07':
        logger.info("ALL the training dataset for version %s has been loaded.", file_version)
        print(f"ALL the training dataset for version {file_version} has been loaded.")
        return train_dataset
    # 对于非0.4版本，进行随机抽样
    else:
        total_samples = len(train_dataset)
        sample_indices = random.sample(range(total_samples), min(total_samples,sample_size_for_other_versions))
        sampled_train_dataset = Subset(train_dataset, sample_indices)
        logger.info("A random sample of %d training dataset for version %s has been loaded.", sample_size_for_other_versions, file_version)
        print(f"A random sample of {sample_size_for_other_versions} training dataset for version {file_version} has been loaded.")
        return sampled_train_dataset
    
def main():

    args = RuntimeContext()

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s", args.device, args.n_gpu,)
    # Set seed
    set_seed(args)

    config = RobertaConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path)
    config.num_labels = 1
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)

    versions = ['0.4_0.7', '0.8']  # 版本顺序列表
    # initial_model_path = "/home/chen/workspace/codeproject/CL4acrossVersionSC/modelSingleVulSingleVersion/pretrained_model/graphcodebert-base"  # 初始预训练模型路径

    model_base_path = "/home/chen/workspace/codeproject/CL4acrossVersionSC/model_OneVulCrossVersion/Peculiar/saved_model04-07with08"  # 保存模型的基本路径
    # 确保基本路径存在
    if not os.path.exists(model_base_path):
        os.makedirs(model_base_path)

    for i, version in enumerate(versions):
        


        if version == '0.4_0.7':
            if os.path.exists(args.model_name_or_path):
                print("Loading the initial pre-trained encoder model...")
                logger.info("Loading the initial pre-trained encoder model...")
                encoder = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
                model = Model(encoder, config, tokenizer, args)
            else:
                print(f"The initial pre-trained model does not exist, please make sure the path is correct: {args.model_name_or_path}")
                logger.error("The initial pre-trained model does not exist, please make sure the path %s is correct!", args.model_name_or_path)

        else:
            # 非版本：加载上一个版本的模型
            previous_version = versions[i-1]  # 获取上一个版本
            model_filename_to_load = f"model_v{previous_version}.bin"
            model_path_to_load = os.path.join(model_base_path, model_filename_to_load)
            
            if os.path.exists(model_path_to_load):
                print(f"Loading the {previous_version} saved model...")
                logger.info("Loading the %s saved model...", previous_version)
                model.load_state_dict(torch.load(model_path_to_load))
            else:
                print(f"Cannot find the trained model file for version {previous_version}: {model_path_to_load}, please check!")
                logger.error("Cannot find the trained model file for version %s: %s, please check!", previous_version, model_path_to_load)

        # 加载当前版本的数据并进行训练...
        # info for load current version num
        logger.info(f"Start loading model on data of version {version} with sample size {args.sample_size_for_other_versions}...")

        file_version = version.replace('.', '')
        train_dataset = load_train_dataset_for_version(file_version, args, tokenizer, args.sample_size_for_other_versions)

        
        
        logger.info("***** Running training for version %s *****", version)
        print(f"Start training model on data of version {version}")
        # wandb.watch(model, log='all')
        train(args, train_dataset, model, tokenizer, file_version)
        
        # 训练完成后，保存当前版本的模型
        model_filename_to_save = f"model_v{version}.bin"
        model_path_to_save = os.path.join(model_base_path, model_filename_to_save)
        torch.save(model.state_dict(), model_path_to_save)
        print(f"Model trained on data of version {version} has been saved at: {model_path_to_save}")
        logger.info("Model trained on data of version %s has been saved at: %s", version, model_path_to_save)


        checkpoint_prefix = 'checkpoint-best-f1/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        print(f"best f1 model on eval data of version {version} has been loaded")
        logger.info("best f1 model on eval data of version %s has been loaded", version)
        test(args, model, tokenizer, file_version, best_threshold=0.5)


if __name__ == "__main__":
    main()
