import torch
from collections import OrderedDict
from torch_geometric.nn import GCNConv, SAGEConv
from torch.nn.functional import cosine_similarity
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_sparse import SparseTensor
from sklearn.preprocessing import normalize
import json
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
import copy
import argparse
import random
from self_parser import parser
import torch.nn.functional as F
import numpy as np
from utility import (transform_dict, cs_categories, product_categories_raw, arxiv24_categories_raw, products_mapping,
                     get_raw_text_arxiv_2023, logits_to_one_hot,
                     remove_key_from_values, build_neighbor_dict,
                     one_hot_to_class_indices, process_tensor,
                     transform_keys, prepare_encodings, prepare_encodings_tmp,
                     normalize_ppl, cal_loss_simple, save_dict_as_pickle,
                     compute_similarity_scores)
import torch_geometric.transforms as T
import os
import pickle

from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer


cs_categories_short = transform_dict(original_dict=cs_categories)


def gen_arxiv_message(context,  example_input=None, example_output=None):
    sys_prompt = f"You are an AI trained to categorize arXiv computer science papers into specific categories based on their abstracts. Your task is to analyze the paper description provided and identify the most relevant category."
    user_prompt = f"Paper description: {context.strip()}\nGive me the category of this content. Respond only with the category key (e.g., 'cs.AI', 'cs.SY'), without any additional text or explanation."

    icl_str = f"This is an example: " \
              f"\nExample 1: \nInput: {example_input[0]}\nOutput: {example_output[0]}"

    messages = [{"role": "system", "content": sys_prompt + f"\n{icl_str}"},
                {"role": "user", "content": user_prompt}]
    return messages


def gen_product_message(context, example_input=None, example_output=None):
    categories_list = ", ".join([f"'{category}'" for category in products_mapping.values()])
    context = context[:1200] if len(context) > 1200 else context
    sys_prompt = f"You are an AI trained to categorize products into specific categories based on their descriptions and characteristics. Your task is to analyze the product description provided, consider its characteristics, and identify the most relevant category among hundreds of possible categories. There are a total of 47 categories, including {categories_list}."
    user_prompt = f"Product description: {context.strip()}\nConsider its characteristics and give me the category of this product. Respond only with the category key (e.g., 'Electronics', 'Toys & Games'), without any additional text or explanation."

    icl_str = ""
    if example_input and example_output and len(example_input) == len(example_output):
        for idx, (inp, out) in enumerate(zip(example_input, example_output)):
            trimmed_input = inp[:800] if len(inp) > 800 else inp
            icl_str += f"\nExample {idx + 1}: \nInput: {trimmed_input}\nOutput: {out}\n\n"
    messages = [{"role": "system",
                 "content": sys_prompt + f"\nHere are some examples to help you understand how to categorize products based on their descriptions:{icl_str}"},
                {"role": "user", "content": user_prompt}]
    return messages


def create_chat_message(context, version='zero-shot', example_input=None, example_output=None,
                        dataset_name='ogbn-arxiv', llm_model='gpt35'):
    if dataset_name in ['ogbn-arxiv', 'arxiv_2023']:
        messages = gen_arxiv_message(context=context,
                                     example_input=example_input,
                                     example_output=example_output,
                                     )
    elif dataset_name == 'ogbn-products':
        messages = gen_product_message(context=context,
                                       example_input=example_input,
                                       example_output=example_output,
                                       )
    return messages

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


def train(model, data, train_idx, optimizer, stru='sage'):
    if stru != 'mlp':
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.adj_t)[train_idx]
        loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    else:
        model.train()
        optimizer.zero_grad()
        out = model(data.x[train_idx])
        loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    return loss, out


@torch.no_grad()
def test(model, data, split_idx, evaluator, stru='sage'):
    if stru != 'mlp':
        model.eval()

        out = model(data.x, data.adj_t)
        y_pred = out.argmax(dim=-1, keepdim=True)

        train_acc = evaluator.eval({
            'y_true': data.y[split_idx['train']],
            'y_pred': y_pred[split_idx['train']],
        })['acc']
        valid_acc = evaluator.eval({
            'y_true': data.y[split_idx['valid']],
            'y_pred': y_pred[split_idx['valid']],
        })['acc']
        test_acc = evaluator.eval({
            'y_true': data.y[split_idx['test']],
            'y_pred': y_pred[split_idx['test']],
        })['acc']
    else:
        x = data.x
        y_true = data.y
        model.eval()

        out = model(x)
        y_pred = out.argmax(dim=-1, keepdim=True)

        train_acc = evaluator.eval({
            'y_true': y_true[split_idx['train']],
            'y_pred': y_pred[split_idx['train']],
        })['acc']
        valid_acc = evaluator.eval({
            'y_true': y_true[split_idx['valid']],
            'y_pred': y_pred[split_idx['valid']],
        })['acc']
        test_acc = evaluator.eval({
            'y_true': y_true[split_idx['test']],
            'y_pred': y_pred[split_idx['test']],
        })['acc']

    return train_acc, valid_acc, test_acc




def main():
    ratio = args.ratio
    epoch_times = args.epoch_times
    selected_sample_num = 1000
    dataset_name = args.dataset_name
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device1 = device
    device2 = device


    if dataset_name == 'ogbn-arxiv':
        text_path = f"your text_path"
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', transform=T.ToSparseTensor())
        dataset_num_classes = dataset.num_classes
        text_data = open(text_path, 'r').readlines()
        data = dataset[0]
        data.adj_t = data.adj_t.to_symmetric()
        data = data.to(device)
        split_idx = dataset.get_idx_split()

    elif dataset_name == 'ogbn-products':
        text_path = f"your text_path"
        dataset = PygNodePropPredDataset(name=dataset_name, transform=T.ToSparseTensor())
        dataset_num_classes = dataset.num_classes
        text_data = open(text_path, 'r').readlines()
        data = dataset[0]
        data.adj_t = data.adj_t.to_symmetric()
        data = data.to(device)
        split_idx = dataset.get_idx_split()

    elif dataset_name == 'arxiv_2023':
        data, text = get_raw_text_arxiv_2023(use_text=True)
        data = data.to(device)
        text_data = text
        split_idx = {}
        split_idx['train'] = torch.where(data.train_mask)[0]
        split_idx['valid'] = torch.where(data.val_mask)[0]
        split_idx['test'] = torch.where(data.test_mask)[0]
        options = set(text['label'])
        dataset_num_classes = len(options)
        edge_index = data.edge_index
        num_nodes = data.num_nodes
        row, col = edge_index
        adj_t = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes)).to_symmetric()
        print(adj_t)
        data.adj_t = adj_t
        data.adj_t = data.adj_t.to_symmetric()
        data.y = process_tensor(data.y)



    if args.llm_model == 'qwen':
        cach_path = 'your model path'
        tokenizer = AutoTokenizer.from_pretrained(cach_path)
        llm_model = AutoModelForCausalLM.from_pretrained(
            cach_path,
            torch_dtype="auto",
            device_map="auto"
        )
    if args.dataset_name == 'ogbn-arxiv':
        categories = cs_categories
    elif args.dataset_name == 'ogbn-products':
        product_categories = {}
        for key, value in product_categories_raw.items():
            product_categories[key] = [products_mapping[value]]
        categories = product_categories
    elif args.dataset_name == 'arxiv_2023':
        arxiv24_categories = {}
        for key, value in arxiv24_categories_raw.items():
            arxiv24_categories[key] = [value]
        categories = arxiv24_categories
    catego_list = []
    for key, value in categories.items():
        catego_list.append(value[0])
    train_idx_full = split_idx['train'].to(device)
    total_train_size = len(train_idx_full)
    train_size = int(ratio * total_train_size)
    perm = torch.randperm(total_train_size)
    train_idx = train_idx_full[perm[:train_size]]

    model = SAGE(data.num_features, args.hidden_channels,
                 dataset_num_classes, args.num_layers,
                 args.dropout).to(device)

    real_labels = data.y.squeeze(1)
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(1, 1 + args.epochs):
        if epoch <= epoch_times:
            loss, out = train(model, data, train_idx, optimizer)
            norm_out = out / out.norm(dim=1, keepdim=True)
            cos_sim_matrix = torch.mm(norm_out, norm_out.t())
            top_k_values, top_k_indices = torch.topk(cos_sim_matrix, k=20, largest=True, dim=-1)
            map_dict, _ = tensor_to_dict(A=train_idx, B=top_k_indices, train_idx=train_idx)
            map_dict = remove_key_from_values(map_dict)
            if args.dataset_name  == 'arxiv_2023':
                raw_data = text_data['abs']
            else:
                raw_data = text_data
            accu_loss = 0
            train_icl_dict = {}
            for train_node_idx_index, train_node_idx in tqdm(enumerate(train_idx), total=len(train_idx), desc="Processing nodes"):
                train_node_idx = train_node_idx.item()
                if train_node_idx not in train_icl_dict:
                    train_icl_dict[train_node_idx] = {}


                train_node_neigh = map_dict[train_node_idx]
                train_node_label = categories[real_labels[train_node_idx].item()][0]
                train_node_neigh_ppl_map_dict = {}
                with torch.no_grad():
                    for select_neigh in train_node_neigh:
                        if select_neigh in train_icl_dict[train_node_idx]:
                            train_node_neigh_ppl_map_dict[select_neigh] = train_icl_dict[train_node_idx][
                                select_neigh]
                        else:
                            raw_content = raw_data[train_node_idx]
                            example_input = [raw_data[select_neigh]]
                            example_output = [categories[real_labels[select_neigh].item()][0]]
                            messages_tmp = create_chat_message(context=raw_content,
                                                               version='zero-shot',
                                                               example_input=example_input,
                                                               example_output=example_output,
                                                               dataset_name=args.dataset_name)
                            messages_template_tmp = tokenizer.apply_chat_template(
                                messages_tmp,
                                tokenize=False,
                                add_generation_prompt=True
                            )
                            result_dict = prepare_encodings_tmp(messages_template=messages_template_tmp,
                                                                categories=catego_list,
                                                                tokenizer=tokenizer,
                                                                device=device1,
                                                                model=llm_model,
                                                                )
                            result_dict = dict(result_dict)
                            ppl_confidence = normalize_ppl(result_dict)
                            final_ppl = ppl_confidence[train_node_label]
                            train_node_neigh_ppl_map_dict[select_neigh] = final_ppl
                            train_icl_dict[train_node_idx][select_neigh] = final_ppl
                    sorted_keys_desc_tmp = sorted(train_node_neigh_ppl_map_dict,
                                                  key=train_node_neigh_ppl_map_dict.get, reverse=True)
                    idx_dict = {value.item(): index for index, value in enumerate(train_idx)}
                    sorted_keys_desc = torch.tensor([idx_dict[nu] for nu in sorted_keys_desc_tmp]).to(device1)
                similarity_scores = compute_similarity_scores(out=out, sorted_keys_desc=sorted_keys_desc, train_node_idx_index=train_node_idx_index)
                t_loss = cal_loss_simple(scores=similarity_scores)
                accu_loss += t_loss
            final_loss = loss + (accu_loss * 0.15) / len(train_idx)
            final_loss.backward()
            optimizer.step()


        elif epoch > epoch_times:
            loss, out = train(model, data, train_idx, optimizer)

            random_indices = torch.randperm(len(split_idx['test']))[:selected_sample_num]
            selected_samples_from_remain = split_idx['test'][random_indices]
            pseudo_label_for_all_node = model(data.x, data.adj_t)
            out_remain_selected = pseudo_label_for_all_node[selected_samples_from_remain]
            out_remain_selected = out_remain_selected.to(device2)
            out = out.to(device2)
            cos_sim = cosine_similarity(out_remain_selected.unsqueeze(1), out.unsqueeze(0), dim=-1)
            top_k_values, top_k_indices = torch.topk(cos_sim, k=3, largest=True, dim=-1)
            train_idx_2 = train_idx.to(device2)
            map_dict, _ = tensor_to_dict(A=selected_samples_from_remain, B=top_k_indices,
                                                       train_idx=train_idx_2)
            final_dict = {
                "map_dict": map_dict,
            }
            save_dict_as_pickle(dictionary=final_dict, file_path='your path')


if __name__ == "__main__":
    main()
