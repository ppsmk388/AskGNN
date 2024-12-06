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

import torch_geometric.transforms as T
import os
import pickle

from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer



cs_categories = {
    10: ["cs.AI", "(Artificial Intelligence)",
         "Covers all areas of AI except Vision, Robotics, Machine Learning, Multiagent Systems, and Computation and Language (Natural Language Processing), which have separate subject areas. In particular, includes Expert Systems, Theorem Proving (although this may overlap with Logic in Computer Science), Knowledge Representation, Planning, and Uncertainty in AI. Roughly includes material in ACM Subject Classes I.2.0, I.2.1, I.2.3, I.2.4, I.2.8, and I.2.11."],
    15: ["cs.AR", "(Hardware Architecture)",
         "Covers systems organization and hardware architecture. Roughly includes material in ACM Subject Classes C.0, C.1, and C.5."],
    9: ["cs.CC", "(Computational Complexity)",
        "Covers models of computation, complexity classes, structural complexity, complexity tradeoffs, upper and lower bounds. Roughly includes material in ACM Subject Classes F.1 (computation by abstract devices), F.2.3 (tradeoffs among complexity measures), and F.4.3 (formal languages), although some material in formal languages may be more appropriate for Logic in Computer Science. Some material in F.2.1 and F.2.2, may also be appropriate here, but is more likely to have Data Structures and Algorithms as the primary subject area."],
    7: ["cs.CE", "(Computational Engineering, Finance, and Science)",
        "Covers applications of computer science to the mathematical modeling of complex systems in the fields of science, engineering, and finance. Papers here are interdisciplinary and applications-oriented, focusing on techniques and tools that enable challenging computational simulations to be performed, for which the use of supercomputers or distributed computing platforms is often required. Includes material in ACM Subject Classes J.2, J.3, and J.4 (economics)."],
    20: ["cs.CG", "(Computational Geometry)",
         "Roughly includes material in ACM Subject Classes I.3.5 and F.2.2."],
    30: ["cs.CL", "(Computation and Language)",
         "Covers natural language processing. Roughly includes material in ACM Subject Class I.2.7. Note that work on artificial languages (programming languages, logics, formal systems) that does not explicitly address natural-language issues broadly construed (natural-language processing, computational linguistics, speech, text retrieval, etc.) is not appropriate for this area."],
    4: ["cs.CR", "(Cryptography and Security)",
        "Covers all areas of cryptography and security including authentication, public key cryptosytems, proof-carrying code, etc. Roughly includes material in ACM Subject Classes D.4.6 and E.3."],
    16: ["cs.CV", "(Computer Vision and Pattern Recognition)",
         "Covers image processing, computer vision, pattern recognition, and scene understanding. Roughly includes material in ACM Subject Classes I.2.10, I.4, and I.5."],
    3: ["cs.CY", "(Computers and Society)",
        "Covers impact of computers on society, computer ethics, information technology and public policy, legal aspects of computing, computers and education. Roughly includes material in ACM Subject Classes K.0, K.2, K.3, K.4, K.5, and K.7."],
    37: ["cs.DB", "(Databases)",
         "Covers database management, datamining, and data processing. Roughly includes material in ACM Subject Classes E.2, E.5, H.0, H.2, and J.1."],
    5: ["cs.DC", "(Distributed, Parallel, and Cluster Computing)",
        "Covers fault-tolerance, distributed algorithms, stabilility, parallel computation, and cluster computing. Roughly includes material in ACM Subject Classes C.1.2, C.1.4, C.2.4, D.1.3, D.4.5, D.4.7, E.1."],
    38: ["cs.DL", "(Digital Libraries)",
         "Covers all aspects of the digital library design and document and text creation. Note that there will be some overlap with Information Retrieval (which is a separate subject area). Roughly includes material in ACM Subject Classes H.3.5, H.3.6, H.3.7, I.7."],
    39: ["cs.DM", "(Discrete Mathematics)",
         "Covers combinatorics, graph theory, applications of probability. Roughly includes material in ACM Subject Classes G.2 and G.3."],
    34: ["cs.DS", "(Data Structures and Algorithms)",
         "Covers data structures and analysis of algorithms. Roughly includes material in ACM Subject Classes E.1, E.2, F.2.1, and F.2.2."],
    18: ["cs.ET", "(Emerging Technologies)",
         "Covers approaches to information processing (computing, communication, sensing) and bio-chemical analysis based on alternatives to silicon CMOS-based technologies, such as nanoscale electronic, photonic, spin-based, superconducting, mechanical, bio-chemical and quantum technologies (this list is not exclusive). Topics of interest include (1) building blocks for emerging technologies, their scalability and adoption in larger systems, including integration with traditional technologies, (2) modeling, design and optimization of novel devices and systems, (3) models of computation, algorithm design and programming for emerging technologies."],
    33: ["cs.FL", "(Formal Languages and Automata Theory)",
         "Covers automata theory, formal language theory, grammars, and combinatorics on words. This roughly corresponds to ACM Subject Classes F.1.1, and F.4.3. Papers dealing with computational complexity should go to cs.CC; papers dealing with logic should go to cs.LO."],
    12: ["cs.GL", "(General Literature)",
         "Covers introductory material, survey material, predictions of future trends, biographies, and miscellaneous computer-science related material. Roughly includes all of ACM Subject Class A, except it does not include conference proceedings (which will be listed in the appropriate subject area)."],
    17: ["cs.GR", "(Graphics)",
         "Covers all aspects of computer graphics. Roughly includes material in all of ACM Subject Class I.3, except that I.3.5 is is likely to have Computational Geometry as the primary subject area."],
    36: ["cs.GT", "(Computer Science and Game Theory)",
         "Covers all theoretical and applied aspects at the intersection of computer science and game theory, including work in mechanism design, learning in games (which may overlap with Learning), foundations of agent modeling in games (which may overlap with Multiagent systems), coordination, specification and formal methods for non-cooperative computational environments. The area also deals with applications of game theory to areas such as electronic commerce."],
    6: ["cs.HC", "(Human-Computer Interaction)",
        "Covers human factors, user interfaces, and collaborative computing. Roughly includes material in ACM Subject Classes H.1.2 and all of H.5, except for H.5.1, which is more likely to have Multimedia as the primary subject area."],
    31: ["cs.IR", "(Information Retrieval)",
         "Covers indexing, dictionaries, retrieval, content and analysis. Roughly includes material in ACM Subject Classes H.3.0, H.3.1, H.3.2, H.3.3, and H.3.4."],
    28: ["cs.IT", "(Information Theory)",
         "Covers theoretical and experimental aspects of information theory and coding. Includes material in ACM Subject Class E.4 and intersects with H.1.1."],
    24: ["cs.LG", "(Machine Learning)",
         "Papers on all aspects of machine learning research (supervised, unsupervised, reinforcement learning, bandit problems, and so on) including also robustness, explanation, fairness, and methodology. cs.LG is also an appropriate primary category for applications of machine learning methods."],
    2: ["cs.LO", "(Logic in Computer Science)",
        "Covers all aspects of logic in computer science, including finite model theory, logics of programs, modal logic, and program verification. Programming language semantics should have Programming Languages as the primary subject area. Roughly includes material in ACM Subject Classes D.2.4, F.3.1, F.4.0, F.4.1, and F.4.2; some material in F.4.3 (formal languages) may also be appropriate here, although Computational Complexity is typically the more appropriate subject area."],
    11: ["cs.MA", "(Multiagent Systems)",
         "Covers multiagent systems, distributed artificial intelligence, intelligent agents, coordinated interactions. and practical applications. Roughly covers ACM Subject Class I.2.11."],
    1: ["cs.MM", "(Multimedia)",
        "Roughly includes material in ACM Subject Class H.5.1."],
    32: ["cs.MS", "(Mathematical Software)",
         "Roughly includes material in ACM Subject Class G.4."],
    0: ["cs.NA", "(Numerical Analysis)",
        "cs.NA is an alias for math.NA. Roughly includes material in ACM Subject Class G.1."],
    13: ["cs.NE", "(Neural and Evolutionary Computing)",
         "Covers neural networks, connectionism, genetic algorithms, artificial life, adaptive behavior. Roughly includes some material in ACM Subject Class C.1.3, I.2.6, I.5."],
    8: ["cs.NI", "(Networking and Internet Architecture)",
        "Covers all aspects of computer communication networks, including network architecture and design, network protocols, and internetwork standards (like TCP/IP). Also includes topics, such as web caching, that are directly relevant to Internet architecture and performance. Roughly includes all of ACM Subject Class C.2 except C.2.4, which is more likely to have Distributed, Parallel, and Cluster Computing as the primary subject area."],
    21: ["cs.OH", "(Other Computer Science)",
         "This is the classification to use for documents that do not fit anywhere else."],
    35: ["cs.OS", "(Operating Systems)",
         "Roughly includes material in ACM Subject Classes D.4.1, D.4.2., D.4.3, D.4.4, D.4.5, D.4.7, and D.4.9."],
    29: ["cs.PF", "(Performance)",
         "Covers performance measurement and evaluation, queueing, and simulation. Roughly includes material in ACM Subject Classes D.4.8 and K.6.2."],
    22: ["cs.PL", "(Programming Languages)",
         "Covers programming language semantics, language features, programming approaches (such as object-oriented programming, functional programming, logic programming). Also includes material on compilers oriented towards programming languages; other material on compilers may be more appropriate in Architecture (AR). Roughly includes material in ACM Subject Classes D.1 and D.3."],
    27: ["cs.RO", "(Robotics)",
         "Roughly includes material in ACM Subject Class I.2.9."],
    14: ["cs.SC", "(Symbolic Computation)",
         "Roughly includes material in ACM Subject Class I.1."],
    25: ["cs.SD", "(Sound)",
         "Covers all aspects of computing with sound, and sound as an information channel. Includes models of sound, analysis and synthesis, audio user interfaces, sonification of data, computer music, and sound signal processing. Includes ACM Subject Class H.5.5, and intersects with H.1.2, H.5.1, H.5.2, I.2.7, I.5.4, I.6.3, J.5, K.4.2."],
    23: ["cs.SE", "(Software Engineering)",
         "Covers design tools, software metrics, testing and debugging, programming environments, etc. Roughly includes material in all of ACM Subject Classes D.2, except that D.2.4 (program verification) should probably have Logics in Computer Science as the primary subject area."],
    26: ["cs.SI", "(Social and Information Networks)",
         "Covers the design, analysis, and modeling of social and information networks, including their applications for on-line information access, communication, and interaction, and their roles as datasets in the exploration of questions in these and other domains, including connections to the social and biological sciences. Analysis and modeling of such networks includes topics in ACM Subject classes F.2, G.2, G.3, H.2, and I.2; applications in computing include topics in H.3, H.4, and H.5; and applications at the interface of computing and other disciplines include topics in J.1--J.7. Papers on computer communication systems and network protocols (e.g. TCP/IP) are generally a closer fit to the Networking and Internet Architecture (cs.NI) category."],
    19: ["cs.SY", "(Systems and Control)",
         "cs.SY is an alias for eess.SY. This section includes theoretical and experimental research covering all facets of automatic control systems. The section is focused on methods of control system analysis and design using tools of modeling, simulation and optimization. Specific areas of research include nonlinear, distributed, adaptive, stochastic and robust control in addition to hybrid and discrete event systems. Application areas include automotive and aerospace control systems, network control, biological systems, multiagent and cooperative control, robotics, reinforcement learning, sensor networks, control of cyber-physical and energy-related systems, and control of computing systems."],
}

product_categories_raw = {0: 'Home & Kitchen',
                          1: 'Health & Personal Care',
                          2: 'Beauty',
                          3: 'Sports & Outdoors',
                          4: 'Books',
                          5: 'Patio, Lawn & Garden',
                          6: 'Toys & Games',
                          7: 'CDs & Vinyl',
                          8: 'Cell Phones & Accessories',
                          9: 'Grocery & Gourmet Food',
                          10: 'Arts, Crafts & Sewing',
                          11: 'Clothing, Shoes & Jewelry',
                          12: 'Electronics',
                          13: 'Movies & TV',
                          14: 'Software',
                          15: 'Video Games',
                          16: 'Automotive',
                          17: 'Pet Supplies',
                          18: 'Office Products',
                          19: 'Industrial & Scientific',
                          20: 'Musical Instruments',
                          21: 'Tools & Home Improvement',
                          22: 'Magazine Subscriptions',
                          23: 'Baby Products',
                          24: 'label 25',
                          25: 'Appliances',
                          26: 'Kitchen & Dining',
                          27: 'Collectibles & Fine Art',
                          28: 'All Beauty',
                          29: 'Luxury Beauty',
                          30: 'Amazon Fashion',
                          31: 'Computers',
                          32: 'All Electronics',
                          33: 'Purchase Circles',
                          34: 'MP3 Players & Accessories',
                          35: 'Gift Cards',
                          36: 'Office & School Supplies',
                          37: 'Home Improvement',
                          38: 'Camera & Photo',
                          39: 'GPS & Navigation',
                          40: 'Digital Music',
                          41: 'Car Electronics',
                          42: 'Baby',
                          43: 'Kindle Store',
                          44: 'Buy a Kindle',
                          45: 'Furniture & D&#233;cor',
                          46: '#508510'}
arxiv24_categories_raw = {2: 'cs.LO', 23: 'cs.LG', 15: 'cs.CV', 10: 'cs.AI', 29: 'cs.CL', 31: 'cs.MS', 4: 'cs.CR', 18: 'cs.SY', 6: 'cs.HC', 0: 'cs.NA', 33: 'cs.DS', 25: 'cs.SI', 30: 'cs.IR', 5: 'cs.DC', 12: 'cs.NE', 22: 'cs.SE', 38: 'cs.DM',
                          9: 'cs.CC', 3: 'cs.CY', 26: 'cs.RO', 27: 'cs.IT', 14: 'cs.AR', 8: 'cs.NI', 35: 'cs.GT', 37: 'cs.DL', 7: 'cs.CE', 16: 'cs.GR', 24: 'cs.SD', 21: 'cs.PL', 36: 'cs.DB', 1: 'cs.MM', 11: 'cs.MA', 28: 'cs.PF', 32: 'cs.FL',
                          17: 'cs.ET', 19: 'cs.CG', 13: 'cs.SC', 20: 'cs.OH', 34: 'cs.OS'}

products_mapping = {'Home & Kitchen': 'Home & Kitchen',
                    'Health & Personal Care': 'Health & Personal Care',
                    'Beauty': 'Beauty',
                    'Sports & Outdoors': 'Sports & Outdoors',
                    'Books': 'Books',
                    'Patio, Lawn & Garden': 'Patio, Lawn & Garden',
                    'Toys & Games': 'Toys & Games',
                    'CDs & Vinyl': 'CDs & Vinyl',
                    'Cell Phones & Accessories': 'Cell Phones & Accessories',
                    'Grocery & Gourmet Food': 'Grocery & Gourmet Food',
                    'Arts, Crafts & Sewing': 'Arts, Crafts & Sewing',
                    'Clothing, Shoes & Jewelry': 'Clothing, Shoes & Jewelry',
                    'Electronics': 'Electronics',
                    'Movies & TV': 'Movies & TV',
                    'Software': 'Software',
                    'Video Games': 'Video Games',
                    'Automotive': 'Automotive',
                    'Pet Supplies': 'Pet Supplies',
                    'Office Products': 'Office Products',
                    'Industrial & Scientific': 'Industrial & Scientific',
                    'Musical Instruments': 'Musical Instruments',
                    'Tools & Home Improvement': 'Tools & Home Improvement',
                    'Magazine Subscriptions': 'Magazine Subscriptions',
                    'Baby Products': 'Baby Products',
                    'label 25': 'label 25',
                    'Appliances': 'Appliances',
                    'Kitchen & Dining': 'Kitchen & Dining',
                    'Collectibles & Fine Art': 'Collectibles & Fine Art',
                    'All Beauty': 'All Beauty',
                    'Luxury Beauty': 'Luxury Beauty',
                    'Amazon Fashion': 'Amazon Fashion',
                    'Computers': 'Computers',
                    'All Electronics': 'All Electronics',
                    'Purchase Circles': 'Purchase Circles',
                    'MP3 Players & Accessories': 'MP3 Players & Accessories',
                    'Gift Cards': 'Gift Cards',
                    'Office & School Supplies': 'Office & School Supplies',
                    'Home Improvement': 'Home Improvement',
                    'Camera & Photo': 'Camera & Photo',
                    'GPS & Navigation': 'GPS & Navigation',
                    'Digital Music': 'Digital Music',
                    'Car Electronics': 'Car Electronics',
                    'Baby': 'Baby',
                    'Kindle Store': 'Kindle Store',
                    'Buy a Kindle': 'Buy a Kindle',
                    'Furniture & D&#233;cor': 'Furniture & Decor',
                    '#508510': '#508510'}





def transform_dict(original_dict):
    new_dict = {}
    for key, value in original_dict.items():
        if len(value) >= 2:
            new_dict[value[0]] = value[1]
    return new_dict


def tensor_to_dict(A, B, train_idx, k=3):
    if not (A.ndim == 1 and B.ndim == 2 and A.shape[0] == B.shape[0]):
        raise ValueError("Invalid tensor shapes. A must be (K,) and B must be (K, M).")
    result_dict = {}
    A_array = A.cpu().numpy()
    for i in range(A.shape[0]):
        result_dict[A_array[i]] = train_idx[B[i]].tolist()
        random_indices = np.random.choice(len(train_idx), k, replace=False)
    return result_dict







def save_dict_as_pickle(dictionary, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(dictionary, file)


def get_raw_text_arxiv_2023(use_text=True):
    edge_index = torch.load(os.path.join(base_path, "processed", "edge_index.pt"))

    titles_df = pd.read_csv(os.path.join(base_path, "raw", "titles.csv.gz"), compression='gzip')
    abstracts_df = pd.read_csv(os.path.join(base_path, "raw", "abstracts.csv.gz"), compression='gzip')
    ids_df = pd.read_csv(os.path.join(base_path, "raw", "ids.csv.gz"), compression='gzip')
    labels_df = pd.read_csv(os.path.join(base_path, "raw", "labels.csv.gz"), compression='gzip')

    train_id_df = pd.read_csv(os.path.join(base_path, "split", "train.csv.gz"), compression='gzip')
    val_id_df = pd.read_csv(os.path.join(base_path, "split", "valid.csv.gz"), compression='gzip')
    test_id_df = pd.read_csv(os.path.join(base_path, "split", "test.csv.gz"), compression='gzip')

    num_nodes = len(ids_df)
    titles = titles_df['titles'].tolist()
    abstracts = abstracts_df['abstracts'].tolist()
    ids = ids_df['ids'].tolist()
    labels = labels_df['labels'].tolist()
    train_id = train_id_df['train_id'].tolist()
    val_id = val_id_df['val_id'].tolist()
    test_id = test_id_df['test_id'].tolist()

    features = torch.load(os.path.join(base_path, "processed", "features.pt"))

    y = torch.load(os.path.join(base_path, "processed", "labels.pt"))

    train_mask = torch.tensor([x in train_id for x in range(num_nodes)])
    val_mask = torch.tensor([x in val_id for x in range(num_nodes)])
    test_mask = torch.tensor([x in test_id for x in range(num_nodes)])

    data = Data(
        x=features,
        y=y,
        paper_id=ids,
        edge_index=edge_index,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_nodes=num_nodes,
    )

    data.train_id = train_id
    data.val_id = val_id
    data.test_id = test_id

    if not use_text:
        return data, None
    text = {'title': titles, 'abs': abstracts, 'label': labels, 'id': ids}

    return data, text
def logits_to_one_hot(logits, num_classes):
    softmax_scores = F.softmax(logits, dim=-1)
    _, predicted_classes = torch.max(softmax_scores, dim=-1)
    one_hot_labels = F.one_hot(predicted_classes, num_classes=num_classes)
    return one_hot_labels


def remove_key_from_values(d):
    new_dict = {}
    for key, value_list in d.items():
        new_dict[key] = [value for value in value_list if value != key]
    return new_dict


def build_neighbor_dict(edge_index):
    neighbors_dict = {}
    for start_node, end_node in edge_index.t().tolist():
        if start_node not in neighbors_dict:
            neighbors_dict[start_node] = []
        neighbors_dict[start_node].append(end_node)
        if end_node not in neighbors_dict:
            neighbors_dict[end_node] = []
        neighbors_dict[end_node].append(start_node)
    for node, neighbors in neighbors_dict.items():
        neighbors_dict[node] = list(set(neighbors))
    return neighbors_dict

def one_hot_to_class_indices(one_hot_labels):
    class_indices = torch.argmax(one_hot_labels, dim=-1)
    class_indices = class_indices.unsqueeze(-1)
    return class_indices


def process_tensor(input_tensor):
    processed_tensor = torch.where(input_tensor > 12, input_tensor - 1, input_tensor)
    return processed_tensor


def transform_keys(input_dict):
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.item()
        new_dict[new_key] = value
    return new_dict

def prepare_encodings(messages_template, categories, tokenizer, device, max_length=None, add_start_token=False):
    encoded_categories = {}
    for category in categories:
        messages_for_category = messages_template.replace("{category}", category)
        predictions = [messages_for_category]
        encodings = tokenizer(predictions, add_special_tokens=False, padding=True,
                              truncation=True if max_length else False,
                              max_length=max_length - 1 if add_start_token else max_length, return_tensors="pt",
                              return_attention_mask=True).to(device)
        encoded_categories[category] = encodings
    return encoded_categories


@torch.no_grad()
def prepare_encodings_tmp(messages_template, categories, tokenizer, device, mini_batch_size=12, max_length=None,
                          add_start_token=False,
                          model=None):
    categories_tmp = copy.deepcopy(categories)
    inputs = tokenizer([messages_template[:-1]], return_tensors="pt").to(device)
    outputs = model(**inputs)
    past_key_values = outputs.past_key_values


    result_list = []
    result_dict = {}

    categories = ["\n" + i.strip() for i in categories]
    inputs_cal = tokenizer(categories, return_tensors="pt", padding=True).to(device)
    for idx in range(inputs['input_ids'].shape[0]):
        example_result = []
        for i in range(0, len(categories), mini_batch_size):
            mini_cal_ids = inputs_cal['input_ids'][i:i + mini_batch_size]
            mini_cal_attention_mask = inputs_cal['attention_mask'][i:i + mini_batch_size]
            current_mini_bs = mini_cal_ids.shape[0]
            expand_current_mini = tuple(
                [tuple([d[[idx]].expand(current_mini_bs, -1, -1, -1) for d in dd]) for dd in past_key_values])
            expand_current_mini_attention_mask = inputs['attention_mask'][[idx]].expand(current_mini_bs, -1)
            cal_attention_mask = torch.cat([expand_current_mini_attention_mask,
                                            mini_cal_attention_mask], dim=1)

            cal_outputs = model(input_ids=mini_cal_ids,
                                attention_mask=cal_attention_mask,
                                past_key_values=expand_current_mini)
            mini_logits = cal_outputs.logits
            mini_labels = torch.where(mini_cal_attention_mask == 0, -100, mini_cal_ids)
            shift_logits = mini_logits[..., :-1, :].contiguous()
            shift_labels = mini_labels[..., 1:].contiguous()

            loss_fct = CrossEntropyLoss(reduction='none')
            shift_logits = shift_logits.view(-1, model.config.vocab_size)
            shift_labels = shift_labels.view(-1)

            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            loss = loss.view(cal_attention_mask.shape[0], -1).sum(1) / (mini_cal_attention_mask.sum(1) - 1)
            example_result.append(loss.cpu())
        result_list.append(torch.cat(example_result, dim=0))

    result_list_tmp = copy.deepcopy(result_list)
    result_list_tmp = result_list_tmp[0].tolist()
    for cat_idx, cat in enumerate(categories_tmp):
        result_dict[cat] = result_list_tmp[cat_idx]
    return result_dict


def normalize_ppl(input_dict):
    initial_confidence = np.array([1.0 / val for val in input_dict.values()])
    normalized_confidence = initial_confidence / np.sum(initial_confidence)
    output_dict = {key: val for key, val in zip(input_dict.keys(), normalized_confidence)}
    return output_dict


def cal_loss_simple(scores):
    exp_scores = torch.exp(scores)
    loss = 0
    for i in range(len(scores)):
        sum_exp = torch.sum(exp_scores[i:])
        loss -= (scores[i] - torch.log(sum_exp))
    return loss

def compute_similarity_scores(out, sorted_keys_desc, train_node_idx_index):
    base_vector = out[train_node_idx_index].unsqueeze(0)
    compare_vectors = out[sorted_keys_desc]
    base_vector = base_vector.repeat(compare_vectors.size(0), 1)
    similarity_scores = F.cosine_similarity(base_vector, compare_vectors,
                                            dim=1)
    return similarity_scores



















