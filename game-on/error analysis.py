import pandas as pd
import dgl
from dgl.data import DGLDataset
import dgl.nn.pytorch as dglnn
from torch import nn
import torch
import numpy as np
import csv


def readcsv(fileName):
    '''
    将所有对话的目标语句、上下文语句、讽刺标签、情感标签、情绪标签都集合到uttDict中
    目标语句和上下文语句都在utterance list中，以对话的时间顺序展开，最后是目标语句
    现在返回的utterance是所有的utt，最多的语句是12句，最少的语句是2句
    '''
    with open(fileName, 'r', encoding='latin') as f:
        reader = csv.reader(f)
        uttNameList = []
        # context = np.array()
        for i in reader:
            if i[0] != '':
                uttNameList.append(i[0])
        uttNameList = list(set(uttNameList))
        uttNameList.remove("KEY")
        uttDict = {}
        for name in uttNameList:
            uttDict[name] = {}
            uttDict[name]['utterance'] = []
            uttDict[name]['sarcasm-label'] = ''
            uttDict[name]['sentiment-label'] = ''
            uttDict[name]['emotion-label'] = ''
            uttDict[name]['utt-number'] = ''

    with open(fileName, 'r', encoding='latin') as f1:
        reader = csv.reader(f1)
        for item in reader:
            if item[0] == 'KEY' or item[0] == '':
                continue
            uttDict[item[0]]['sarcasm-label'] = item[4]
            uttDict[item[0]]['sentiment-label'] = item[5]
            uttDict[item[0]]['emotion-label'] = item[7]
            uttDict[item[0]]['utterance'].append(item[2])
            uttDict[item[0]]['utt-number'] = item[0]
    return uttDict, uttNameList


class GraphDataset_Memotion(DGLDataset):
    def __init__(self, df, root_dir, image_vec_dir, text_vec_dir, dataset_name="GraphDataset"):
        """ Create Graph Dataset for Fake News Detection Task

        Args:
            df (pd.DataFrame)
            root_dir (str)
            image_id (str)
            text_id (str)
            image_vec_dir (str)
            text_vec_dir (str)
            dataset_name (str, optional). Defaults to "GraphDataset".
        """
        super(GraphDataset_Memotion, self).__init__(name=dataset_name,
                                           verbose=True)

        ## Main CSV file
        self.df = df

        ## Base data folder
        self.root_dir = root_dir

        ## directory that contains node embeddings for image graph
        self.image_vec_dir = image_vec_dir

        ## directory that contains node embeddings for image graph
        self.text_vec_dir = text_vec_dir

        ## to resize the imagefeature vector from pre-trained model
        self.adaptive_pooling = nn.AdaptiveAvgPool1d(2048)


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """

        Args:
            idx : index of sample to be created

        Returns:
            dgl.graph: multimodal graph for a news post
            torch.tensor(): classification label corresponding to the news post
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.df.iloc[idx]

        ## filenames for the index
        file_name = row.iloc[1].split(".")[0]
        sentence = row.iloc[2]
        # print(file_name)
        # file_name = self.df['image_name'][idx].split(".")[0]
        # print(file_name)

        #### Adding image modality in Graph Dict

        ## Load full image node embedding
        image_vec_full = np.load(f'{self.root_dir}{self.image_vec_dir}{file_name}_full_image.npy')

        ## Load node embeddings for objects present in the image
        try:
            image_vec = np.load(f'{self.root_dir}{self.image_vec_dir}{file_name}.npy')
            all_image_vec = np.concatenate([image_vec_full, image_vec], axis=0)
            # image_vec = all_image_vec
        except:
            image_vec = image_vec_full

        ## Resize the image vectors to match the text embedding dimension
        # image_vec = self.adaptive_pooling(torch.tensor(image_vec).float().unsqueeze(0)).squeeze(0)
        # print(image_vec.shape)
        # 创建图像图
        num_local_image_nodes = image_vec.shape[0]
        num_global_image_nodes = image_vec_full.shape[0]

        # 节点ID
        local_image_node_ids = torch.arange(num_local_image_nodes)
        global_image_node_ids = torch.arange(num_local_image_nodes, num_local_image_nodes + num_global_image_nodes)

        # 边：局部图像特征节点与局部图像特征节点之间的边
        local_image_edges = (torch.arange(num_local_image_nodes).repeat(num_local_image_nodes),
                             torch.arange(num_local_image_nodes).repeat_interleave(num_local_image_nodes))

        # 边：全局图像特征节点与全局图像特征节点之间的边
        global_image_edges = (torch.arange(num_global_image_nodes).repeat(num_global_image_nodes),
                              torch.arange(num_global_image_nodes).repeat_interleave(num_global_image_nodes))

        # 边：局部图像特征节点与全局图像特征节点之间的边
        local_to_global_image_edges = (local_image_node_ids.repeat(num_global_image_nodes),
                                       global_image_node_ids.repeat(num_local_image_nodes))

        # 创建图
        edges = torch.cat([local_image_edges[0], global_image_edges[0], local_to_global_image_edges[0]]), \
            torch.cat([local_image_edges[1], global_image_edges[1], local_to_global_image_edges[1]])

        image_graph = dgl.graph(edges)

        all_image_vec = np.concatenate([image_vec_full, image_vec], axis=0)

        # 添加节点特征
        image_all_vec = torch.cat([
            torch.tensor(image_vec).float(),
            torch.tensor(image_vec_full).float()
        ])

        image_graph.ndata['features'] = image_all_vec

        #### Adding text modality in Graph Dict
        ## Load node embeddings for tokens present in the text
        text_vec = np.load(f'{self.root_dir}{self.text_vec_dir}{file_name}.npy')

        ## Load full image node embedding
        text_vec_full = np.load(f'{self.root_dir}{self.text_vec_dir}{file_name}_full_text.npy')

        # text_vec = self.adaptive_pooling(torch.tensor(text_vec).float().unsqueeze(0)).squeeze(0)
        # print(text_vec.shape)
        # 创建文本图
        num_local_nodes = text_vec.shape[0]
        num_global_nodes = text_vec_full.shape[0]

        # 节点ID
        local_node_ids = torch.arange(num_local_nodes)
        global_node_ids = torch.arange(num_local_nodes, num_local_nodes + num_global_nodes)

        # 边：局部特征节点与局部特征节点之间的边
        local_edges = (torch.arange(num_local_nodes).repeat(num_local_nodes),
                       torch.arange(num_local_nodes).repeat_interleave(num_local_nodes))

        # 边：全局特征节点与全局特征节点之间的边
        global_edges = (torch.arange(num_global_nodes).repeat(num_global_nodes),
                        torch.arange(num_global_nodes).repeat_interleave(num_global_nodes))

        # 边：局部特征节点与全局特征节点之间的边
        local_to_global_edges = (local_node_ids.repeat(num_global_nodes),
                                 global_node_ids.repeat(num_local_nodes))

        # 创建图
        edges = torch.cat([local_edges[0], global_edges[0], local_to_global_edges[0]]), \
            torch.cat([local_edges[1], global_edges[1], local_to_global_edges[1]])

        text_graph = dgl.graph(edges)

        all_text_vec = np.concatenate([text_vec_full, text_vec], axis=0)
        # text_vec = all_text_vec

        # 添加节点特征
        text_all_vec = torch.cat([torch.tensor(text_vec).float(), torch.tensor(text_vec_full).float()], dim=0)

        text_graph.ndata['features'] = text_all_vec

        # text_vec, image_vec, audio_vec 是不同模态的特征
        # 合并所有特征
        all_vec = torch.cat([text_all_vec, image_all_vec], dim=0)

        # 节点数量
        num_text_nodes = text_all_vec.shape[0]
        num_image_nodes = image_all_vec.shape[0]

        # 计算图像和音频节点的索引偏移
        image_offset = num_text_nodes

        # 1. 创建自模态边（不需要改变）
        text_edges = (torch.arange(num_text_nodes).repeat(num_text_nodes),
                      torch.arange(num_text_nodes).repeat_interleave(num_text_nodes))

        image_edges = (torch.arange(num_image_nodes).repeat(num_image_nodes) + image_offset,
                       torch.arange(num_image_nodes).repeat_interleave(num_image_nodes) + image_offset)

        # 2. 创建跨模态边，并为非文本模态增加偏移
        text_to_image_edges = (torch.arange(num_text_nodes).unsqueeze(1).repeat(1, num_image_nodes).view(-1),
                               torch.arange(num_image_nodes).unsqueeze(0).repeat(num_text_nodes, 1).view(
                                   -1) + image_offset)

        # 3. 合并所有边
        u = torch.cat([text_edges[0], image_edges[0], text_to_image_edges[0]])

        v = torch.cat([text_edges[1], image_edges[1], text_to_image_edges[1]])

        # 4. 创建图，包含所有节点和边
        g = dgl.graph((u, v), num_nodes=num_text_nodes + num_image_nodes)

        # 添加节点特征
        g.ndata['features'] = all_vec

        ## find the label
        sarcasm = row['sarcasm']
        sentiment = row['overall_sentiment']
        emotion = row['humour']

        return g, text_graph, image_graph, sarcasm, file_name, sentence

class GraphDataset_Mustard(DGLDataset):
    def __init__(self, datatye, root_dir, image_vec_dir, text_vec_dir, audio_vec_dir, dataset_name="GraphDataset"):
        super(GraphDataset_Mustard, self).__init__(name=dataset_name, verbose=True)
        self.datatye = datatye
        if self.datatye == 'train':
            datafile = "/home/zxl/MultiTask classification/M2Seq2Seq-master/mustard-dataset-train.csv"
        elif self.datatye == 'dev':
            datafile = "/home/zxl/MultiTask classification/M2Seq2Seq-master/mustard-dataset-dev.csv"
        elif self.datatye == 'test':
            datafile = "/home/zxl/MultiTask classification/M2Seq2Seq-master/mustard-dataset-test.csv"
        self.uttDict, self.uttNameList = readcsv(datafile)
        self.uttList = list(self.uttDict.values())

        self.root_dir = root_dir
        self.image_vec_dir = image_vec_dir
        self.text_vec_dir = text_vec_dir
        self.audio_vec_dir = audio_vec_dir
        self.adaptive_pooling = nn.AdaptiveAvgPool1d(2048)

        self.graphs = []
        self.text_graph = []
        self.image_graph = []
        self.audio_graph = []

        for idx in range(len(self.uttNameList)):
            uttName = self.uttNameList[idx]
            text = list(self.uttDict[uttName]['utterance'])
            # print(len(text))
            filename_image = f'{self.root_dir}{self.image_vec_dir}' + uttName
            filename_text = f'{self.root_dir}{self.text_vec_dir}' + uttName
            filename_audio = f'{self.root_dir}{self.audio_vec_dir}' + datatye + '/' + uttName
            # 初始化全局节点特征和边的容器
            all_image_vecs = []
            all_image_edges_src = []
            all_image_edges_dst = []

            all_audio_edge_src = []
            all_audio_edge_dst = []
            all_audio_vecs = []

            all_text_vecs = []
            all_text_edges_src = []
            all_text_edges_dst = []

            image_node_offset = 0  # 用来记录当前图的节点ID偏移
            text_node_offset = 0
            audio_node_offset = 0
            for i in range(len(text)):
                image_vec_full = np.load(f"{filename_image}_{i}_full_image.npy")
                image_vec_full = image_vec_full.astype(np.float32)  # Ensure correct type

                try:
                    image_vec = np.load(f"{filename_image}_{i}.npy")
                    image_vec = image_vec.astype(np.float32)  # Ensure correct type
                    # all_image_vec = np.concatenate([image_vec_full, image_vec], axis=0)
                    # image_vec = all_image_vec
                except FileNotFoundError:
                    image_vec = image_vec_full

                # 创建图像图
                num_local_image_nodes = image_vec.shape[0]
                num_global_image_nodes = image_vec_full.shape[0]

                # 节点ID (带有偏移)
                local_image_node_ids = torch.arange(num_local_image_nodes) + image_node_offset
                global_image_node_ids = torch.arange(num_local_image_nodes,
                                                     num_local_image_nodes + num_global_image_nodes) + image_node_offset

                # 边：局部图像特征节点与局部图像特征节点之间的边
                local_image_edges = (local_image_node_ids.repeat(num_local_image_nodes),
                                     local_image_node_ids.repeat_interleave(num_local_image_nodes))

                # 边：全局图像特征节点与全局图像特征节点之间的边
                global_image_edges = (global_image_node_ids.repeat(num_global_image_nodes),
                                      global_image_node_ids.repeat_interleave(num_global_image_nodes))

                # 边：局部图像特征节点与全局图像特征节点之间的边
                local_to_global_image_edges = (local_image_node_ids.repeat(num_global_image_nodes),
                                               global_image_node_ids.repeat(num_local_image_nodes))

                # 累积边 (将每次循环生成的边存入全局边容器)
                all_image_edges_src.append(
                    torch.cat([local_image_edges[0], global_image_edges[0], local_to_global_image_edges[0]]))
                all_image_edges_dst.append(
                    torch.cat([local_image_edges[1], global_image_edges[1], local_to_global_image_edges[1]]))

                all_image_vec = np.concatenate([image_vec_full, image_vec], axis=0)
                image_vec = all_image_vec

                # 累积节点特征 (将每次循环生成的节点特征存入全局特征容器)
                all_image_vecs.append(torch.tensor(image_vec).float())

                # 更新偏移量 (确保下一个图的节点ID不重叠)
                image_node_offset += num_local_image_nodes + num_global_image_nodes

                # image_vec = self.adaptive_pooling(torch.tensor(image_vec).float().unsqueeze(0)).squeeze(0)
                # print(image_vec.shape)

                text_feature_full = np.load(f"{filename_text}_{i}_full_text.npy")
                text_vec_full = text_feature_full.astype(np.float32)  # 确保类型正确

                text_vec = np.load(f"{filename_text}_{i}.npy")
                text_vec = text_vec.astype(np.float32)  # 确保类型正确

                # 将局部和全局文本特征拼接
                # all_text_vec = np.concatenate([text_vec_full, text_vec], axis=0)
                # text_vec = all_text_vec

                # 创建文本图
                num_local_nodes = text_vec.shape[0]
                num_global_nodes = text_vec_full.shape[0]

                # 节点ID（带有偏移）
                local_node_ids = torch.arange(num_local_nodes) + text_node_offset
                global_node_ids = torch.arange(num_local_nodes, num_local_nodes + num_global_nodes) + text_node_offset

                # 边：局部特征节点与局部特征节点之间的边
                local_edges = (local_node_ids.repeat(num_local_nodes),
                               local_node_ids.repeat_interleave(num_local_nodes))

                # 边：全局特征节点与全局特征节点之间的边
                global_edges = (global_node_ids.repeat(num_global_nodes),
                                global_node_ids.repeat_interleave(num_global_nodes))

                # 边：局部特征节点与全局特征节点之间的边
                local_to_global_edges = (local_node_ids.repeat(num_global_nodes),
                                         global_node_ids.repeat(num_local_nodes))

                # 累积边 (将每次循环生成的边存入全局边容器)
                all_text_edges_src.append(torch.cat([local_edges[0], global_edges[0], local_to_global_edges[0]]))
                all_text_edges_dst.append(torch.cat([local_edges[1], global_edges[1], local_to_global_edges[1]]))

                # 将局部和全局文本特征拼接
                all_text_vec = np.concatenate([text_vec_full, text_vec], axis=0)
                text_vec = all_text_vec

                # 累积节点特征 (将每次循环生成的节点特征存入全局特征容器)
                all_text_vecs.append(torch.tensor(text_vec).float())

                # 更新偏移量 (确保下一个图的节点ID不重叠)
                text_node_offset += num_local_nodes + num_global_nodes


            # 最后，连接所有的边和节点特征，构建全局图
            image_edges_src = torch.cat(all_image_edges_src)
            image_edges_dst = torch.cat(all_image_edges_dst)
            image_all_edges = (image_edges_src, image_edges_dst)

            # 将所有图的节点特征连接在一起
            image_all_features = torch.cat(all_image_vecs)

            # 创建最终的全局图
            global_image_graph = dgl.graph(image_all_edges)

            # 添加全局节点特征
            global_image_graph.ndata['features'] = image_all_features

            self.image_graph.append(global_image_graph)  # 将最终图保存下来

            # 最后，连接所有的边和节点特征，构建全局文本图
            edges_src = torch.cat(all_text_edges_src)
            edges_dst = torch.cat(all_text_edges_dst)
            all_edges = (edges_src, edges_dst)

            # 将所有图的节点特征连接在一起
            all_features = torch.cat(all_text_vecs)

            # 创建最终的全局图，注意节点数量
            global_text_graph = dgl.graph(all_edges)

            # 添加全局节点特征
            global_text_graph.ndata['features'] = all_features

            self.text_graph.append(global_text_graph)  # 将最终图保存下来

            audio_vec_full = np.loadtxt(
                filename_audio,
                dtype=float,
                delimiter=",")
            audio_vec_full = audio_vec_full.astype(np.float32)
            # 创建音频图
            num_audio_nodes = audio_vec_full.shape[0]
            audio_node_ids = torch.arange(num_audio_nodes) + audio_node_offset

            # 边：音频特征节点与自身之间的边
            audio_edges = (audio_node_ids.repeat(num_audio_nodes),
                           audio_node_ids.repeat_interleave(num_audio_nodes))

            # 累积边
            all_audio_edge_src.append(audio_edges[0])
            all_audio_edge_dst.append(audio_edges[1])

            # 累积音频特征
            all_audio_vecs.append(torch.tensor(audio_vec_full).float())

            # 更新偏移量
            audio_node_offset += num_audio_nodes

            # 创建全局音频图
            audio_edges_src = torch.cat(all_audio_edge_src)
            audio_edges_dst = torch.cat(all_audio_edge_dst)
            audio_all_edges = (audio_edges_src, audio_edges_dst)
            audio_all_features = torch.cat(all_audio_vecs)

            global_audio_graph = dgl.graph(audio_all_edges)
            global_audio_graph.ndata['features'] = audio_all_features

            self.audio_graph.append(global_audio_graph)

            # 假设 text_vec, image_vec, audio_vec 是不同模态的特征
            text_vec = torch.tensor(all_features).float()
            image_vec = torch.tensor(image_all_features).float()
            audio_vec = torch.tensor(audio_vec_full).float()

            # 合并所有特征
            all_vec = torch.cat([text_vec, image_vec, audio_vec], dim=0)
            # print(all_vec.shape)

            # 节点数量
            num_text_nodes = text_vec.shape[0]
            num_image_nodes = image_vec.shape[0]
            num_audio_nodes = audio_vec.shape[0]

            # 计算图像和音频节点的索引偏移
            image_offset = num_text_nodes
            audio_offset = num_text_nodes + num_image_nodes

            # 1. 创建自模态边（不需要改变）
            text_edges = (torch.arange(num_text_nodes).repeat(num_text_nodes),
                          torch.arange(num_text_nodes).repeat_interleave(num_text_nodes))

            image_edges = (torch.arange(num_image_nodes).repeat(num_image_nodes) + image_offset,
                           torch.arange(num_image_nodes).repeat_interleave(num_image_nodes) + image_offset)

            audio_edges = (torch.arange(num_audio_nodes).repeat(num_audio_nodes) + audio_offset,
                           torch.arange(num_audio_nodes).repeat_interleave(num_audio_nodes) + audio_offset)

            # 2. 创建跨模态边，并为非文本模态增加偏移
            text_to_image_edges = (torch.arange(num_text_nodes).unsqueeze(1).repeat(1, num_image_nodes).view(-1),
                                   torch.arange(num_image_nodes).unsqueeze(0).repeat(num_text_nodes, 1).view(
                                       -1) + image_offset)

            text_to_audio_edges = (torch.arange(num_text_nodes).unsqueeze(1).repeat(1, num_audio_nodes).view(-1),
                                   torch.arange(num_audio_nodes).unsqueeze(0).repeat(num_text_nodes, 1).view(
                                       -1) + audio_offset)

            image_to_audio_edges = (
                torch.arange(num_image_nodes).unsqueeze(1).repeat(1, num_audio_nodes).view(-1) + image_offset,
                torch.arange(num_audio_nodes).unsqueeze(0).repeat(num_image_nodes, 1).view(-1) + audio_offset)

            # 3. 合并所有边
            u = torch.cat([text_edges[0], image_edges[0], audio_edges[0],
                           text_to_image_edges[0], text_to_audio_edges[0], image_to_audio_edges[0]])

            v = torch.cat([text_edges[1], image_edges[1], audio_edges[1],
                           text_to_image_edges[1], text_to_audio_edges[1], image_to_audio_edges[1]])

            # 4. 创建图，包含所有节点和边
            g = dgl.graph((u, v), num_nodes=num_text_nodes + num_image_nodes + num_audio_nodes)

            # 添加节点特征
            g.ndata['features'] = all_vec

            self.graphs.append(g)


    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        uttName = self.uttNameList[idx]
        text = list(self.uttDict[uttName]['utterance'])
        sarcasmStr = self.uttList[idx]['sarcasm-label']
        sentimentStr = self.uttList[idx]['sentiment-label']
        emotionStr = self.uttList[idx]['emotion-label']

        if 'True' == sarcasmStr:
            sarcasmLabel = np.array([0, 1], dtype=np.int8)
        else:
            sarcasmLabel = np.array([1, 0], dtype=np.int8)

        sentimentLabel = np.zeros(3, dtype=np.int8)
        if -1 == int(sentimentStr):
            sentimentLabel[0] = 1
        elif 0 == int(sentimentStr):
            sentimentLabel[1] = 1
        else:
            sentimentLabel[2] = 1
        emotionLabel = np.zeros(9, dtype=np.int8)
        emotionLabel[int(emotionStr.split(',')[0]) - 1] = 1

        return self.graphs[idx], self.text_graph[idx], self.image_graph[idx], self.audio_graph[idx], sarcasmLabel, uttName, text
