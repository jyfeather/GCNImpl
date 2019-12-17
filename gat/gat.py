"""
Graph Attention Network
DGL implementations: https://docs.dgl.ai/en/latest/tutorials/models/1_gnn/9_gat.html
"""

from dgl import DGLGraph
from dgl.data import citation_graph as citegrh
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np


class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.g = g
        self.W = nn.Linear(in_dim, out_dim, bias=False)  # W \in R^{F,F'}
        self.a = nn.Linear(2*out_dim, 1, bias=False)  # a \in R^{2F'}

    def edge_attention(self, edges):
        """
        calculate edge attention coefficients, Equation 1
        """
        z1 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        e = self.a(z1)
        return {'e': F.leaky_relu(e)}

    def message_func(self, edges):
        """
        send message
        """
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        """
        Equation 2,3,4
        """
        alpha = torch.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, h):
        z = self.W(h)
        self.g.ndata['z'] = z
        self.g.apply_edges(self.edge_attention)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')


class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim))
        self.merge = merge

    def forward(self, h):
        head_outputs = [attn_head(h) for attn_head in self.heads]
        if self.merge == 'cat':
            return torch.cat(head_outputs, dim=1)
        else:
            return torch.mean(torch.stack(head_outputs))


class GAT(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, num_heads):
        super(GAT, self).__init__()
        # Equation 5 or Figure 1 right part in the Paper
        # concatenate/average multi head hidden embeddings
        self.layer1 = MultiHeadGATLayer(g, in_dim, hidden_dim, num_heads)
        self.layer2 = MultiHeadGATLayer(g, hidden_dim*num_heads, out_dim, 1)

    def forward(self, h):
        h = self.layer1(h)
        h = F.elu(h)
        h = self.layer2(h)
        return h


def load_cora_data():
    data = citegrh.load_cora()
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    mask = torch.ByteTensor(data.train_mask)
    g = data.graph
    """
    add self loop
    """
    g.remove_edges_from(nx.selfloop_edges(g))
    g = DGLGraph(g)
    g.add_edges(g.nodes(), g.nodes())
    return g, features, labels, mask


if __name__ == '__main__':
    g, features, labels, mask = load_cora_data()

    net = GAT(
        g,
        in_dim=features.size()[1],
        hidden_dim=8,
        out_dim=7,
        num_heads=2
    )

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    # main loop
    dur = []
    for epoch in range(30):
        if epoch >= 3:
            t0 = time.time()

        logits = net(features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[mask], labels[mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        print("Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f}".format(epoch, loss.item(), np.mean(dur)))


