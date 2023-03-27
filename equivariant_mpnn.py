import torch 
from utils import *
from models import *
from torch import nn

from sklearn.metrics import pairwise_distances


class EquivariantMPNNLayer(MessagePassing):
    def __init__(self, emb_dim=64, edge_dim=4, aggr='add'):
        """Message Passing Neural Network Layer

        This layer is equivariant to 3D rotations and translations.

        Args:
            emb_dim: (int) - hidden dimension `d`
            edge_dim: (int) - edge feature dimension `d_e`
            aggr: (str) - aggregation function `\oplus` (sum/mean/max)
        """
        # Set the aggregation function
        super().__init__(aggr=aggr)

        self.emb_dim = emb_dim
        self.edge_dim = edge_dim
        self.coord_dim = 3

        # ============ YOUR CODE HERE ==============
        # Define the MLPs constituting your new layer.
        # At the least, you will need `\psi` and `\phi` 
        # (but their definitions may be different from what
        # we used previously).
        #
        self.mlp_msg = Sequential(
            Linear(2*emb_dim + edge_dim + 1, emb_dim), BatchNorm1d(emb_dim), ReLU(),
            Linear(emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU()
          )
        
        self.mlp_upd = Sequential(
            Linear(2*emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU(), 
            Linear(emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU()
        )

        self.mlp_msg_coord = Sequential(
            Linear(self.coord_dim + 1, self.coord_dim), BatchNorm1d(self.coord_dim), ReLU(),
            Linear(self.coord_dim, self.coord_dim), BatchNorm1d(self.coord_dim), ReLU()
          )

        self.mlp_upd_coord = Sequential(
            Linear(2*self.coord_dim, self.coord_dim), BatchNorm1d(self.coord_dim), ReLU(), 
            Linear(self.coord_dim, self.coord_dim), BatchNorm1d(self.coord_dim), ReLU()
        )


    def get_pdist(self, a, b, index):

        pdist = nn.PairwiseDistance(p=index)
        return torch.unsqueeze(pdist(a, b),1)

    def forward(self, h, pos, edge_index, edge_attr):
        """
        The forward pass updates node features `h` via one round of message passing.

        Args:
            h: (n, d) - initial node features
            pos: (n, 3) - initial node coordinates
            edge_index: (e, 2) - pairs of edges (i, j)
            edge_attr: (e, d_e) - edge features

        Returns:
            out: [(n, d),(n,3)] - updated node features
        """
        # ============ YOUR CODE HERE ==============
        # Notice that the `forward()` function has a new argument 
        # `pos` denoting the initial node coordinates. Your task is
        # to update the `propagate()` function in order to pass `pos`
        # to the `message()` function along with the other arguments.
        #
        out = self.propagate(edge_index, h=h, edge_attr=edge_attr, pos=pos)
        return out
       


    def message(self, h_i, h_j, edge_attr, pos_i, pos_j):
        dist_2 = self.get_pdist(pos_i, pos_j, 2)
        dist_1 = self.get_pdist(pos_i, pos_j, 1)
        # print(pos_i)
        
        # print(dist_1)
        msg = torch.cat([h_i, h_j, edge_attr, dist_1], dim=-1)
        # pos_msg = torch.cat([pos_j, edge_attr, dist_1], dim=-1)
        pos_msg = torch.cat([pos_j, dist_1], dim=-1)
 
        return  self.mlp_msg(msg), self.mlp_msg_coord(pos_msg)
    
    def aggregate(self, inputs, index):

        return scatter(inputs[0], index, dim=self.node_dim, reduce=self.aggr), scatter(inputs[1], index, dim=self.node_dim, reduce=self.aggr) 
    
    def update(self, aggr_out, h, pos):

        upd_out = torch.cat([h, aggr_out[0]], dim=-1)
        upd_out_coord = torch.cat([pos, aggr_out[1]], dim=-1)

        return self.mlp_upd(upd_out), self.mlp_upd_coord(upd_out_coord)
    
    

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})')


class FinalMPNNModel(MPNNModel):
    def __init__(self, num_layers=4, emb_dim=64, in_dim=11, edge_dim=4, out_dim=1):
        """Message Passing Neural Network model for graph property prediction

        This model uses both node features and coordinates as inputs, and
        is invariant to 3D rotations and translations (the constituent MPNN layers
        are equivariant to 3D rotations and translations).

        Args:
            num_layers: (int) - number of message passing layers `L`
            emb_dim: (int) - hidden dimension `d`
            in_dim: (int) - initial node feature dimension `d_n`
            edge_dim: (int) - edge feature dimension `d_e`
            out_dim: (int) - output dimension (fixed to 1)
        """
        super().__init__()
        
        # Linear projection for initial node features
        # dim: d_n -> d
        self.lin_in = Linear(in_dim, emb_dim)
        
        # Stack of MPNN layers
        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(EquivariantMPNNLayer(emb_dim, edge_dim, aggr='add'))
        
        # Global pooling/readout function `R` (mean pooling)
        # PyG handles the underlying logic via `global_mean_pool()`
        self.pool = global_mean_pool

        # Linear prediction head
        # dim: d -> out_dim
        self.lin_pred = Linear(emb_dim, out_dim)
        
    def forward(self, data):
        """
        Args:
            data: (PyG.Data) - batch of PyG graphs

        Returns: 
            out: (batch_size, out_dim) - prediction for each graph
        """
        h = self.lin_in(data.x) # (n, d_n) -> (n, d)
        pos = data.pos
        
        for conv in self.convs:
            # Message passing layer
            h_update, pos_update = conv(h, pos, data.edge_index, data.edge_attr)
            
            # Update node features
            h = h + h_update # (n, d) -> (n, d)
            # Note that we add a residual connection after each MPNN layer
            
            # Update node coordinates
            pos = pos_update # (n, 3) -> (n, 3)

        h_graph = self.pool(h, data.batch) # (n, d) -> (batch_size, d)

        out = self.lin_pred(h_graph) # (batch_size, d) -> (batch_size, 1)

        return out.view(-1)