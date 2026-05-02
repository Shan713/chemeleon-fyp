# https://github.com/jiaor17/DiffCSP
import math
from collections import namedtuple

import torch
import torch.nn as nn

from torch_geometric.utils import dense_to_sparse

from chemeleon.utils.data_utils import (
    radius_graph_pbc,
    repeat_blocks,
)
from chemeleon.utils.scatter import scatter_mean

DECODER_OUTPUTS = namedtuple(
    "DECODER_OUTPUTS", ["atom_types_out", "lattice_out", "coords_out", "node_features"]
)


class SinusoidalTimeEmbeddings(nn.Module):
    """Attention is all you need."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class SinusoidsEmbedding(nn.Module):
    "Embedding for periodic distance features."

    def __init__(self, n_frequencies=10, n_space=3):
        super().__init__()
        self.n_frequencies = n_frequencies
        self.n_space = n_space
        self.frequencies = 2 * math.pi * torch.arange(self.n_frequencies)
        self.dim = self.n_frequencies * 2 * self.n_space

    def forward(self, x):
        emb = x.unsqueeze(-1) * self.frequencies[None, None, :].to(x.device)
        emb = emb.reshape(-1, self.n_frequencies * self.n_space)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.detach()


class FilmLayer(nn.Module):
    """FiLM layer for efficient incorporation of time or conditional embeddings."""

    def __init__(
        self,
        hidden_dim=128,
        time_dim=256,
        text_dim=128,
        act_fn=nn.SiLU(),
    ):
        super(FilmLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        self.text_dim = text_dim
        self.act_fn = act_fn
        self.mlp_cond = nn.Sequential(
            nn.Linear(time_dim + text_dim, hidden_dim * 2),
            act_fn,
        )
        # block
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, time_embeds=None, text_embeds=None):
        # Feature-wise Linear Modulation
        if time_embeds is not None and text_embeds is not None:
            cond_emb = self.mlp_cond(torch.cat([time_embeds, text_embeds], dim=1))
        elif time_embeds is not None:
            cond_emb = self.mlp_cond(time_embeds)
        elif text_embeds is not None:
            cond_emb = self.mlp_cond(text_embeds)
        else:
            raise ValueError("Either time or text embeddings must be provided.")
        scale, shift = cond_emb.chunk(2, dim=1)

        # residual block
        x_init = x.clone()
        x = self.proj(x)
        x = self.norm(x)
        x = x * scale + shift
        x = self.act_fn(x)
        x = x + x_init
        return x


class CSPLayer(nn.Module):
    """Message passing layer for cspnet."""

    def __init__(
        self, hidden_dim=128, act_fn=nn.SiLU(), dis_emb=None, ln=False, ip=True
    ):
        super(CSPLayer, self).__init__()

        self.dis_dim = 3
        self.dis_emb = dis_emb
        self.ip = ip
        if dis_emb is not None:
            self.dis_dim = dis_emb.dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 9 + self.dis_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
        )
        self.ln = ln
        if self.ln:
            self.layer_norm = nn.LayerNorm(hidden_dim)

    def edge_model(
        self,
        node_features,
        frac_coords,
        lattices,
        edge_index,
        edge2graph,
        frac_diff=None,
    ):
        hi, hj = node_features[edge_index[0]], node_features[edge_index[1]]
        if frac_diff is None:
            xi, xj = frac_coords[edge_index[0]], frac_coords[edge_index[1]]
            frac_diff = (xj - xi) % 1.0
        if self.dis_emb is not None:
            frac_diff = self.dis_emb(frac_diff)  # fourier transform
        if self.ip:
            lattice_ips = lattices @ lattices.transpose(-1, -2)
        else:
            lattice_ips = lattices
        lattice_ips_flatten = lattice_ips.view(-1, 9)
        lattice_ips_flatten_edges = lattice_ips_flatten[edge2graph]
        edges_input = torch.cat([hi, hj, lattice_ips_flatten_edges, frac_diff], dim=1)
        edge_features = self.edge_mlp(edges_input)
        return edge_features

    def node_model(self, node_features, edge_features, edge_index):
        agg = scatter_mean(
            edge_features,
            edge_index[0],
            dim=0,
            dim_size=node_features.shape[0],
        )
        agg = torch.cat([node_features, agg], dim=1)
        out = self.node_mlp(agg)
        return out

    def forward(
        self,
        node_features,
        frac_coords,
        lattices,
        edge_index,
        edge2graph,
        frac_diff=None,
    ):
        node_input = node_features
        if self.ln:
            node_features = self.layer_norm(node_input)
        edge_features = self.edge_model(
            node_features, frac_coords, lattices, edge_index, edge2graph, frac_diff
        )
        node_output = self.node_model(node_features, edge_features, edge_index)
        return node_input + node_output


class CSPNet(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        time_dim=256,
        text_dim=128,
        num_layers=4,
        max_atoms=103,
        act_fn="silu",
        dis_emb="sin",
        num_freqs=10,
        edge_style="fc",
        cutoff=6.0,
        max_neighbors=20,
        ln=False,
        ip=True,
        smooth=True,
        pred_atom_types=True,
    ):
        super(CSPNet, self).__init__()

        self.ip = ip
        self.smooth = smooth
        if self.smooth:
            self.node_embedding = nn.Linear(max_atoms, hidden_dim)
        else:
            self.node_embedding = nn.Embedding(max_atoms, hidden_dim)
        if time_dim > 0 or text_dim > 0:
            self.film_layer = FilmLayer(hidden_dim, time_dim, text_dim)
        if act_fn == "silu":
            self.act_fn = nn.SiLU()
        if dis_emb == "sin":
            self.dis_emb = SinusoidsEmbedding(n_frequencies=num_freqs)
        elif dis_emb == "none":
            self.dis_emb = None
        for i in range(0, num_layers):
            self.add_module(
                f"csp_layer_{i}",
                CSPLayer(hidden_dim, self.act_fn, self.dis_emb, ln=ln, ip=ip),
            )
        self.num_layers = num_layers
        self.coord_out = nn.Linear(hidden_dim, 3, bias=False)
        self.lattice_out = nn.Linear(hidden_dim, 9, bias=False)
        self.type_out = nn.Linear(hidden_dim, max_atoms)
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.ln = ln
        self.edge_style = edge_style
        if self.ln:
            self.final_layer_norm = nn.LayerNorm(hidden_dim)
        self.pred_atom_types = pred_atom_types

    def select_symmetric_edges(self, tensor, mask, reorder_idx, inverse_neg):
        # Mask out counter-edges
        tensor_directed = tensor[mask]
        # Concatenate counter-edges after normal edges
        sign = 1 - 2 * inverse_neg
        tensor_cat = torch.cat([tensor_directed, sign * tensor_directed])
        # Reorder everything so the edges of every image are consecutive
        tensor_ordered = tensor_cat[reorder_idx]
        return tensor_ordered

    def reorder_symmetric_edges(self, edge_index, cell_offsets, neighbors, edge_vector):
        """
        Reorder edges to make finding counter-directional edges easier.

        Some edges are only present in one direction in the data,
        since every atom has a maximum number of neighbors. Since we only use i->j
        edges here, we lose some j->i edges and add others by
        making it symmetric.
        We could fix this by merging edge_index with its counter-edges,
        including the cell_offsets, and then running torch.unique.
        But this does not seem worth it.
        """

        # Generate mask
        mask_sep_atoms = edge_index[0] < edge_index[1]
        # Distinguish edges between the same (periodic) atom by ordering the cells
        cell_earlier = (
            (cell_offsets[:, 0] < 0)
            | ((cell_offsets[:, 0] == 0) & (cell_offsets[:, 1] < 0))
            | (
                (cell_offsets[:, 0] == 0)
                & (cell_offsets[:, 1] == 0)
                & (cell_offsets[:, 2] < 0)
            )
        )
        mask_same_atoms = edge_index[0] == edge_index[1]
        mask_same_atoms &= cell_earlier
        mask = mask_sep_atoms | mask_same_atoms

        # Mask out counter-edges
        edge_index_new = edge_index[mask[None, :].expand(2, -1)].view(2, -1)

        # Concatenate counter-edges after normal edges
        edge_index_cat = torch.cat(
            [
                edge_index_new,
                torch.stack([edge_index_new[1], edge_index_new[0]], dim=0),
            ],
            dim=1,
        )

        # Count remaining edges per image
        batch_edge = torch.repeat_interleave(
            torch.arange(neighbors.size(0), device=edge_index.device),
            neighbors,
        )
        batch_edge = batch_edge[mask]
        neighbors_new = 2 * torch.bincount(batch_edge, minlength=neighbors.size(0))

        # Create indexing array
        edge_reorder_idx = repeat_blocks(
            neighbors_new // 2,
            repeats=2,
            continuous_indexing=True,
            repeat_inc=edge_index_new.size(1),
        )

        # Reorder everything so the edges of every image are consecutive
        edge_index_new = edge_index_cat[:, edge_reorder_idx]
        cell_offsets_new = self.select_symmetric_edges(
            cell_offsets, mask, edge_reorder_idx, True
        )
        edge_vector_new = self.select_symmetric_edges(
            edge_vector, mask, edge_reorder_idx, True
        )

        return (
            edge_index_new,
            cell_offsets_new,
            neighbors_new,
            edge_vector_new,
        )

    def gen_edges(self, num_atoms, frac_coords, lattices, node2graph):
        if self.edge_style == "fc":
            lis = [torch.ones(n, n, device=num_atoms.device) for n in num_atoms]
            fc_graph = torch.block_diag(*lis)
            fc_edges, _ = dense_to_sparse(fc_graph)
            return fc_edges, (frac_coords[fc_edges[1]] - frac_coords[fc_edges[0]]) % 1.0
        elif self.edge_style == "knn":
            lattice_nodes = lattices[node2graph]
            cart_coords = torch.einsum("bi,bij->bj", frac_coords, lattice_nodes)

            edge_index, to_jimages, num_bonds = radius_graph_pbc(
                pos=cart_coords,
                cell=lattices,
                natoms=num_atoms,
                max_num_neighbors_threshold=self.max_neighbors,
            )
            j_index, i_index = edge_index
            distance_vectors = frac_coords[j_index] - frac_coords[i_index]
            distance_vectors += to_jimages.float()

            edge_index_new, _, _, edge_vector_new = self.reorder_symmetric_edges(
                edge_index, to_jimages, num_bonds, distance_vectors
            )

            return edge_index_new, -edge_vector_new

    def forward(
        self,
        atom_types,
        frac_coords,
        lattices,
        num_atoms,
        node2graph,
        t=None,  # time embeddings can be omitted when training contrastive learning
        text_embeds=None,
    ):
        edges, frac_diff = self.gen_edges(num_atoms, frac_coords, lattices, node2graph)
        edge2graph = node2graph[edges[0]]
        node_features = self.node_embedding(atom_types)

        if t is not None:
            t_per_atom = t.repeat_interleave(num_atoms, dim=0)  # [B_n, time_dim]
        else:
            t_per_atom = None

        if text_embeds is not None:
            text_per_atom = text_embeds.repeat_interleave(
                num_atoms, dim=0
            )  # [B_n, text_dim]
        else:
            text_per_atom = None

        for i in range(0, self.num_layers):
            if t_per_atom is not None or text_per_atom is not None:
                node_features = self.film_layer(
                    node_features, time_embeds=t_per_atom, text_embeds=text_per_atom
                )
            node_features = self._modules[f"csp_layer_{i}"](
                node_features,
                frac_coords,
                lattices,
                edges,
                edge2graph,
                frac_diff=frac_diff,
            )

        if self.ln:
            node_features = self.final_layer_norm(node_features)

        coord_out = self.coord_out(node_features)

        graph_features = scatter_mean(node_features, node2graph, dim=0)
        lattice_out = self.lattice_out(graph_features)
        lattice_out = lattice_out.view(-1, 3, 3)
        if self.ip:
            lattice_out = torch.einsum("bij,bjk->bik", lattice_out, lattices)
        if self.pred_atom_types:
            type_out = self.type_out(node_features)
        else:
            type_out = None

        return DECODER_OUTPUTS(
            atom_types_out=type_out,  # [B_n, max_atoms]
            lattice_out=lattice_out,  # [B, 3, 3]
            coords_out=coord_out,  # [B_n, 3]
            node_features=node_features,  # [B_n, hidden_dim]
        )
