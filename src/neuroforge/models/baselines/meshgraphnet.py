"""MeshGraphNet (Pfaff et al., ICLR 2021) baseline for AirfRANS — point-cloud GNN.

This is a faithful, pure-PyTorch implementation of the Encode-Process-Decode
graph-network surrogate from

    Pfaff, Fortunato, Sanchez-Gonzalez, Battaglia. "Learning Mesh-Based
    Simulation with Graph Networks." ICLR 2021 (DeepMind).
    https://arxiv.org/abs/2010.03409

It is the THIRD backbone family in the NeuroForge comparison (after Transolver =
attention/slice-token and FNO = spectral), giving a *message-passing* point of
comparison. Two faithfulness notes:

* **Mesh-free graph.** The original MeshGraphNet runs on a simulation mesh and
  uses the mesh edges. AirfRANS gives us a raw point cloud, not a CFD mesh, and
  the whole NeuroForge premise is to stay mesh-free, so we build the graph as a
  fixed-``k`` **kNN over the point cloud** (see :func:`knn_edges`). The
  Encode-Process-Decode architecture, the Battaglia-style GN message-passing
  block (separate edge and node updates with residual + LayerNorm), and the
  relative-position edge features are reproduced as in the paper; only the edge
  *set* is kNN rather than a generated mesh. We do NOT add the world-edge / mesh-
  edge split (single-edge-type graph) since there is one geometric graph.
* **No new deps.** Pure ``torch`` — no ``torch_geometric`` / ``torch_scatter`` /
  ``torch_cluster`` (fragile on Windows/CUDA, not installed). Scatter-aggregation
  is done with ``torch.zeros(...).index_add_(...)``.

Why it is NOT in the frozen ``models.base`` registry
----------------------------------------------------
Same reasoning as ``transolver.py``: the registry's contract is grid-shaped
``(B, C, H, W) -> (B, C, H, W)`` solvers, whereas a MeshGraphNet maps a point
cloud ``(B, N, F_IN) -> (B, N, F_OUT)`` and additionally needs an edge set.
Keeping it standalone avoids misrepresenting the frozen grid contract (see
``docs/baseline_plan.md`` §3).

Edge-feature convention (built by the CALLER, consumed by the model)
--------------------------------------------------------------------
``edge_attr`` has ``edge_in = 3`` columns, built from the **RAW** positions::

    rel = pos[dst] - pos[src]                         # (E, 2)
    edge_attr = [ rel / L_xy , ||rel|| / L_norm ]     # (E, 3)

where ``L_xy`` is a per-axis length scale (we use the train ``PointNormalizer``
``std_in[:2]``, since input features [0:2] are the raw x,y) and
``L_norm = ||L_xy||``. The graph and these features MUST be built from the raw
``.pos`` — never the z-scored in-feature coords, since z-scoring is anisotropic
and distorts neighbour distances. The model just consumes the already-scaled
``edge_attr``; see :func:`edge_features`.

Memory note (the full-run gotcha)
---------------------------------
The bottleneck at whole-cloud eval (~180k pts, k=8 symmetric -> ~2.9M edges) is
the *edge* update, not the nodes: naively forming ``[e, h_src, h_dst]`` is an
``(E, 3*width)`` tensor (~10 GB at width=290). We avoid ever materialising it:
the first edge-MLP linear is split into ``e @ We + h@Ws[src] + h@Wd[dst] + b``
(mathematically identical to ``Linear(3*width, width)`` on the concat), and both
the edge update and the scatter aggregation are processed in **edge chunks**
(:meth:`Processor.forward` ``edge_chunk``). Both are exact for MGN (edge updates
are per-edge independent; node aggregation is a scatter-add accumulated over
chunks), unlike Transolver's slice-chunking which is only an approximation.

At **training** time the activations must be retained for backward, so edge-
chunking cannot bound memory there. Instead pass ``grad_checkpoint=True`` (or the
``--grad-checkpoint`` flag of ``scripts/run_mgn.py``): each processor block is
wrapped in ``torch.utils.checkpoint`` so only its ``(h, e)`` inputs are kept and
the inner activations are recomputed in backward. This is numerically exact (fp32,
no AMP) — outputs and gradients are unchanged — trading ~1.3x compute for a large
memory reduction, which is what lets the matched-budget config (width=290,
n_layers=12, n_points=16384) fit the 12 GB RTX 4070 Ti.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.utils.checkpoint

__all__ = [
    "MeshGraphNetPointModel",
    "knn_edges",
    "edge_features",
]

_ACT = {"gelu": nn.GELU, "relu": nn.ReLU}


# --------------------------------------------------------------------------- #
# Graph construction (pure torch, bounded memory)
# --------------------------------------------------------------------------- #
def knn_edges(pos: torch.Tensor, k: int, chunk: int = 4096) -> torch.Tensor:
    """Fixed-``k`` kNN graph over a point cloud, computed in bounded memory.

    Returns a directed ``edge_index`` of shape ``(2, E)`` (``[src; dst]``) made
    **symmetric** (each kNN edge plus its reverse). Distances are computed in
    row-blocks of ``chunk`` query points (``torch.cdist`` per block,
    ``topk(k+1)`` then drop the self-match), so peak memory is ``O(chunk * N)``
    rather than ``O(N^2)`` — essential for the ~180k-point whole-cloud eval.

    A fixed-``k`` graph (NOT a radius graph) keeps the edge count, and therefore
    the message-passing memory, bounded and predictable.

    Parameters
    ----------
    pos : torch.Tensor
        ``(N, 2)`` RAW (un-normalised) point positions.
    k : int
        Number of nearest neighbours per node (excluding self).
    chunk : int
        Number of query rows per distance block.

    Returns
    -------
    torch.Tensor
        ``(2, E)`` long tensor on ``pos.device``; row 0 = src, row 1 = dst.
    """
    n = pos.shape[0]
    if n <= 1:
        return torch.empty(2, 0, dtype=torch.long, device=pos.device)
    k_eff = min(int(k), n - 1)
    src_list: list[torch.Tensor] = []
    dst_list: list[torch.Tensor] = []
    idx_all = torch.arange(n, device=pos.device)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        block = pos[s:e]  # (b, 2)
        b = block.shape[0]
        d = torch.cdist(block, pos)  # (b, N)
        # topk smallest distances incl. self -> take k+1 then drop self,
        # keeping the k_eff nearest non-self per row (fully vectorised, no
        # per-row Python loop so it stays fast at ~180k points).
        nbr = torch.topk(d, k=min(k_eff + 1, n), dim=1, largest=False).indices  # (b, k+1)
        q = idx_all[s:e].unsqueeze(1).expand_as(nbr)  # (b, k+1) query index
        # Push self-matches to the end, then take the first k_eff columns. If a
        # row has no self-match (numerical), this drops its farthest neighbour.
        is_self = nbr == q
        order = torch.argsort(is_self.to(torch.int8), dim=1, stable=True)  # non-self first
        nbr = torch.gather(nbr, 1, order)[:, :k_eff]  # (b, k_eff)
        dst_list.append(nbr.reshape(-1))
        src_list.append(idx_all[s:e].unsqueeze(1).expand(b, k_eff).reshape(-1))
    src = torch.cat(src_list)
    dst = torch.cat(dst_list)
    # Symmetrise: add reverse edges so the message graph is undirected.
    src_sym = torch.cat([src, dst])
    dst_sym = torch.cat([dst, src])
    # Deduplicate: if a is a kNN of b AND b a kNN of a, symmetrising would emit the
    # (a,b) edge twice and double-weight it under sum aggregation. Encode each
    # directed edge as a single integer key and keep uniques -> each neighbour pair
    # contributes exactly once per direction (strict MeshGraphNet faithfulness).
    key = src_sym.to(torch.int64) * n + dst_sym.to(torch.int64)
    key = torch.unique(key)
    src_u = (key // n).to(torch.long)
    dst_u = (key % n).to(torch.long)
    return torch.stack([src_u, dst_u], dim=0).contiguous()


def edge_features(
    pos: torch.Tensor, edge_index: torch.Tensor, l_xy: torch.Tensor, l_norm: float
) -> torch.Tensor:
    """Build ``(E, 3)`` edge features from RAW positions and a length scale.

    ``edge_attr = [ (pos[dst]-pos[src]) / l_xy , ||pos[dst]-pos[src]|| / l_norm ]``.

    Parameters
    ----------
    pos : torch.Tensor
        ``(N, 2)`` RAW positions.
    edge_index : torch.Tensor
        ``(2, E)`` ``[src; dst]``.
    l_xy : torch.Tensor
        ``(2,)`` per-axis length scale (e.g. ``PointNormalizer.std_in[:2]``).
    l_norm : float
        Scalar length scale for the distance column (e.g. ``||l_xy||``).
    """
    src, dst = edge_index[0], edge_index[1]
    rel = pos[dst] - pos[src]  # (E, 2)
    l_xy = l_xy.to(pos.device, pos.dtype).reshape(1, 2)
    rel_s = rel / l_xy
    dist = torch.linalg.norm(rel, dim=1, keepdim=True) / float(l_norm)
    return torch.cat([rel_s, dist], dim=1)  # (E, 3)


# --------------------------------------------------------------------------- #
# MLP building block
# --------------------------------------------------------------------------- #
class _MLP(nn.Module):
    """2-layer MLP ``in -> hidden -> out`` with activation, optional output LN."""

    def __init__(self, n_in: int, n_hidden: int, n_out: int, act: str = "relu", norm: bool = True):
        super().__init__()
        act_cls = _ACT[act]
        self.net = nn.Sequential(
            nn.Linear(n_in, n_hidden),
            act_cls(),
            nn.Linear(n_hidden, n_out),
        )
        self.norm = nn.LayerNorm(n_out) if norm else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return self.norm(x) if self.norm is not None else x


# --------------------------------------------------------------------------- #
# Message-passing block (Battaglia-style GN, memory-bounded edge update)
# --------------------------------------------------------------------------- #
class _GNBlock(nn.Module):
    """One MeshGraphNet processor step: edge update then node update, residual+LN.

    edge: ``e' = e + LN(edge_mlp([e, h_src, h_dst]))``        (Linear(3W,W) split)
    node: ``h' = h + LN(node_mlp([h, aggregate_dst(e')]))``   (sum/mean over in-edges)

    The first edge-MLP linear ``Linear(3*width, width)`` over ``[e, h_src, h_dst]``
    is split into three contributions so the ``(E, 3*width)`` concat is never
    formed (the memory gotcha): an edge term plus two per-NODE projections gathered
    at ``src``/``dst``. This is exactly equivalent to the concatenated linear.
    """

    def __init__(self, width: int, act: str = "relu", aggr: str = "sum"):
        super().__init__()
        self.width = int(width)
        self.aggr = aggr
        # --- edge MLP, first layer split into e/src/dst pieces (shared bias) ---
        self.edge_lin1_e = nn.Linear(width, width)  # carries the bias
        self.edge_lin1_src = nn.Linear(width, width, bias=False)
        self.edge_lin1_dst = nn.Linear(width, width, bias=False)
        self.edge_act = _ACT[act]()
        self.edge_lin2 = nn.Linear(width, width)
        self.edge_norm = nn.LayerNorm(width)
        # --- node MLP: [h, agg] (2W) -> W -> W ---
        self.node_mlp = _MLP(2 * width, width, width, act=act, norm=True)

    def _edge_update(
        self, e: torch.Tensor, h: torch.Tensor, src: torch.Tensor, dst: torch.Tensor,
        edge_chunk: int | None,
    ) -> torch.Tensor:
        """Compute ``e' = e + LN(edge_mlp([e, h_src, h_dst]))`` without (E,3W) concat."""
        # Per-node projections (N is small): h @ Ws and h @ Wd.
        t_src = self.edge_lin1_src(h)  # (N, W)
        t_dst = self.edge_lin1_dst(h)  # (N, W)
        E = e.shape[0]
        if edge_chunk is None or E <= edge_chunk:
            m = self.edge_lin1_e(e) + t_src[src] + t_dst[dst]  # (E, W) == Linear(3W,W)
            m = self.edge_lin2(self.edge_act(m))
            return e + self.edge_norm(m)
        out = torch.empty_like(e)
        for s in range(0, E, edge_chunk):
            sl = slice(s, min(s + edge_chunk, E))
            m = self.edge_lin1_e(e[sl]) + t_src[src[sl]] + t_dst[dst[sl]]
            m = self.edge_lin2(self.edge_act(m))
            out[sl] = e[sl] + self.edge_norm(m)
        return out

    def _aggregate(
        self, e: torch.Tensor, dst: torch.Tensor, n_nodes: int, edge_chunk: int | None
    ) -> torch.Tensor:
        """Scatter-sum (or mean) incoming edge messages to their dst node."""
        agg = torch.zeros(n_nodes, e.shape[1], dtype=e.dtype, device=e.device)
        E = e.shape[0]
        if edge_chunk is None or E <= edge_chunk:
            agg.index_add_(0, dst, e)
        else:
            for s in range(0, E, edge_chunk):
                sl = slice(s, min(s + edge_chunk, E))
                agg.index_add_(0, dst[sl], e[sl])
        if self.aggr == "mean":
            deg = torch.zeros(n_nodes, 1, dtype=e.dtype, device=e.device)
            ones = torch.ones(E, 1, dtype=e.dtype, device=e.device)
            deg.index_add_(0, dst, ones)
            agg = agg / deg.clamp_min(1.0)
        return agg

    def forward(
        self, h: torch.Tensor, e: torch.Tensor, edge_index: torch.Tensor,
        edge_chunk: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index[0], edge_index[1]
        e = self._edge_update(e, h, src, dst, edge_chunk)
        agg = self._aggregate(e, dst, h.shape[0], edge_chunk)
        m = self.node_mlp(torch.cat([h, agg], dim=1))  # (N, W)
        h = h + m
        return h, e


def _block_runner(block: _GNBlock, edge_index: torch.Tensor, edge_chunk: int | None):
    """Closure adapting a GN block to ``checkpoint``'s ``fn(*tensors)`` signature.

    The edge set and chunk size carry no gradient, so they are captured here
    rather than passed as checkpointed inputs; only the ``(h, e)`` latents are
    recomputed in backward.
    """

    def _fn(h: torch.Tensor, e: torch.Tensor):
        return block(h, e, edge_index, edge_chunk=edge_chunk)

    return _fn


# --------------------------------------------------------------------------- #
# Full Encode-Process-Decode model
# --------------------------------------------------------------------------- #
class MeshGraphNetPointModel(nn.Module):
    """MeshGraphNet (Pfaff et al. 2021) point-cloud regressor for AirfRANS.

    Encode-Process-Decode:

    * **Encoders.** A 2-layer node MLP ``in_features -> width -> width`` and a
      2-layer edge MLP ``edge_in -> width -> width`` (both LayerNorm on output).
    * **Processor.** ``n_layers`` Battaglia-style GN message-passing blocks
      (:class:`_GNBlock`): per-edge update then scatter-aggregated node update,
      each with a residual connection and LayerNorm.
    * **Decoder.** A 2-layer MLP ``width -> width -> out_features`` (no final
      norm), reading the final node latents.

    Maps per-point input features ``(B, N, in_features)`` to per-point predictions
    ``(B, N, out_features)`` = ``(u, v, p, nut)``. ``B`` is supported for the
    harness convention of one cloud at a time (``B = 1``); the model squeezes the
    batch, operates on ``(N, F)`` with the supplied edge set, and restores it.

    The matched-budget FULL config ``width=290, n_layers=12`` gives 7,351,214
    params (vs the 7,350,420 Transolver/FNO anchor, 0.01%); the exact count is
    reported by :meth:`num_params`.
    """

    def __init__(
        self,
        in_features: int = 7,
        out_features: int = 4,
        width: int = 128,
        n_layers: int = 12,
        edge_in: int = 3,
        act: str = "relu",
        aggr: str = "sum",
        edge_chunk: int | None = None,
        grad_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        if aggr not in ("sum", "mean"):
            raise ValueError(f"aggr must be 'sum' or 'mean', got {aggr!r}")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.width = int(width)
        self.n_layers = int(n_layers)
        self.edge_in = int(edge_in)
        self.edge_chunk = edge_chunk
        self.grad_checkpoint = bool(grad_checkpoint)

        self.node_encoder = _MLP(in_features, width, width, act=act, norm=True)
        self.edge_encoder = _MLP(edge_in, width, width, act=act, norm=True)
        self.processor = nn.ModuleList(
            [_GNBlock(width, act=act, aggr=aggr) for _ in range(n_layers)]
        )
        self.decoder = _MLP(width, width, out_features, act=act, norm=False)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor,
        edge_chunk: int | None = None, grad_checkpoint: bool | None = None,
    ) -> torch.Tensor:
        """``(B, N, F_IN), (2, E), (E, edge_in) -> (B, N, F_OUT)``.

        Only ``B = 1`` is supported (one point cloud + its edge set per forward),
        matching the harness which forwards a single cloud at a time. ``edge_chunk``
        overrides the instance default for this call (memory-bounded eval).
        ``grad_checkpoint`` overrides the instance default for this call.
        """
        squeeze = False
        if x.dim() == 3:
            if x.shape[0] != 1:
                raise ValueError(
                    f"MeshGraphNetPointModel supports B=1 (one cloud + edge set per "
                    f"forward); got batch {x.shape[0]}."
                )
            x = x.squeeze(0)
            squeeze = True
        ec = edge_chunk if edge_chunk is not None else self.edge_chunk
        gc = grad_checkpoint if grad_checkpoint is not None else self.grad_checkpoint
        # Checkpointing only helps (and is only valid) during a grad-enabled training
        # pass; in eval / no_grad there are no activations to trade, so skip it.
        use_ckpt = gc and self.training and torch.is_grad_enabled()

        h = self.node_encoder(x)  # (N, W)
        e = self.edge_encoder(edge_attr)  # (E, W)
        for block in self.processor:
            if use_ckpt:
                # Capture the (non-grad) edge set + chunk in a closure so only the
                # (h, e) latents are checkpointed inputs; activations inside the
                # block are dropped and recomputed in backward (exact, memory↓).
                h, e = torch.utils.checkpoint.checkpoint(
                    _block_runner(block, edge_index, ec), h, e, use_reentrant=False,
                )
            else:
                h, e = block(h, e, edge_index, edge_chunk=ec)
        out = self.decoder(h)  # (N, F_OUT)
        return out.unsqueeze(0) if squeeze else out

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
