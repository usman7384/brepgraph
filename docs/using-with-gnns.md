# Using the graph with a GNN

brepgraph produces plain NumPy arrays; `to_pyg()` and `to_dgl()` only wrap them in each framework's container.
Every feature stays a separate named attribute, so a model decides how to encode each one.

The usual recipe is to encode each feature group, concatenate the results into one vector per face (and per
edge), and run message passing on those vectors:

| Group | Features | Typical encoder |
| --- | --- | --- |
| categorical | `surface_type`, `curve_type`, `convexity`, `num_loops` | `nn.Embedding` on the integer index (or `transforms.one_hot`) |
| small scalars / vectors | `area`, `centroid`, `length`, `dihedral_angle` | `nn.Linear` |
| grids | `uv_grid` `[7, n, n]`, `u_grid` `[6, n]` | 2-D / 1-D CNN with global pooling |
| face pairs | `shortest_path`, `d2`, `a3` | attention bias in a graph transformer (BrepMFR); not message passing |

[examples/gnn_with_pyg.py](../examples/gnn_with_pyg.py) is a complete, runnable version for PyTorch Geometric:
a face encoder, an edge encoder and three `GINEConv` layers that classify every face. It trains on random
labels, since its purpose is to show the wiring.

```python
results = bg.load_many(files, normalize=True, all_solids=True)     # parallel, one result per file
graphs = [g.to_pyg() for r in results if r.ok for g in r.graphs]   # failed files carry r.error
loader = DataLoader(graphs, batch_size=32, shuffle=True)
```

For a large dataset, convert once with `brepgraph batch data/ -o graphs/ --all-solids` and load the `.npz` files
with `bg.BrepGraph.load` in your `Dataset`, instead of parsing STEP files every epoch.

Practical points:

- **Both edge directions.** `to_pyg()` and `to_dgl()` duplicate every B-rep edge into two directed edges and repeat
  the edge features. Edge tangents in `u_grid` keep the B-rep orientation in both copies.
- **Scale.** Areas, lengths and coordinates vary by orders of magnitude between parts. Load with
  `normalize=True` (longest side 1) or standardise per feature across your dataset.
- **Mask channel.** Channel 6 of `uv_grid` marks samples inside the trimmed face; a CNN can use it as an input
  channel or you can multiply the other channels by it.
- **Flat features only.** `g.to_pyg(x=("area", "centroid"), edge_attr=("length", "dihedral_angle"))` builds
  `data.x` and `data.edge_attr` for models such as `GCNConv` or `GATv2Conv` that expect a single matrix. Grids
  stay separate attributes.
- **Pair features** are dense `[F, F, ...]` arrays. PyG cannot batch them for graphs of different sizes without a
  custom collate that pads them. Histograms stored as `uint16` counts arrive as `torch.long`; divide by the sample
  count for frequencies.
