"""Feed brepgraph features to a small PyG model (per-face classification).

Run:  python examples/gnn_with_pyg.py [folder-with-step-files]

With no argument it uses generated sample parts. The labels here are random: the point is to show
how each feature group is encoded and combined, not to train something useful.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv

import brepgraph as bg


class FaceEncoder(nn.Module):
    """surface_type (embedding) + area/centroid (linear) + uv_grid (2-D CNN) -> one vector per face."""

    def __init__(self, dim: int):
        super().__init__()
        self.surface = nn.Embedding(11, 16)
        self.flat = nn.Linear(1 + 3, 16)
        self.grid = nn.Sequential(
            nn.Conv2d(7, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.out = nn.Linear(16 + 16 + 64, dim)

    def forward(self, batch):
        flat = torch.cat([batch.area[:, None], batch.centroid], dim=1)
        parts = [self.surface(batch.surface_type), self.flat(flat), self.grid(batch.uv_grid)]
        return self.out(torch.cat(parts, dim=1))


class EdgeEncoder(nn.Module):
    """curve_type / convexity (embeddings) + length / dihedral_angle (linear) + u_grid (1-D CNN)."""

    def __init__(self, dim: int):
        super().__init__()
        self.curve = nn.Embedding(9, 8)
        self.convexity = nn.Embedding(3, 8)
        self.grid = nn.Sequential(
            nn.Conv1d(6, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten()
        )
        self.out = nn.Linear(8 + 8 + 2 + 32, dim)

    def forward(self, batch):
        flat = torch.stack([batch.length, batch.dihedral_angle], dim=1)
        parts = [self.curve(batch.curve_type), self.convexity(batch.convexity), flat, self.grid(batch.u_grid)]
        return self.out(torch.cat(parts, dim=1))


class FaceClassifier(nn.Module):
    def __init__(self, dim: int = 64, num_classes: int = 5, layers: int = 3):
        super().__init__()
        self.face_encoder = FaceEncoder(dim)
        self.edge_encoder = EdgeEncoder(dim)
        self.convs = nn.ModuleList(
            GINEConv(nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)), edge_dim=dim)
            for _ in range(layers)
        )
        self.head = nn.Linear(dim, num_classes)

    def forward(self, batch):
        x, edge_attr = self.face_encoder(batch), self.edge_encoder(batch)
        for conv in self.convs:
            x = torch.relu(conv(x, batch.edge_index, edge_attr))
        return self.head(x)


def sample_files(folder: Path | None) -> list[Path]:
    if folder is not None:
        return sorted(p for p in folder.rglob("*") if p.suffix.lower() in {".step", ".stp"})[:16]
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
    import fixtures

    out = Path(tempfile.mkdtemp())
    shapes = [fixtures.pocket_box(), fixtures.filleted_box(), fixtures.plate_with_hole(), fixtures.cylinder()]
    return [fixtures.write_step(s, out / f"part{i}.step") for i, s in enumerate(shapes)]


def main() -> None:
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    # parallel loading; each solid of a multi-body file becomes a graph; unit size keeps features scaled
    results = bg.load_many(sample_files(folder), normalize=True, all_solids=True)
    graphs = []
    for result in results:
        if not result.ok:
            print(f"skipping {Path(result.source).name}: {result.error}")
            continue
        for g in result.graphs:
            data = g.to_pyg()
            data.y = torch.randint(0, 5, (data.num_nodes,))  # placeholder per-face labels
            graphs.append(data)
    print(f"{len(graphs)} graphs from {len(results)} files")

    model = FaceClassifier()
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(graphs, batch_size=4, shuffle=True)
    for epoch in range(3):
        total = 0.0
        for batch in loader:
            optimiser.zero_grad()
            loss = nn.functional.cross_entropy(model(batch), batch.y)
            loss.backward()
            optimiser.step()
            total += loss.item()
        print(f"epoch {epoch}: loss {total / len(loader):.3f}")


if __name__ == "__main__":
    main()
