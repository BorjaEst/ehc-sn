# Head comments

from ehc_sn.mec.core import GridCellsBase, MECBase
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import RegularPolygon


class GridLayerView(GridCellsBase):
    # A view class for grid cell layers that inherits core functionality and adds visualization

    def show(self, ax=None, cell_size=0.8, edgecolor="black", title=None):
        # Creates a visual representation of the grid cell layer

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))

        # Create proper colormap for activity coloring
        cmap = plt.get_cmap("viridis")
        # Normalize activity values to [0, 1] for colormap
        norm = colors.Normalize(vmin=0, vmax=np.max(self.activity) or 1.0)

        # Plot hexagons at each grid cell position
        for i in range(self.height):
            for j in range(self.width):
                pos = self.positions[i, j]

                # Create hexagon
                hex_cell = RegularPolygon(
                    pos,
                    numVertices=6,
                    radius=cell_size * self.spacing / 2,
                    orientation=np.pi / 6,  # Orientation for flat-topped hexagon
                    edgecolor=edgecolor,
                    facecolor=cmap(norm(self.activity[i, j])),
                    alpha=0.8,
                )
                ax.add_patch(hex_cell)

        # Set axis limits a bit larger than the grid
        max_pos = np.max(self.positions) + self.spacing
        min_pos = np.min(self.positions) - self.spacing
        ax.set_xlim(min_pos, max_pos)
        ax.set_ylim(min_pos, max_pos)
        ax.set_aspect("equal")

        # Set title
        if title is None:
            title = f"Grid Cell Layer Structure (spacing={self.spacing})"
        ax.set_title(title)

        return ax


class MECView(MECBase):
    # A view class for MEC networks that inherits core functionality and adds visualization

    def show(self, figsize=(16, 10)):
        # Creates a visual representation of all grid cell layers in the network

        n_grids = len(self.grid_cells)

        # Create a figure with just the grid structures
        fig, axes = plt.subplots(1, n_grids, figsize=figsize)

        # Handle case with single grid
        if n_grids == 1:
            axes = [axes]

        # Plot each grid layer
        for i, grid in enumerate(self.grid_cells):
            grid.show(
                ax=axes[i],
                title=f"Grid Cells (spacing={grid.spacing:.2f}, orientation={grid.orientation:.2f})",
            )

        plt.tight_layout()
        return fig
