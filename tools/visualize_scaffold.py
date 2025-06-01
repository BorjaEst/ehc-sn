import matplotlib.pyplot as plt


def visualize_scaffold(scaffold):
    """Visualize the current state of the memory spatial scaffold."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot basic scaffold activations
    ax = axes[0, 0]
    im = ax.imshow(scaffold.scaffold.grid.T, origin='lower', cmap='viridis')
    ax.set_title('Spatial Scaffold Activations')
    plt.colorbar(im, ax=ax)
    
    # Plot one grid cell module
    ax = axes[0, 1]
    im = ax.imshow(scaffold.grid_modules[0].pattern.T, origin='lower', cmap='viridis')
    ax.set_title(f'Grid Cell Module (Scale {scaffold.grid_modules[0].scale})')
    plt.colorbar(im, ax=ax)
    
    # Plot place cell centers and current activation
    ax = axes[1, 0]
    ax.scatter(scaffold.place_cells.centers[:, 0], 
               scaffold.place_cells.centers[:, 1], 
               c='blue', alpha=0.5, s=10)
    
    # Highlight the currently active place cells
    activations = scaffold.place_cells.get_activations(scaffold.current_position)
    active_cells = activations > 0.3
    ax.scatter(scaffold.place_cells.centers[active_cells, 0],
               scaffold.place_cells.centers[active_cells, 1],
               c='red', s=50)
    
    ax.set_xlim(0, scaffold.dimensions[0])
    ax.set_ylim(0, scaffold.dimensions[1])
    ax.set_title('Place Cell Centers (active in red)')
    
    # Plot the position encoding
    ax = axes[1, 1]
    encoding = scaffold.get_position_encoding()
    ax.bar(range(len(encoding)), encoding)
    ax.set_title('Position Encoding Vector')
    ax.set_xlabel('Encoding Element')
    ax.set_ylabel('Activation')
    
    # Mark the current position on all spatial plots
    for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
        ax.plot(scaffold.current_position[0], scaffold.current_position[1], 'ro', markersize=10)
    
    plt.tight_layout()
    plt.show()
