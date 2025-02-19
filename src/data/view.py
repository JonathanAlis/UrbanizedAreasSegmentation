import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
import math
import torch
import numpy as np



def get_rgb(multi_channel_image:torch.Tensor):
    if multi_channel_image.shape[0]!=3:
        rgb_image = multi_channel_image[1:4,:,:]
        rgb_image = rgb_image.permute(1, 2, 0)
        return rgb_image.cpu().detach().numpy()

def move_to_0_1(image):
    image = image - np.min(image)
    image /= np.max(image)
    return image

def show_batches(img_batch:torch.Tensor, label_batch:torch.Tensor, pred_batch:torch.Tensor, mask_color=(1, 0, 0), mask_alpha=0.5, border_color='red', border_width=2, max_shown = None):
    #[num_batches, 12, W,H]
    max_per_line = 4
    
    mask_only = True
    colors = ['black', 'yellow', 'red', 'blue', 'white']
    cmap = mcolors.ListedColormap(colors)
    bounds = [0, 1, 2, 3, 4, 5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    num_imgs = img_batch.shape[0]
    if max_shown is None:
        max_shown = num_imgs
    if max_shown < num_imgs:
        unique_counts = torch.tensor([len(torch.unique(tensor)) for tensor in label_batch])
        indices = torch.argsort(unique_counts, descending=True)[:max_shown]
    else:
        indices = range(num_imgs)
    num_lines = math.ceil(max_shown/max_per_line) 
    
    plt.figure(figsize=(20,num_lines*20//4))
    for count, i in enumerate(indices):
        plt.subplot(num_lines, min([num_imgs, max_per_line]), count+1)
        img = img_batch[i]
        mask = np.squeeze(label_batch[i].cpu().detach().numpy())

        #print(np.unique(mask))
        #colors = [(0, 0, 0, 0), mask_color + (mask_alpha,)]
        #cmap = LinearSegmentedColormap.from_list("custom", colors, N=2)
        if mask_only:
            plt.imshow(move_to_0_1(get_rgb(img)))
        plt.imshow(mask, alpha = 0.5, cmap=cmap, norm=norm)
        #plt.colorbar(ticks=[0.5, 1.5, 2.5, 3.5, 4.5], label='Value')
        #plt.contour(mask, colors=border_color, linewidths=border_width, levels=[0.5])
        #plt.imshow(mask, cmap=cmap, alpha = 0.5)
        plt.title(f'Ground Truth image {count+1}')
    plt.show()

    plt.figure(figsize=(20,num_lines*20//4))
    for count, i in enumerate(indices):
        plt.subplot(num_lines, min([num_imgs, max_per_line]), count+1)
        img = img_batch[i]
        mask = pred_batch[i].cpu().detach().numpy()
        mask = np.argmax(mask, axis = 0)
        #colors = [(0, 0, 0, 0), mask_color + (mask_alpha,)]
        #cmap = LinearSegmentedColormap.from_list("custom", colors, N=2)
        if mask_only:
            plt.imshow(move_to_0_1(get_rgb(img)))
        plt.imshow(mask, alpha = 0.5, cmap=cmap, norm=norm)
        #plt.colorbar(ticks=[0.5, 1.5, 2.5, 3.5, 4.5], label='Value')
        #plt.contour(mask, colors=border_color, linewidths=border_width, levels=[0.5])

        plt.title(f'Prediction image {count+1}')
    plt.show()


def plt_tile(label_mask:np.ndarray, pred_mask:np.ndarray, img_batch:np.ndarray = None, mask_color=(1, 0, 0), mask_alpha=0.5, border_color='red', border_width=2):
    #[num_batches, 12, W,H]
    max_per_line = 4
    
    mask_only = True
    colors = ['black', 'yellow', 'red', 'blue', 'white']
    cmap = mcolors.ListedColormap(colors)
    bounds = [0, 1, 2, 3, 4, 5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(20,20))
    plt.subplot(1, 2, 1)
    plt.imshow(label_mask, alpha = mask_alpha, cmap=cmap, norm=norm)
    plt.colorbar(ticks=[0.5, 1.5, 2.5, 3.5, 4.5], label='Value')
    #plt.contour(mask, colors=border_color, linewidths=border_width, levels=[0.5])
    #plt.imshow(mask, cmap=cmap, alpha = 0.5)
    plt.title(f'Label mask')
    
    plt.subplot(1, 2, 2)
    plt.imshow(pred_mask, alpha = mask_alpha, cmap=cmap, norm=norm)
    plt.colorbar(ticks=[0.5, 1.5, 2.5, 3.5, 4.5], label='Value')
    #plt.contour(mask, colors=border_color, linewidths=border_width, levels=[0.5])
    #plt.imshow(mask, cmap=cmap, alpha = 0.5)
    plt.title(f'Label mask')
    

    plt.show()


from scipy.stats import zscore

def plot_metrics(history, c='loss', save_file: str = None):
    branches = []
    last_lr = -1
    best_val_loss = np.inf
    branch = []
    epoch_last_best = -1

    branches_train = []
    branch_train = []
    branches_val = []
    branch_val = []
    all_values = []
    for epoch, info in enumerate(history):
        if epoch!=info['epoch']:
            print('error...')
        #print(info['patience'])
        #print(f"{info['lr']:2f}")
        #print(info['val_loss'])
        #print(branch)
        #check here
        all_values.append(info[f'train_{c}'])
        all_values.append(info[f'val_{c}'])

        if info['lr'] != last_lr:
            print("Novo LR")
            branches.append(branch) #nao apend, termina e cria outro
            branches_train.append(branch_train) #nao apend, termina e cria outro
            branches_val.append(branch_val) #nao apend, termina e cria outro
            
            if epoch_last_best>=0:
                branch = [(history[epoch_last_best]['epoch'], history[epoch_last_best]['val_loss'], history[epoch_last_best]['lr']),
                            (info['epoch'], info['val_loss'], info['lr'])]
                branch_train = [(history[epoch_last_best]['epoch'], history[epoch_last_best][f'train_{c}'], history[epoch_last_best]['lr']),
                            (info['epoch'], info[f'train_{c}'], info['lr'])]
                branch_val = [(history[epoch_last_best]['epoch'], history[epoch_last_best][f'val_{c}'], history[epoch_last_best]['lr']),
                            (info['epoch'], info[f'val_{c}'], info['lr'])]
                
            else:
                branch = [(info['epoch'], info['val_loss'], info['lr'])]
                branch_train = [(info['epoch'], info[f'train_{c}'], info['lr'])]
                branch_val = [(info['epoch'], info[f'val_{c}'], info['lr'])]
        else:
            branch.append((info['epoch'], info['val_loss'], info['lr']))
            branch_train.append((info['epoch'], info[f'train_{c}'], info['lr']))
            branch_val.append((info['epoch'], info[f'val_{c}'], info['lr']))
            
        if info['val_loss']<best_val_loss:
            best_val_loss = info['val_loss']
            #print('NEW BEST VAL LOSS',best_val_loss)
            epoch_last_best = info['epoch']
        
        last_patience = info['patience']
        last_lr = info['lr']
    branches.append(branch) #nao apend, termina e cria outro
    branches_train.append(branch_train) #nao apend, termina e cria outro
    branches_val.append(branch_val) #nao apend, termina e cria outro
    fig, ax = plt.subplots(figsize=(10, 6))
    linestyles=['-', '--', '-.', ':']
    lsi = 0
    markers = ["o", "s"]
    mi = 0
    lowest_val_loss = np.inf
    lowest_val_loss_epoch = -1
    for branch, branch_train, branch_val in zip(branches, branches_train, branches_val):
        if len(branch)>0:
            
            epochs = [b[0]+1 for b in branch]
            value_train = [b[1] for b in branch_train]
            value_val = [b[1] for b in branch_val]
            lr = [b[2] for b in branch]
    
            # Plot training and validation losses
            ax.plot(epochs, value_train, label=f"Train {c}, LR: {lr[-1]:3f}", marker=markers[mi], linestyle=linestyles[lsi], color="blue")
            ax.plot(epochs, value_val, label=f"Val {c}, LR: {lr[-1]:3f}", marker=markers[mi], linestyle=linestyles[lsi], color="red")
            lsi+=1
            if lsi==4:
                lsi = 0
                mi+=1
            #ax.plot(epochs, val_metric, label=f"Validation {column}", marker="o", linestyle="-", color="red")
    
    #ax.scatter(epoch_last_best+1, best_val_loss, color='green', marker='*', s=100, label="New Point")
    ax.axvline(x=epoch_last_best+1, color='green', linestyle='--', label="Best epoch")

    all_values = np.array(all_values)
    z_scores = zscore(all_values)
    filtered_data = all_values[np.abs(z_scores) < 3]
    y_min, y_max = min(filtered_data), max(filtered_data)
    margin = (y_max - y_min) * 0.2  # 10% extra margin
    ax.set_ylim(y_min - margin, y_max + margin)


    # Add labels, title, and legend
    ax.set_title(f"Training and Validation {c}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(c)
    ax.legend()
    ax.grid(True)
        # If a filename is provided, save the plot as a .png file
    if save_file:
        plt.savefig(save_file)
        print(f"Plot saved to {save_file}")

import torch
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cm: torch.Tensor, class_names=None, normalize=True, cmap="Blues"):
    """
    Plots two confusion matrices: one with merged classes (binary) and one with all classes.

    Args:
        cm (torch.Tensor): The confusion matrix of shape [num_classes, num_classes].
        class_names (list, optional): List of class names corresponding to indices.
        normalize (bool): Whether to normalize the confusion matrix to percentages.
        cmap (str): Color map for visualization.
    """
    
    cm = cm.float().cpu()  # Ensure it's a CPU tensor for plotting

    # Normalize
    if normalize:
        cm = cm / cm.sum(dim=1, keepdim=True)
        cm = cm.nan_to_num(0)

    # Indices for merged classes
    group_0 = [0, 3]  # Merging classes 0 and 3
    group_1 = [1, 2, 4]  # Merging classes 1, 2, and 4
    binary_cm = torch.zeros(2, 2, dtype=torch.float32)
    binary_cm[0, 0] = cm[group_0, :][:, group_0].sum()  # (0,3) predicted as (0,3)
    binary_cm[0, 1] = cm[group_0, :][:, group_1].sum()  # (0,3) predicted as (1,2,4)
    binary_cm[1, 0] = cm[group_1, :][:, group_0].sum()  # (1,2,4) predicted as (0,3)
    binary_cm[1, 1] = cm[group_1, :][:, group_1].sum()  # (1,2,4) predicted as (1,2,4)
    if normalize:
        binary_cm = binary_cm / binary_cm.sum(dim=1, keepdim=True)
        binary_cm = binary_cm.nan_to_num(0)
    num_classes = cm.shape[0]
    class_labels = class_names if class_names else list(range(num_classes))
    binary_labels = ["(0,3)", "(1,2,4)"]


    # Indices for merged classes
    group_0 = [0, 3]  # Merging classes 0 and 3
    group_1 = [1]
    group_2 = [2]
    group_3 = [4]
    join_cm = torch.zeros(4, 4, dtype=torch.float32)
    join_cm[0, 0] = cm[group_0, :][:, group_0].sum()  # (0,3) predicted as (0,3)
    join_cm[0, 1] = cm[group_0, :][:, group_1].sum()  # (0,3) predicted as (1,2,4)
    join_cm[1, 0] = cm[group_1, :][:, group_0].sum()  # (1,2,4) predicted as (0,3)
    join_cm[1, 1] = cm[group_1, :][:, group_1].sum()  # (1,2,4) predicted as (1,2,4)
    join_cm[2, 0] = cm[group_2, :][:, group_0].sum()  # (1,2,4) predicted as (0,3)
    join_cm[0, 2] = cm[group_0, :][:, group_2].sum()  # (1,2,4) predicted as (1,2,4)
    join_cm[2, 2] = cm[group_2, :][:, group_2].sum()  # (1,2,4) predicted as (1,2,4)
    join_cm[3, 0] = cm[group_3, :][:, group_0].sum()  # (1,2,4) predicted as (0,3)
    join_cm[0, 3] = cm[group_0, :][:, group_3].sum()  # (1,2,4) predicted as (1,2,4)
    join_cm[3, 3] = cm[group_3, :][:, group_3].sum()  # (1,2,4) predicted as (1,2,4)
    join_cm[2, 1] = cm[group_2, :][:, group_1].sum()  # (1,2,4) predicted as (0,3)
    join_cm[1, 2] = cm[group_1, :][:, group_2].sum()  # (1,2,4) predicted as (1,2,4)
    join_cm[2, 3] = cm[group_2, :][:, group_3].sum()  # (1,2,4) predicted as (0,3)
    join_cm[3, 2] = cm[group_3, :][:, group_2].sum()  # (1,2,4) predicted as (1,2,4)
    join_cm[3, 1] = cm[group_3, :][:, group_1].sum()  # (1,2,4) predicted as (0,3)
    join_cm[1, 3] = cm[group_1, :][:, group_3].sum()  # (1,2,4) predicted as (1,2,4)
    if normalize:
        join_cm = join_cm / join_cm.sum(dim=1, keepdim=True)
        join_cm = join_cm.nan_to_num(0)
    num_classes = cm.shape[0]
    class_labels = class_names if class_names else list(range(num_classes))
    join_labels = ["(0,3)", "(1)", "(2)", "(4)"]


    # Create subplot
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    # Binary confusion matrix
    sns.heatmap(binary_cm.numpy(), annot=True, fmt=".2f" if normalize else "d", cmap=cmap, linewidths=0.5, square=True, ax=axes[0])
    axes[0].set_xticklabels(binary_labels, rotation=45, ha="right", fontsize=10)
    axes[0].set_yticklabels(binary_labels, rotation=0, fontsize=10)
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")
    axes[0].set_title("Binary Confusion Matrix (Merged Classes)")

    sns.heatmap(join_cm.numpy(), annot=True, fmt=".2f" if normalize else "d", cmap=cmap, linewidths=0.5, square=True, ax=axes[1])
    axes[1].set_xticklabels(join_labels, rotation=45, ha="right", fontsize=10)
    axes[1].set_yticklabels(join_labels, rotation=0, fontsize=10)
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")
    axes[1].set_title("Joined Confusion Matrix (Merged Classes)")

    # Multiclass confusion matrix
    sns.heatmap(cm.numpy(), annot=True, fmt=".2f" if normalize else "d", cmap=cmap, linewidths=0.5, square=True, ax=axes[2])
    axes[2].set_xticklabels(class_labels, rotation=45, ha="right", fontsize=10)
    axes[2].set_yticklabels(class_labels, rotation=0, fontsize=10)
    axes[2].set_xlabel("Predicted Label")
    axes[2].set_ylabel("True Label")
    axes[2].set_title("Confusion Matrix (5 Classes)")

    plt.tight_layout()
    plt.show()


def plot_pca_batch(images: torch.Tensor, images_per_row: int = 4, title: str = "PCA Transformed Images"):
    """
    Plots a batch of PCA-transformed images in a grid layout.

    Args:
        images (torch.Tensor): Tensor of shape [B, 3, W, H], where the 3 channels are PCA components.
        images_per_row (int): Number of images to display per row.
        title (str): Title of the plot.
    """
    B, C, W, H = images.shape
    assert C == 3, "Expected 3 channels for PCA visualization"

    # Normalize images for visualization
    images_min = images.amin(dim=(2, 3), keepdim=True)
    images_max = images.amax(dim=(2, 3), keepdim=True)
    images_norm = (images - images_min) / (images_max - images_min)  # Normalize to [0,1]

    # Compute grid size
    rows = math.ceil(B / images_per_row)
    cols = min(B, images_per_row)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

    if rows == 1:
        axes = [axes]  # Ensure iterable for a single row
    axes = [ax for row in axes for ax in (row if isinstance(row, np.ndarray) else [row])]  # Flatten

    for i in range(B):
        img = images_norm[i].permute(1, 2, 0).cpu().numpy()  # Convert to (W, H, 3)
        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(f"Image {i+1}")

    # Hide unused subplots (if B is not a multiple of images_per_row)
    for j in range(B, len(axes)):
        axes[j].axis("off")

    plt.suptitle(title)
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_masked_image(mask, label_map, image=None, title="Mask Overlay", ax=None):
    """
    Plots a label mask with colored overlays (with transparency) and optionally a 
    background image if provided. Returns the axis used for plotting, so that the 
    plot can be embedded in subplots or further customized.
    
    Parameters
    ----------
    mask : np.ndarray
        The label mask as a NumPy array of shape (H, W) where pixel values indicate labels.
    label_map : dict
        A dictionary mapping label values to a tuple (cmap_name, alpha, legend_label).
        Example for five labels:
            {
                1: ('Reds', 0.3, 'Label 1'),
                2: ('Blues', 0.7, 'Label 2'),
                3: ('Greens', 0.4, 'Label 3'),
                4: ('Purples', 0.6, 'Label 4'),
                5: ('Oranges', 0.5, 'Label 5')
            }
    image : np.ndarray, optional
        An RGB image as a NumPy array of shape (H, W, 3). If provided, the image is used as the background.
        If not provided, a blank (white) background is used.
    title : str, optional
        The title of the plot.
    ax : matplotlib.axes.Axes, optional
        An existing Axes to plot on. If None, a new figure and axes are created.
    
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plotted image and overlay.
    """
    # Create an axis if not provided.
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Display the background image if provided; otherwise, use a white background.
    if image is not None:
        ax.imshow(image)
    else:
        white_bg = np.ones((mask.shape[0], mask.shape[1], 3))
        ax.imshow(white_bg)
        
    legend_handles = []
    
    # Loop through each label in the label_map and create a colored overlay.
    for label_value, (cmap_name, alpha, legend_label) in label_map.items():
        # Create a boolean mask for the current label.
        mask_bool = (mask == label_value)
        
        # Create an empty RGBA overlay (initialize with zeros).
        overlay = np.zeros((mask.shape[0], mask.shape[1], 4))
        
        # Get a representative color from the colormap.
        # The value 0.6 is arbitrary; you can adjust it as needed.
        color = list(plt.get_cmap(cmap_name)(0.6))
        # Override the colormap's alpha with the desired transparency.
        color[3] = alpha
        
        # Fill the overlay only where the mask equals the current label.
        overlay[mask_bool] = color
        
        # Plot the overlay (the overlay carries its own alpha).
        ax.imshow(overlay, interpolation='none')
        
        # Create a legend patch using the same color.
        patch = mpatches.Patch(color=color, label=legend_label)
        legend_handles.append(patch)
    
    ax.legend(handles=legend_handles, loc='upper right')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    
    return ax

def plot_masks(masks):

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 7, rows * 7))  # Adjust figure size
    axes = axes.flatten()  # Flatten for easy iteration


    class_colors = {
        0: "#000000",  # Black (Background)
        1: "#0000FF",  # Cyan
        2: "#FF00FF",  # Magenta
        3: "#FFFF00",  # Yellow
        4: "#FFFFFF"   # White
    }

    cmap = mcolors.ListedColormap([class_colors[i] for i in sorted(class_colors.keys())])
    bounds = sorted(class_colors.keys()) + [max(class_colors.keys()) + 1]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)


def plot_mask_list(masks, titles = None, save_to = None):
    
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.colors as mcolors

    cols = min(3, math.floor(math.sqrt(len(masks))))
    
    rows = math.ceil(len(masks)/cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 7, rows * 7))  # Adjust figure size
    axes = axes.flatten()  # Flatten for easy iteration


    class_colors = {
        0: "#000000",  # Black (Background)
        1: "#0000FF",  # Cyan
        2: "#FF00FF",  # Magenta
        3: "#FFFF00",  # Yellow
        4: "#FFFFFF"   # White
    }

    cmap = mcolors.ListedColormap([class_colors[i] for i in sorted(class_colors.keys())])
    bounds = sorted(class_colors.keys()) + [max(class_colors.keys()) + 1]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    

    for i, (ax, mask) in enumerate(zip(axes,masks)):
        im = ax.imshow(mask, cmap=cmap, extent=[0, mask.shape[1], mask.shape[0], 0], alpha=1)
        
        if titles is not None:
            ax.title(titles[i])

    class_labels = {
            0: "Fundo",
            1: "Loteamento vazio",
            2: "Outros equipamentos",
            3: "Vazio intraurbano",
            4: "Área Urbanizada"
        }

    legend_patches = [
    patches.Patch(facecolor=class_colors[val], edgecolor="black", linewidth=1, label=class_labels[val])
        for val in class_colors.keys()
    ]
    legend_patches = [patches.Patch(color=class_colors[val], label=class_labels[val]) for val in class_colors.keys()]
    fig.legend(handles=legend_patches, loc="upper right", title="Classes")
    # Assign class labels if provided
    
    plt.tight_layout()#rect=[0, 0, 0.85, 1])  # Adjust layout to fit colorbar
    if save_to:
        plt.savefig(save_to, dpi=300, bbox_inches="tight")
    plt.show()


def plot_horizontal(values, names, metric_name, set = "Treinamento", save_to = None):
    import numpy as np
    import matplotlib.pyplot as plt

    # Bar chart settings
    y = np.arange(len(names))  # Positions for groups
    height = 0.15  # Bar height

    fig, ax = plt.subplots(figsize=(12, 16))

    # Plot bars for each key
    ax.grid(axis="x", linestyle="--", linewidth=0.7, alpha=0.7)
    keys = [f'{metric_name} C{i}' for i in range(5)]
    for i, key in enumerate(keys):
        ax.barh(y + i * height, [100*v[i] for v in values], height, label=key)
    for pos in y[:-1]:  # Avoid last position
        ax.axhline(pos + height * len(keys), color="gray", linestyle="--", alpha=0.5)

    # Format plot
    ax.set_yticks(y + height)  # Center labels
    model_names = [n.replace("-type", "").replace(".pth", "") for n in names]
    ax.set_yticklabels(model_names)
    ax.grid(axis="x", linestyle="--", linewidth=0.7, alpha=0.7)
    ax.set_xticks(np.arange(0, 101, 5))  # Ticks from 0 to 1 with step 0.1

    #ax.grid(axis="x", which="minor", linestyle=":", linewidth=0.5)  # Dotted minor grid

    ax.legend(title=f"Classes")
    ax.set_xlabel("Percentual")
    ax.set_title(f"{metric_name} {set}")
    #ax.set_yticklabels(ax.get_ytickslabels(), rotation=45)  # Rotate labels by 45 degrees
    plt.tight_layout()#rect=[0, 0, 0.85, 1])  # Adjust layout to fit colorbar

    if save_to:
        plt.savefig(save_to, dpi=300, bbox_inches="tight")

    plt.show()

def plot_vertical(values, names, metric_name, set="Treinamento", save_to=None):
    import numpy as np
    import matplotlib.pyplot as plt

    # Bar chart settings
    x = np.arange(len(names))  # Positions for groups
    width = 0.15  # Bar width

    # Split data into two rows for better visualization
    half = len(names) // 2
    names_row1, names_row2 = names[:half], names[half:]
    values_row1, values_row2 = values[:half], values[half:]

    # Create subplots with 2 rows and 1 column
    fig, axes = plt.subplots(2, 1, figsize=(12, 18), sharex=False)

    # Plot bars for each row
    keys = [f'{metric_name} C{i}' for i in range(5)]
    for ax, names_row, values_row in zip(axes, [names_row1, names_row2], [values_row1, values_row2]):
        x_row = np.arange(len(names_row))  # Recalculate positions for the current row
        for i, key in enumerate(keys):
            ax.bar(x_row + i * width, [100 * v[i] for v in values_row], width, label=key)
        for pos in x_row[:-1]:  # Avoid last position
            ax.axvline(pos + width * len(keys), color="gray", linestyle="--", alpha=0.5)

        # Format plot
        ax.set_xticks(x_row + width * (len(keys) - 1) / 2)  # Center labels
        model_names = [n.replace("-type", "").replace(".pth", "") for n in names_row]
        ax.set_xticklabels(model_names, rotation=45, ha="right")
        ax.set_yticks(np.arange(0, 101, 5))  # Ticks from 0 to 1 with step 0.1
        ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.7)
        ax.set_ylabel("Percentual")
        #ax.set_title(f"{metric_name} {set} - Parte {list(axes).index(ax) + 1}")

    # Add legend to the first subplot
    axes[0].legend(title="Classes", loc="upper right")

    # Adjust layout
    plt.tight_layout()

    # Save figure if save_to is provided
    if save_to:
        plt.savefig(save_to, dpi=300, bbox_inches="tight")

    plt.show()

def plot_horizontal_metric_combination(infos, names, keys, metric_names = None, save_to = None):

    import numpy as np
    import matplotlib.pyplot as plt
    values = [[infos[name][k] for k in keys] for name in names]  # Data for each group

    # Bar chart settings
    y = np.arange(len(names))  # Positions for groups
    height = 0.15  # Bar width

    fig, ax = plt.subplots(figsize=(12, 16)) 
    # Plot bars for each key
    for i, key in enumerate(keys):
        if not metric_names:
            ax.barh(y + i * height, [100*v[i] for v in values], height, label=key)
        else:
            ax.barh(y + i * height, [100*v[i] for v in values], height, label=metric_names[i])
            
    for pos in y[:-1]:  # Avoid last position
        ax.axhline(pos + height * len(keys), color="gray", linestyle="--", alpha=0.5)
    # Format plot
    ax.set_yticks(y + height)  # Center labels
    model_names = [n.replace("-type", "").replace(".pth", "") for n in names]
    ax.set_yticklabels(model_names)
    ax.grid(axis="x", linestyle="--", linewidth=0.7, alpha=0.7)
    ax.set_xticks(np.arange(0, 101, 5))  # Ticks from 0 to 1 with step 0.1
    ax.legend(title="Metricas")
    ax.set_xlabel("Percentual")
    ax.set_title("Métricas")
    #ax.set_xticklabels(ax.get_xticks(), rotation=45)  # Rotate labels by 45 degrees
    #ax.set_yticklabels(ax.get_ytickslabels(), rotation=45)  # Rotate labels by 45 degrees
    plt.tight_layout()#rect=[0, 0, 0.85, 1])  # Adjust layout to fit colorbar

    if save_to:
        plt.savefig(save_to, dpi=300, bbox_inches="tight")
    plt.show()

def plot_vertical_metric_combination(infos, names, keys, metric_names=None, save_to=None):
    import numpy as np
    import matplotlib.pyplot as plt

    # Extract data for each group
    values = [[infos[name][k] for k in keys] for name in names]

    # Split data into two rows for better visualization
    half = len(names) // 2
    names_row1, names_row2 = names[:half], names[half:]
    values_row1, values_row2 = values[:half], values[half:]

    # Bar chart settings
    width = 0.15  # Bar width

    # Create subplots with 2 rows and 1 column
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=False)

    # Plot bars for each row
    for ax, names_row, values_row in zip(axes, [names_row1, names_row2], [values_row1, values_row2]):
        x = np.arange(len(names_row))  # Positions for groups
        for i, key in enumerate(keys):
            if not metric_names:
                ax.bar(x + i * width, [100 * v[i] for v in values_row], width, label=key)
            else:
                ax.bar(x + i * width, [100 * v[i] for v in values_row], width, label=metric_names[i])

        # Add vertical lines to separate groups
        for pos in x[:-1]:  # Avoid last position
            ax.axvline(pos + width * len(keys), color="gray", linestyle="--", alpha=0.5)

        # Format plot
        ax.set_xticks(x + width * (len(keys) - 1) / 2)  # Center labels
        model_names = [n.replace("-type", "").replace(".pth", "") for n in names_row]
        ax.set_xticklabels(model_names, rotation=45, ha="right")
        ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.7)
        ax.set_ylabel("Percentual")
        ax.set_title(f"Métricas - Parte {list(axes).index(ax) + 1}")

    # Add legend to the first subplot
    axes[0].legend(title="Métricas", loc="upper right")

    # Adjust layout
    plt.tight_layout()

    # Save figure if save_to is provided
    if save_to:
        plt.savefig(save_to, dpi=300, bbox_inches="tight")

    plt.show()

def plot_vertical_metric_combination_by_suffix(infos, names, keys, suffixes, metric_names=None, save_to=None):
    import numpy as np
    import matplotlib.pyplot as plt

    # Extract data for each group
    values = [[infos[name][k] for k in keys] for name in names]

    # Group names and values by suffix
    grouped_names = {}
    grouped_values = {}
    for name, value in zip(names, values):
        for suffix in suffixes:
            if name.endswith(suffix):
                if suffix not in grouped_names:
                    grouped_names[suffix] = []
                    grouped_values[suffix] = []
                grouped_names[suffix].append(name)
                grouped_values[suffix].append(value)
                break  # Stop checking other suffixes once a match is found

    # Bar chart settings
    width = 0.15  # Bar width

    # Create subplots with one row per suffix
    num_rows = len(grouped_names)
    fig, axes = plt.subplots(num_rows, 1, figsize=(14, 5 * num_rows), sharex=True)

    # Handle the case where there's only one row (axes is not iterable in this case)
    if num_rows == 1:
        axes = [axes]

    # Plot bars for each suffix group
    for ax, suffix in zip(axes, grouped_names.keys()):
        names_row = grouped_names[suffix]
        values_row = grouped_values[suffix]
        x = np.arange(len(names_row))  # Positions for groups

        for i, key in enumerate(keys):
            if not metric_names:
                ax.bar(x + i * width, [100 * v[i] for v in values_row], width, label=key)
            else:
                ax.bar(x + i * width, [100 * v[i] for v in values_row], width, label=metric_names[i])

        # Add vertical lines to separate groups
        for pos in x[:-1]:  # Avoid last position
            ax.axvline(pos + width * len(keys), color="gray", linestyle="--", alpha=0.5)

        # Format plot
        ax.set_xticks(x + width * (len(keys) - 1) / 2)  # Center labels
        model_names = [n.replace("-type", "").replace(".pth", "") for n in names_row]
        ax.set_xticklabels(model_names, rotation=45, ha="right")
        ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.7)
        ax.set_yticks(np.arange(0, 101, 10))  # Ticks from 0 to 1 with step 0.1
        ax.set_ylabel("Percentual")
        ax.set_title(f"'{suffix.replace('-type','').replace('.pth','')}'")

    # Add legend to the first subplot
    axes[0].legend(title="Métricas", loc="upper right")

    # Adjust layout
    plt.tight_layout()

    # Save figure if save_to is provided
    if save_to:
        plt.savefig(save_to, dpi=300, bbox_inches="tight")

    plt.show()


def plot_metric(values, names, suffixes, metric_name=None, save_to=None):
    import numpy as np
    import matplotlib.pyplot as plt

    # Step 1: Calculate unique prefixes by removing suffixes
    prefixes = set()
    for name in names:
        for suffix in suffixes:
            if name.endswith(suffix):
                prefix = name[: -len(suffix)]  # Remove the suffix to get the prefix
                prefixes.add(prefix)
                break  # Stop checking other suffixes once a match is found

    prefixes = sorted(prefixes)  # Sort prefixes for consistent ordering

    # Step 2: Group names and values by suffix and prefix
    grouped_names = {suffix: {prefix: [] for prefix in prefixes} for suffix in suffixes}
    grouped_values = {suffix: {prefix: [] for prefix in prefixes} for suffix in suffixes}

    # Populate the dictionaries based on suffixes and prefixes
    for name, value in zip(names, values):
        for suffix in suffixes:
            if name.endswith(suffix):
                prefix = name[: -len(suffix)]  # Remove the suffix to get the prefix
                grouped_names[suffix][prefix].append(name)
                grouped_values[suffix][prefix].append(value)
                break  # Stop checking other suffixes once a match is found

    # Bar chart settings
    width = 0.15  # Bar width

    # Create subplots with one row per suffix
    num_rows = len(suffixes)
    fig, axes = plt.subplots(num_rows, 1, figsize=(10, 4 * num_rows), sharex=True)

    # Handle the case where there's only one row (axes is not iterable in this case)
    if num_rows == 1:
        axes = [axes]

    keys = [f'{metric_name} C{i}' for i in range(5)]

    # Plot bars for each suffix group
    for ax, suffix in zip(axes, suffixes):
        # Flatten the names and values for plotting, ensuring consistent prefix order
        names_row = []
        values_row = []

        for prefix in prefixes:
            if grouped_names[suffix][prefix]:  # If there are names for this prefix
                names_row.extend(grouped_names[suffix][prefix])
                values_row.extend(grouped_values[suffix][prefix])
            else:  # If no names for this prefix, add placeholders
                names_row.append("")  # Placeholder for missing prefix
                values_row.append([0] * len(keys))  # Placeholder for missing values
        print(names_row)
        x = np.arange(len(names_row))  # Positions for groups

        for i, key in enumerate(keys):
            ax.bar(x + i * width, [100 * v[i] for v in values_row], width, label=key)
            
        # Add vertical lines to separate groups
        for pos in x[:-1]:  # Avoid last position
            ax.axvline(pos + width * len(keys), color="gray", linestyle="--", alpha=0.5)

        # Format plot
        ax.set_xticks(x + width * (len(keys) - 1) / 2)  # Center labels
        model_names = [n.replace("-type", "").replace(".pth", "").replace(suffix, "") if n else "" for n in names_row]
        ax.set_xticklabels(model_names, rotation=45, ha="right")
        ax.set_yticks(np.arange(0, 101, 10))  # Ticks from 0 to 1 with step 0.1
        ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.7)
        ax.set_ylabel("Percentual")
        ax.set_title(f"Configuração: '{suffix.replace('-type-', '').replace('.pth', '')}'")

    # Add legend to the first subplot
    axes[0].legend(title="Métricas", loc="lower left", framealpha=0.5)

    # Adjust layout
    plt.tight_layout()

    # Save figure if save_to is provided
    if save_to:
        plt.savefig(save_to, dpi=300, bbox_inches="tight")
        print('saved to ', save_to)
    plt.show()




def plot_metric_combination(infos, names, keys, suffixes, metric_names=None, save_to=None):
    import numpy as np
    import matplotlib.pyplot as plt

    # Extract data for each group
    values = [[infos[name][k] for k in keys] for name in names]

    # Step 1: Calculate unique prefixes by removing suffixes
    prefixes = set()
    for name in names:
        for suffix in suffixes:
            if name.endswith(suffix):
                prefix = name[: -len(suffix)]  # Remove the suffix to get the prefix
                prefixes.add(prefix)
                break  # Stop checking other suffixes once a match is found

    prefixes = sorted(prefixes)  # Sort prefixes for consistent ordering

    # Step 2: Group names and values by suffix and prefix
    grouped_names = {suffix: {prefix: [] for prefix in prefixes} for suffix in suffixes}
    grouped_values = {suffix: {prefix: [] for prefix in prefixes} for suffix in suffixes}

    # Populate the dictionaries based on suffixes and prefixes
    for name, value in zip(names, values):
        for suffix in suffixes:
            if name.endswith(suffix):
                prefix = name[: -len(suffix)]  # Remove the suffix to get the prefix
                grouped_names[suffix][prefix].append(name)
                grouped_values[suffix][prefix].append(value)
                break  # Stop checking other suffixes once a match is found

    # Bar chart settings
    width = 0.15  # Bar width

    # Create subplots with one row per suffix
    num_rows = len(suffixes)
    fig, axes = plt.subplots(num_rows, 1, figsize=(10, 4 * num_rows), sharex=True)

    # Handle the case where there's only one row (axes is not iterable in this case)
    if num_rows == 1:
        axes = [axes]

    # Plot bars for each suffix group
    for ax, suffix in zip(axes, suffixes):
        # Flatten the names and values for plotting, ensuring consistent prefix order
        names_row = []
        values_row = []

        for prefix in prefixes:
            if grouped_names[suffix][prefix]:  # If there are names for this prefix
                names_row.extend(grouped_names[suffix][prefix])
                values_row.extend(grouped_values[suffix][prefix])
            else:  # If no names for this prefix, add placeholders
                names_row.append("")  # Placeholder for missing prefix
                values_row.append([0] * len(keys))  # Placeholder for missing values
        print(names_row)
        x = np.arange(len(names_row))  # Positions for groups

        for i, key in enumerate(keys):
            if not metric_names:
                ax.bar(x + i * width, [100 * v[i] for v in values_row], width, label=key)
            else:
                ax.bar(x + i * width, [100 * v[i] for v in values_row], width, label=metric_names[i])

        # Add vertical lines to separate groups
        for pos in x[:-1]:  # Avoid last position
            ax.axvline(pos + width * len(keys), color="gray", linestyle="--", alpha=0.5)

        # Format plot
        ax.set_xticks(x + width * (len(keys) - 1) / 2)  # Center labels
        model_names = [n.replace("-type", "").replace(".pth", "").replace(suffix, "") if n else "" for n in names_row]
        ax.set_xticklabels(model_names, rotation=45, ha="right")
        ax.set_yticks(np.arange(0, 101, 10))  # Ticks from 0 to 1 with step 0.1
        ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.7)
        ax.set_ylabel("Percentual")
        ax.set_title(f"Configuração: '{suffix.replace('-type-', '').replace('.pth', '')}'")

    # Add legend to the first subplot
    axes[0].legend(title="Métricas", loc="lower left", framealpha=0.5)

    # Adjust layout
    plt.tight_layout()

    # Save figure if save_to is provided
    if save_to:
        plt.savefig(save_to, dpi=300, bbox_inches="tight")

    plt.show()