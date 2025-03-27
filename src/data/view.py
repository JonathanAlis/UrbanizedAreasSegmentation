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

def get_rgb_np(multi_channel_image: np.ndarray):
    if multi_channel_image.shape[0] != 3:
        rgb_image = multi_channel_image[1:4, :, :]
        rgb_image = np.transpose(rgb_image, (1, 2, 0))  # Permute dimensions
        return rgb_image
    
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

import numpy as np

def calculate_recalls(confusion_matrix):
    confusion_matrix = np.array(confusion_matrix)
    num_classes = confusion_matrix.shape[0]
    TP = np.diag(confusion_matrix)  # True Positives are the diagonal elements
    FN = confusion_matrix.sum(axis=1) - TP  # False Negatives are row sums minus TP
    class_totals = confusion_matrix.sum(axis=1)  # Total examples per class (row sums)
    total_examples = confusion_matrix.sum()  # Total examples in the dataset
    recalls = TP / (TP + FN)  # Recall = TP / (TP + FN)
    macro_recall = np.mean(recalls)
    weights = class_totals / total_examples
    weighted_recall = np.sum(weights * recalls)
    global_recall = TP.sum() / total_examples
    return {
        'macro_recall': macro_recall,
        'weighted_recall': weighted_recall,
        'global_recall': global_recall
    }

import torch
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cm: torch.Tensor, classes_list = [2, 4, 5], class_names=None, cmap="Blues", save_to = None):
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
    norm_cm = cm / cm.sum(dim=1, keepdim=True)
    norm_cm = norm_cm.nan_to_num(0)

    # Indices for merged classes
    group_0 = [0, 3]  # Merging classes 0 and 3
    group_1 = [1, 2, 4]  # Merging classes 1, 2, and 4
    binary_cm = torch.zeros(2, 2, dtype=torch.float32)
    binary_cm[0, 0] = cm[group_0, :][:, group_0].sum()  # (0,3) predicted as (0,3)
    binary_cm[0, 1] = cm[group_0, :][:, group_1].sum()  # (0,3) predicted as (1,2,4)
    binary_cm[1, 0] = cm[group_1, :][:, group_0].sum()  # (1,2,4) predicted as (0,3)
    binary_cm[1, 1] = cm[group_1, :][:, group_1].sum()  # (1,2,4) predicted as (1,2,4)

    binary_norm_cm = binary_cm / binary_cm.sum(dim=1, keepdim=True)
    binary_norm_cm = binary_norm_cm.nan_to_num(0)
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
    join_norm_cm = join_cm / join_cm.sum(dim=1, keepdim=True)
    join_norm_cm = join_norm_cm.nan_to_num(0)
    num_classes = cm.shape[0]
    class_labels = class_names if class_names else list(range(num_classes))
    class_labels = [str(cl) for cl in class_labels]
    join_labels = ["(0,3)", "(1)", "(2)", "(4)"]


    # Create subplot
    num_plots = 0
    plot_idxs = []
    cms = []
    titles = []
    labels_ = []
    recalls = {}
    if 2 in classes_list:
        plot_idxs.append(num_plots)
        num_plots+=1        
        cms.append(binary_norm_cm)
        titles.append("Binary Confusion Matrix")
        labels_.append(binary_labels)
        recalls['binary'] = calculate_recalls(binary_cm)
        for i, bl in enumerate(binary_labels):
            recalls['binary'][bl+'_recall']=binary_norm_cm[i,i]

    if 4 in classes_list:
        plot_idxs.append(num_plots)
        num_plots+=1        
        cms.append(join_norm_cm)
        titles.append("4-Class Confusion Matrix")
        labels_.append(join_labels)
        recalls['4class'] = calculate_recalls(join_cm)
        for i, jl in enumerate(join_labels):
            recalls['4class'][jl+'_recall']=join_norm_cm[i,i]
    if 5 in classes_list:
        plot_idxs.append(num_plots)
        num_plots+=1        
        cms.append(norm_cm)
        titles.append("5-Class Confusion Matrix")
        labels_.append(class_labels)
        recalls['5class'] = calculate_recalls(cm)
        for i, l in enumerate(class_labels):
            recalls['5class'][l+'_recall']=norm_cm[i,i]


    fig, axes = plt.subplots(1, num_plots, figsize=(4*num_plots, 4))

    for plot_idx, title, confusion_matrix, labels in zip(plot_idxs, titles, cms, labels_): 
        try:
            ax = axes[plot_idx]
        except:
            ax = axes
        sns.heatmap(confusion_matrix.numpy(), annot_kws={"size": 16}, annot=True, fmt=".2f", cmap=cmap, linewidths=1, square=True, ax=ax, cbar=False)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=14)
        ax.set_yticklabels(labels, rotation=0, fontsize=14)
        ax.set_xlabel("Predicted Label", fontsize=18)
        ax.set_ylabel("True Label", fontsize=18)
        #ax.set_title(title)

    plt.tight_layout()
    
    if save_to:
        print(f'Saving to {save_to}')
        plt.savefig(save_to, bbox_inches='tight', pad_inches=0)
    plt.show()
    return recalls

def plot_confusion_matrix_simple(cm: torch.Tensor, cmap="Blues", save_to = None):
    """
    Plots two confusion matrices: one with merged classes (binary) and one with all classes.

    Args:
        cm (torch.Tensor): The confusion matrix of shape [num_classes, num_classes].
        class_names (list, optional): List of class names corresponding to indices.
        normalize (bool): Whether to normalize the confusion matrix to percentages.
        cmap (str): Color map for visualization.
    """
    
    cm = cm.cpu()  # Ensure it's a CPU tensor for plotting
    
    # Normalize
    norm_cm = cm / cm.sum(dim=1, keepdim=True)
    norm_cm = norm_cm.nan_to_num(0)
    labels = list(range(norm_cm.shape[0]))

    num_plots = 1
    fig, axes = plt.subplots(1, num_plots, figsize=(4*num_plots, 4))
    ax = axes
    sns.heatmap(norm_cm.numpy(), annot_kws={"size": 16}, annot=True, fmt=".2f", cmap=cmap, linewidths=1, square=True, ax=ax, cbar=False)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=14)
    ax.set_yticklabels(labels, rotation=0, fontsize=14)
    ax.set_xlabel("Predicted Label", fontsize=18)
    ax.set_ylabel("True Label", fontsize=18)
    
    plt.tight_layout()
    
    if save_to:
        print(f'Saving to {save_to}')
        plt.savefig(save_to, bbox_inches='tight', pad_inches=0)
    plt.show()
    return norm_cm

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

def plot_single_confusion_matrix(cm: torch.Tensor, class_names=None, cmap="Blues", save_to = None):
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
    norm_cm = cm / cm.sum(dim=1, keepdim=True)
    norm_cm = norm_cm.nan_to_num(0)

    fig, ax = plt.subplots(figsize=(4, 4))

    sns.heatmap(norm_cm.numpy(), annot_kws={"size": 16}, annot=True, fmt=".2f", cmap=cmap, linewidths=1, square=True, ax=ax, cbar=False)
    #ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=14)
    #ax.set_yticklabels(labels, rotation=0, fontsize=14)
    ax.set_xlabel("Predicted Label", fontsize=18)
    ax.set_ylabel("True Label", fontsize=18)
        #ax.set_title(title)

    plt.tight_layout()
    
    if save_to:
        print(f'Saving to {save_to}')
        plt.savefig(save_to, bbox_inches='tight', pad_inches=0)
    plt.show()
    
def plot_confusion_matrix_simple(cm: torch.Tensor, cmap="Blues", save_to = None):
    """
    Plots two confusion matrices: one with merged classes (binary) and one with all classes.

    Args:
        cm (torch.Tensor): The confusion matrix of shape [num_classes, num_classes].
        class_names (list, optional): List of class names corresponding to indices.
        normalize (bool): Whether to normalize the confusion matrix to percentages.
        cmap (str): Color map for visualization.
    """
    
    cm = cm.cpu()  # Ensure it's a CPU tensor for plotting
    
    # Normalize
    norm_cm = cm / cm.sum(dim=1, keepdim=True)
    norm_cm = norm_cm.nan_to_num(0)
    labels = list(range(norm_cm.shape[0]))

    num_plots = 1
    fig, axes = plt.subplots(1, num_plots, figsize=(4*num_plots, 4))
    ax = axes
    sns.heatmap(norm_cm.numpy(), annot_kws={"size": 16}, annot=True, fmt=".2f", cmap=cmap, linewidths=1, square=True, ax=ax, cbar=False)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=14)
    ax.set_yticklabels(labels, rotation=0, fontsize=14)
    ax.set_xlabel("Predicted Label", fontsize=18)
    ax.set_ylabel("True Label", fontsize=18)
    
    plt.tight_layout()
    
    if save_to:
        print(f'Saving to {save_to}')
        plt.savefig(save_to, bbox_inches='tight', pad_inches=0)
    plt.show()
    return norm_cm

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

def plot_masked_image(mask, label_map, image=None, title="Mask Overlay", ax=None, figsize = (10,8)):
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
        fig, ax = plt.subplots(figsize=figsize)
    
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


def plot_mask_list(masks, titles = None, background = None, save_to = None, num_classes = 5):
    
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.colors as mcolors

    cols = min(3, math.floor(math.sqrt(len(masks))))
    
    rows = math.ceil(len(masks)/cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))  # Adjust figure size
    axes = axes.flatten()  # Flatten for easy iteration

    if num_classes == 5:
        class_colors = {
            0: "#000000",  # Black (Background)
            1: "#0000FF",  # Cyan
            2: "#FF00FF",  # Magenta
            3: "#FFFF00",  # Yellow
            4: "#FFFFFF"   # White
        }
        class_labels = {
            0: "Fundo",
            1: "Loteamento vazio",
            2: "Outros equipamentos",
            3: "Vazio intraurbano",
            4: "Área Urbanizada"
        }

    elif num_classes == 4:
        class_colors = {
            0: "#000000",  # Black (Background)
            1: "#0000FF",  # Cyan
            2: "#FFFF00",  # Yellow
            3: "#FFFFFF"   # White
        }
        class_labels = {
            0: "Fundo",
            1: "Loteamento vazio",
            2: "Outros equipamentos",
            3: "Área Urbanizada"
        }

    cmap = mcolors.ListedColormap([class_colors[i] for i in sorted(class_colors.keys())])
    bounds = sorted(class_colors.keys()) + [max(class_colors.keys()) + 1]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    if background is not None:
        pass#background = raster_to_rgb(background)
    if background is not None:
        for i in range(3):
            min_val, max_val = np.nanpercentile(background[i], (2,98))
            # Avoid division by zero
            if max_val > min_val:
                background[i] = np.clip((background[i] - min_val) / (max_val - min_val), 0, 1)
            else:
                background[i] = np.zeros_like(background[i])
            background[i] = (background[i]*255).astype(np.uint8)
    for i, (ax, mask) in enumerate(zip(axes,masks)):

        if background is not None:
            bg_mask = mask == 0
            
            output = background.copy()#np.stack([mask] * 3, axis=-1)
            
            import cv2
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            r, g, b = cv2.split(output)
            r_clahe = clahe.apply(r)
            g_clahe = clahe.apply(g)
            b_clahe = clahe.apply(b)
            output = cv2.merge([r_clahe, g_clahe, b_clahe])

            for classe in range(1,num_classes):
                mask_classe = mask == classe
                color_rgb = tuple(int(class_colors[classe][i:i+2], 16) for i in (1, 3, 5))  # (255, 255, 0)
                output[mask_classe] = color_rgb

            im = ax.imshow(output)#, cmap=cmap, extent=[0, mask.shape[1], mask.shape[0], 0], alpha=1)
        else:
            im = ax.imshow(mask, cmap=cmap, extent=[0, mask.shape[1], mask.shape[0], 0], alpha=1)
        
        if titles is not None:
            ax.set_title(titles[i])
        ax.axis("off")


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

        # Add text annotations above each group of bars
        if 0:
            for i, prefix in enumerate(prefixes):
                group_start = i * len(keys)  # Start index of the group
                group_end = group_start + len(keys)  # End index of the group
                group_values = [100 * v[j] for v in values_row[group_start:group_end] for j in range(len(keys))]
                group_mean = np.mean(group_values)  # Mean value of the group
                group_max = np.max(group_values)  # Max value of the group

                # Position for the text annotations
                x_pos = x[group_start] + width * (len(keys) - 1) / 2  # Center of the group
                y_pos_mean = group_mean + 5  # Slightly above the mean
                y_pos_max = group_max + 10  # Higher up for the max value

                # Add the mean value annotation
                ax.text(
                    x_pos, y_pos_mean, f"Mean: {group_mean:.1f}", ha="center", fontsize=8, color="blue"
                )

                # Add the max value annotation
                ax.text(
                    x_pos, y_pos_max, f"Max: {group_max:.1f}", ha="center", fontsize=8, color="red"
                )
        # Format plot
        ax.set_xticks(x + width * (len(keys) - 1) / 2)  # Center labels
        model_names = [n.replace("-type", "").replace(".pth", "").replace(suffix, "").replace("-DS-CEW","") if n else "" for n in names_row]
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
        model_names = [n.replace("-type", "").replace(".pth", "").replace(suffix, "").replace("-DS-CEW", "") if n else "" for n in names_row]
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



import numpy as np
import rasterio
import os

def raster_to_rgb(
    raster_input,
    rgb_bands=[3, 2, 1],  # Default RGB bands (4,3,2 in 1-indexed like QGIS)
    contrast_enhancement="StdDev",
    min_max_values=None,
    std_dev_factor=2.0,  # Default standard deviation factor in QGIS
    cumulative_count_cut=(2.0, 98.0)  # Default percentile values in QGIS
):
    """
    Convert a multi-band raster to RGB image array with contrast enhancement similar to QGIS.
    
    Parameters:
    -----------
    raster_input : str or numpy.ndarray
        Either path to raster file or numpy array containing raster data.
        If numpy array, should have shape (bands, height, width)
    rgb_bands : list
        List of 3 band indices to use for R, G, B (0-indexed)
    contrast_enhancement : str
        Type of contrast enhancement: 'MinMax', 'StdDev', 'Cumulative', or 'NoEnhancement'
    min_max_values : list or None
        List of (min, max) values for each band. If None, calculated from data
    std_dev_factor : float
        Factor for standard deviation enhancement (default is 2.0 as in QGIS)
    cumulative_count_cut : tuple
        Lower and upper percentile values for cumulative count cut (2-98% by default in QGIS)
        
    Returns:
    --------
    numpy.ndarray
        The processed RGB image array with shape (height, width, 3)
    """
    # Validate rgb_bands input
    if len(rgb_bands) != 3:
        raise ValueError("rgb_bands must contain exactly 3 band indices")
    
    # Handle different input types
    if isinstance(raster_input, str):
        # Input is a file path
        if not os.path.exists(raster_input):
            raise FileNotFoundError(f"Raster file not found: {raster_input}")
            
        with rasterio.open(raster_input) as src:
            # Check band indices are valid
            if max(rgb_bands) >= src.count:
                raise ValueError(f"Band index out of range. Available bands: 0-{src.count-1}")
            
            # Read selected bands
            rgb = np.vstack([src.read(b+1) for b in rgb_bands])  # +1 because rasterio is 1-indexed
            
            # Handle nodata values if present
            nodata = src.nodata
            if nodata is not None:
                mask = np.logical_or.reduce([band == nodata for band in rgb])
                for i in range(3):
                    # Set nodata pixels to NaN for proper handling
                    rgb[i][mask] = np.nan
    
    elif isinstance(raster_input, np.ndarray):
        # Input is a numpy array
        if raster_input.ndim < 3:
            raise ValueError("Numpy array input must be 3D with shape (bands, height, width)")
        
        # Check band indices are valid
        if max(rgb_bands) >= raster_input.shape[0]:
            raise ValueError(f"Band index out of range. Available bands: 0-{raster_input.shape[0]-1}")
        
        # Extract selected bands
        rgb = np.vstack([raster_input[b:b+1] for b in rgb_bands])
    
    else:
        raise TypeError("raster_input must be either a file path (str) or a numpy array")
    
    # Apply contrast enhancement
    rgb_stretched = np.zeros_like(rgb, dtype=np.float32)
    
    if contrast_enhancement == "NoEnhancement":
        # No enhancement, just normalize to 0-1 if needed
        for i in range(3):
            if min_max_values is not None:
                min_val, max_val = min_max_values[i]
            else:
                min_val, max_val = np.nanmin(rgb[i]), np.nanmax(rgb[i])
            
            # Avoid division by zero
            if max_val > min_val:
                rgb_stretched[i] = np.clip((rgb[i] - min_val) / (max_val - min_val), 0, 1)
            else:
                rgb_stretched[i] = np.zeros_like(rgb[i])
    
    elif contrast_enhancement == "MinMax":
        # Linear min-max stretch
        for i in range(3):
            if min_max_values is not None:
                min_val, max_val = min_max_values[i]
            else:
                min_val, max_val = np.nanmin(rgb[i]), np.nanmax(rgb[i])
            
            # Avoid division by zero
            if max_val > min_val:
                rgb_stretched[i] = np.clip((rgb[i] - min_val) / (max_val - min_val), 0, 1)
            else:
                rgb_stretched[i] = np.zeros_like(rgb[i])
    
    elif contrast_enhancement == "StdDev":
        # Standard deviation stretch (QGIS default)
        for i in range(3):
            mean = np.nanmean(rgb[i])
            std = np.nanstd(rgb[i])
            
            min_val = mean - (std_dev_factor * std)
            max_val = mean + (std_dev_factor * std)
            
            rgb_stretched[i] = np.clip((rgb[i] - min_val) / (max_val - min_val), 0, 1)
    
    elif contrast_enhancement == "Cumulative":
        # Percentile-based stretch (similar to QGIS's cumulative count cut)
        min_percent, max_percent = cumulative_count_cut
        for i in range(3):
            min_val, max_val = np.nanpercentile(rgb[i], (min_percent, max_percent))
            
            # Avoid division by zero
            if max_val > min_val:
                rgb_stretched[i] = np.clip((rgb[i] - min_val) / (max_val - min_val), 0, 1)
            else:
                rgb_stretched[i] = np.zeros_like(rgb[i])
    
    # Replace NaNs with zeros in the final output
    rgb_stretched = np.nan_to_num(rgb_stretched)
    
    # Transpose for proper shape (bands last)
    rgb_display = np.transpose(rgb_stretched, (1, 2, 0))
    
    return rgb_display
# Example usage:
# rgb_image = raster_to_rgb('path/to/file.tif', rgb_bands=[3, 2, 1])