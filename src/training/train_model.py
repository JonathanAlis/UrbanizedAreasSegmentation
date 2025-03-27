import itertools
import os
import torch
from tqdm import tqdm
import pandas as pd
import csv
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_lr_finder import LRFinder
import torch.optim as optim
import numpy as np
import time
import copy 
import math 

import src.data.utils as utils
import src.data.view as view

def iterate_parameter_grid(param_grid):
    """Iterates over a grid of parameters, yielding all possible combinations.

    Args:
        param_grid: A dictionary where keys are parameter names and values are lists of possible values.

    Yields:
        A dictionary containing a combination of parameters.
    """

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    for params in itertools.product(*values):
        yield dict(zip(keys, params))

def focal_loss(outputs, targets, weights = None, alpha=0.25, gamma=2.0, ):
    ce_loss = F.cross_entropy(outputs, targets, weight = weights, reduction='none')  # Cross-entropy loss
    pt = torch.exp(-ce_loss)  # Probability of the true class
    focal_loss = (alpha * (1 - pt) ** gamma * ce_loss).mean()  # Focal loss
    return focal_loss

import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedLoss(nn.Module):
    def __init__(self, loss_mode = 'CE', weights = None , return_all_losses = False):#, groupings):
        """
        Initialize the combined loss function.

        Parameters:
        weights (float): Weighting factor for each class for use on Cross-Entropy Loss.
        loss_mode (str): CE, dice, CE-dice, groups

        """
        super(CombinedLoss, self).__init__()
        if weights is not None:
            self.weights = torch.Tensor([w/sum(weights) for w in weights])
        else:
            self.weights = None
        self.loss_mode = loss_mode
        self.return_all_losses = return_all_losses
        assert loss_mode in ['CE', 'dice', 'CE-dice', 'groups'] or loss_mode.startswith('macroF')

        if self.loss_mode=='CE':
            self.alpha = 1.0
        if self.loss_mode=='dice':
            self.alpha = 0.0
        if self.loss_mode=='CE-dice':
            self.alpha = 0.5
        if self.loss_mode=='macroF1':
            pass
        if self.loss_mode=='groups':
            self.groupings = {
                "5-group": {
                    0: [0],      # Group 1: Class 1 + 4
                    1: [1],      # Group 2: Class 2
                    2: [2],      # Group 3: Class 3
                    3: [3],       # Group 4: Class 5
                    4: [4]       # Group 4: Class 5
                },

                "4-group": {
                    0: [0, 3],   # Group 1: Class 1 + 4
                    1: [1],      # Group 2: Class 2
                    2: [2],      # Group 3: Class 3
                    3: [4]       # Group 4: Class 5
                },
                "3-group": {
                    0: [0, 3],   # Group 1: Class 1 + 4
                    1: [1, 2],   # Group 2: Class 2 + 3
                    2: [4]       # Group 3: Class 5
                },
                "2-group": {
                    0: [0, 3],   # Group 1: Class 0 + 3
                    1: [1, 2, 4],   # Group 2: Class 1 + 2 + 4
                },
            }
        
    def set_return(self, return_all_losses):
        self.return_all_losses = return_all_losses

    def group_loss(self, y_pred, y_true):
        """
        Dice Loss per group
        First get the groups of classes, given by self.groupings
        """
        device = y_pred.device
        preds_groups = {}
        label_groups = {}
        losses_group = {}
        for g, group in self.groupings.items():
            pred_group = {}
            for k,v in group.items():
                selected = torch.index_select(y_pred, dim=1, index=torch.tensor(v, device=device))
                summed = selected.sum(dim=1, keepdim=True)
                torch.sum(y_pred[:,v,:,:])
                pred_group[k] = summed
            #print([pred_group[i].shape for i in range(len(pred_group))])
            preds_groups[g] = torch.cat([pred_group[i] for i in range(len(pred_group))], dim=1)

            new_tensor = y_pred.clone()
            for key, values in group.items():
                mask = torch.isin(new_tensor, torch.tensor(values, device=device))  # Find elements to replace
                new_tensor = torch.where(mask, torch.tensor(key), new_tensor) 
            label_groups[g] = new_tensor
            losses_group[g] = self.multiclass_dice_loss(y_pred, y_true)
        return losses_group


    def forward(self, y_pred, y_true):
        """
        Compute the combined loss.

        Parameters:
        y_pred (torch.Tensor): Predicted logits, shape (N, C, ...).
        y_true (torch.Tensor): Ground truth labels, shape (N, ...).

        Returns:
        torch.Tensor: The combined loss value.
        """

        # Ensure y_true is 3D (N, H, W)
        if y_true.dim() == 4:  # If y_true is (N, 1, H, W)
            y_true = y_true.squeeze(1)  # Remove the extra dimension
        elif y_true.dim() != 3:  # If y_true is not 3D
            raise RuntimeError(f"y_true must be 3D (N, H, W) or 4D (N, 1, H, W). Got shape: {y_true.shape}")
        
        if self.weights is not None:
            self.weights = self.weights.to(y_pred.device)
        ce_loss = F.cross_entropy(y_pred, y_true, weight=self.weights)
        dice_loss = self.multiclass_dice_loss(y_pred, y_true)
            
        if self.loss_mode.startswith('macroF'):
            beta = int(self.loss_mode.split('macroF')[1])
            macroFLoss = self.macro_fbeta_loss(y_pred, y_true, beta = beta)
            if self.return_all_losses:
                return macroFLoss, ce_loss, dice_loss
            else: 
                return macroFLoss

        if self.loss_mode=='groups':
            losses_group = self.group_loss(y_pred, y_true)
            combined_loss = 0.8 * ce_loss + 0.2* sum([loss for k, loss in losses_group.items()])
            if self.return_all_losses:
                return combined_loss, ce_loss, losses_group["5-group"]
            else: 
                return combined_loss
        else:
            combined_loss = self.alpha * ce_loss + (1 - self.alpha) * dice_loss
            if self.return_all_losses:
                return combined_loss, ce_loss, dice_loss
            else: 
                return combined_loss

    def multiclass_dice_loss(self, y_pred, y_true):
        """
        Compute the multiclass Dice Loss.

        Parameters:
        y_pred (torch.Tensor): Predicted logits, shape (N, C, H, W).
        y_true (torch.Tensor): Ground truth labels, shape (N, H, W) or (N, 1, H, W).

        Returns:
        torch.Tensor: The Dice Loss value.
        """        

        # Ensure y_true contains valid class indices
        if y_true.min() < 0 or y_true.max() >= y_pred.size(1):
            raise RuntimeError(f"y_true contains invalid class indices. Expected values in [0, {y_pred.size(1) - 1}].")

        # Convert y_true to one-hot encoding
        num_classes = y_pred.size(1)
        y_true_one_hot = F.one_hot(y_true, num_classes).permute(0, 3, 1, 2).float()  # Shape: (N, C, H, W)

        # Apply softmax to y_pred to get probabilities
        y_pred_softmax = F.softmax(y_pred, dim=1) #sigmoid?

        # Compute intersection and union
        intersection = torch.sum(y_pred_softmax * y_true_one_hot, dim=(2, 3))  # Sum over spatial dimensions
        union = torch.sum(y_pred_softmax + y_true_one_hot, dim=(2, 3))

        # Compute Dice coefficient
        dice = (2.0 * intersection + 1e-6) / (union + 1e-6)  # Add epsilon to avoid division by zero

        # Compute Dice Loss
        dice_loss = 1.0 - dice.mean()  # Average over classes and batch

        return dice_loss



    def macro_fbeta_loss(self, pred, target, beta = 2.0, class_weights=None, smooth=1e-6):
        """
        Calcula o 1 - Soft Macro F1-score como loss, ponderado por pesos de classe.

        Args:
            pred: Tensor de probabilidades [batch, num_classes, H, W].
            target: Tensor de índices de classe [batch, H, W].
            class_weights: Tensor de pesos [num_classes] ou None para pesos iguais.
            smooth: Pequeno valor para suavização numérica.

        Returns:
            Loss baseada no Macro F1-score ponderado.
        """
        num_classes = pred.shape[1]
        
        # Converter target para one-hot
        target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        # Softmax para normalizar pred (se ainda não for probabilidades)
        pred = F.softmax(pred, dim=1)
        
        # TP, FP, FN para cada classe
        tp = (pred * target_one_hot).sum(dim=(0, 2, 3))
        fp = ((1 - target_one_hot) * pred).sum(dim=(0, 2, 3))
        fn = (target_one_hot * (1 - pred)).sum(dim=(0, 2, 3))
        beta_sq = beta ** 2
        soft_fbeta = ((1 + beta_sq) * tp + smooth) / ((1 + beta_sq) * tp + beta_sq * fn + fp + smooth)
    
        # Aplicar pesos se fornecidos
        if self.weights is not None:
            class_weights = self.weights.to(pred.device)  # Garantir que os pesos estão no mesmo device
            macro_fbeta = (soft_fbeta * class_weights).sum() / class_weights.sum()
        else:
            macro_fbeta = soft_fbeta.mean()

        return 1 - macro_fbeta  # Loss, pois queremos minimizar



def run_epoch(mode:str, model, dataloader, criterion = None, num_classes = None, optimizer=None, simulated_batch_size:int = 64, device:str=torch.device('cpu'), return_report:bool = False, show_pred:bool|int = False, yield_data:bool = False):
    # Alpha [0-1] is the weight of losses: 0 for cross entropy, 1 for dice
    print(f'Running an epoch on the {mode} mode.')
    if mode == 'train':
        model.train()
    else:
        model.eval()

    metrics_tracker = Metrics(num_classes=num_classes)
    accumulation_steps = simulated_batch_size//dataloader.batch_size #eg 64//4 = 16, accumulate gradients for 16 iterations

    # Disable gradient tracking for eval or test mode
    with torch.set_grad_enabled(mode == 'train'):
        optimizer.zero_grad() 
        for i, (images, labels) in enumerate(tqdm(dataloader)):
            #load and prepare the data
            images=images.to(device)
            labels=labels.to(device)
            logits = model(images)
            pred = F.softmax(logits, dim=1)
            metrics_tracker.track_preds(labels, pred)
            
            loss, CE, dice = criterion(logits, labels)
            loss, CE, dice = loss/accumulation_steps, CE/accumulation_steps, dice/accumulation_steps
            if mode == 'train':
                # Backward pass and optimize, if training
                loss.backward()
                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()  # Update model parameters
                    optimizer.zero_grad()  # Reset gradients

            if show_pred:
                try:
                    view.show_batches(images, labels, pred, max_shown=4)
                    #for j in range(images.shape[0]):  # Iterate over each image in the batch
                    #    yield images[j].cpu(), labels[j].cpu(), pred[j].cpu()
                    if isinstance(show_pred, int):
                        show_pred-=1
                except:
                    print("Error in the image visualization.")
            if yield_data:
                yield images, labels, logits
            metrics_tracker.store_metrics(loss.item(), CE.item(), dice.item(), batch_size=images.shape[0])
            
    avg_loss, avg_CE, avg_dice = metrics_tracker.avg_metrics()
    if return_report:
        report, acc, cm = metrics_tracker.class_report()
        return avg_loss, avg_CE, avg_dice, report, acc, cm

    return avg_loss, avg_CE, avg_dice

class EpochRunner:
    def __init__(self, mode:str, model, dataloader, criterion = None, num_classes = None, optimizer=None, simulated_batch_size:int = 64, device:str=torch.device('cpu')):
        self.mode = mode
        self.model = model
        self.dataloader = dataloader
        self.criterion = CombinedLoss(loss_mode = 'CE', weights = None, return_all_losses=True)

        self.num_classes = num_classes
        self.optimizer = optimizer
        self.simulated_batch_size = simulated_batch_size
        self.device = device

        self.metrics_tracker = Metrics(num_classes=num_classes)
        self.accumulation_steps = simulated_batch_size // dataloader.batch_size
        self.finish = False

    def run_generator(self, show_pred:bool|int = False):
        """ Runs the epoch, yielding data if enabled. Stores final metrics internally. """
        print("running...")
        self.model.train() if self.mode == 'train' else self.model.eval()
        with torch.set_grad_enabled(self.mode == 'train'):
            if self.mode == 'train':
                self.optimizer.zero_grad()            
            for i, batch in enumerate(tqdm(self.dataloader)):
                if len(batch)==2:
                    images, labels = batch
                if len(batch)==5:
                    images, labels, x, y, f = batch
                images, labels = images.to(self.device), labels.to(self.device)
                logits = self.model(images)
                pred = F.softmax(logits, dim=1)                
                self.metrics_tracker.track_preds(labels, pred)
                if self.criterion:
                    loss, CE, dice = self.criterion(logits, labels)
                    loss, CE, dice = loss / self.accumulation_steps, CE / self.accumulation_steps, dice / self.accumulation_steps

                    if self.mode == 'train':
                        loss.backward()
                        if (i + 1) % self.accumulation_steps == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()

                    self.metrics_tracker.store_metrics(loss.item(), CE.item(), dice.item(), batch_size=images.shape[0])
                
                self.metrics_tracker.update_batch_count(batch_size=images.shape[0]) 
                # 🔹 Show predictions if required
                if show_pred: # if true, show in all batches, if a number n, show only n first batches
                    try:
                        view.show_batches(images, labels, pred, max_shown=4)
                        if isinstance(show_pred, int):
                            show_pred -= 1
                    except Exception as e:
                        print(f"Error in visualization: {e}")
                if len(batch)==2:
                    yield images, labels, logits, pred
                if len(batch)==5:
                    yield images, labels, logits, pred, x, y, f 
        self.finish = True
        # 🔹 Store final metrics internally
    
    def run(self, show_pred: bool | int = False):
        """Runs the epoch non-generator style: consumes the generator and discards yielded data."""
        # Consume the generator
        for _ in self.run_generator(show_pred=show_pred):
            pass
    
    def get_metrics(self):
        """ Returns stored metrics after run() has been called. """
        if self.finish:
            self.avg_loss, self.avg_CE, self.avg_dice = self.metrics_tracker.avg_metrics()
            self.report, self.acc, self.cm = self.metrics_tracker.class_report()
            self.final_metrics = (self.avg_loss, self.avg_CE, self.avg_dice, self.report, self.acc, self.cm)
            return self.final_metrics        
        else:
            raise RuntimeError("Error: You must call run() first before getting metrics.")





def inference(model, dataloader, num_classes = None, simulated_batch_size:int = 64, device:str=torch.device('cpu'), return_report:bool = False):

    metrics_tracker = Metrics(num_classes=num_classes)
    accumulation_steps = simulated_batch_size//dataloader.batch_size #eg 64//4 = 16, accumulate gradients for 16 iterations

    model.eval()
    for i, (images, labels) in enumerate(tqdm(dataloader)):
        #load and prepare the data
        images=images.to(device)
        labels=labels.to(device)
        
        logits = model(images)
        pred = F.softmax(logits, dim=1)  
        metrics_tracker.track_preds(labels, pred)
        
        
        for j in range(images.shape[0]):  # Iterate over each image in the batch
            yield (
                images[j].cpu(),
                labels[j].cpu(),
                pred[j].cpu(),
            ) 
        
    avg_loss, avg_CE, avg_dice = metrics_tracker.avg_metrics()
    if return_report:
        report, acc = metrics_tracker.class_report()
        return avg_loss, avg_CE, avg_dice, report, acc

    return avg_loss, avg_CE, avg_dice


def calculate_iterations(lr_min, lr_max, steps_per_order=50):
    num_orders = math.log10(lr_max) - math.log10(lr_min)
    total_iterations = math.ceil(steps_per_order * num_orders)
    return total_iterations

def update_lr(model, train_loader, val_loader, optimizer, criterion, num_iter=100, start_lr=1e-6, end_lr=1e-2):
    criterion.set_return(return_all_losses=False)

    print('Finding learning rate: ')
    lr_finder = LRFinder(model, optimizer, criterion)
    num_iter = min(num_iter, calculate_iterations(lr_min = start_lr, lr_max=end_lr))
    lr_finder.range_test(train_loader, #val_loader=val_loader, 
                         start_lr=start_lr, 
                         end_lr=end_lr, 
                         num_iter=num_iter, 
                         step_mode="exp")
    
    try:
        ax, best_lr = lr_finder.plot(suggest_lr = True) 
    except Exception as e:
        best_lr = end_lr
        print(f"Error finding learning rate: {e}")
        print("Using start_lr=", start_lr)

    print('Chosen LR: ', best_lr)
    lr_finder.reset() #reset model and optimizer before calling lr_finder
    criterion.set_return(return_all_losses=True)
    for param_group in optimizer.param_groups:
        print(param_group['lr'])

    for param_group in optimizer.param_groups:
        param_group['lr'] = best_lr
    
    for param_group in optimizer.param_groups:
        print(param_group['lr'])

    return best_lr


############### ---------- TRAINING ---------------##########################

def train_model(model, train_loader, val_loader, loss_mode, device, num_classes, 
                epochs = 15, simulated_batch_size = 64, patience = 0, weights = None, 
                improvement_criterion = 'loss_macrof1', improvement_tolerance = 0.05,
                save_to = None, show_batches = 3, save_subfolder = None, working_dir = None, 
                minimum_lr = 1e-7, maximum_lr = 0.1):

### -------------- PREPARING OPTIMIZER, LOSS, METADATA -------------------

    if working_dir is None:
        working_dir = os.path.abspath('..')

    current_patience = 0
    history = []
    start_epoch = 0
    best_model_state = None
    best_optimizer_state = None
    best_epoch = -1
    best_val_loss = float('inf')
    best_val_macrof1 = float(0)
    best_epoch_info = {}
    is_loss_decreasing = False    
    optimizer = optim.AdamW(model.parameters(), lr=minimum_lr, weight_decay=0.05)
    
    if weights is not None:
        if not isinstance(weights, torch.Tensor):
            weights = torch.FloatTensor(weights)
        weights = weights.to(device)
        print('Using Weighted loss: ', weights)    
    criterion = CombinedLoss(loss_mode = loss_mode, weights = weights, return_all_losses=True)
    
    if not save_subfolder:
        save_to = os.path.join(working_dir, 'models', save_to)
    else:
        folder = os.path.join(working_dir, 'models', save_subfolder)
        os.makedirs(folder, exist_ok=True)
        save_to = os.path.join(folder, save_to)
    
    i = save_to.rfind('/')
    j = save_to.rfind('.')
    model_name = save_to[i+1:j] #sem /, até antes do ponto

    if not save_subfolder:
        csv_filename = os.path.join(working_dir, 'experimental_results', model_name+'.csv')
    else:
        folder = os.path.join(working_dir, 'experimental_results', save_subfolder)
        os.makedirs(folder, exist_ok=True)
        csv_filename = os.path.join(folder, model_name+'.csv')

    metadata = {}
    metadata['file_path']=csv_filename
    metadata['model_name']=model_name
    metadata['loss_mode']=loss_mode
    metadata['weighted']=weights != None
    metadata['model_filename']=save_to

    ### -------------- LOADING CHECKPOINT -------------------
    if save_to and os.path.exists(save_to):
        print('Model is already saved')
        checkpoint = torch.load(save_to, weights_only=False)
        
        start_epoch = checkpoint['epoch'] + 1
        best_epoch = checkpoint['best_epoch']
        best_model_state = checkpoint['best_model_state_dict']
        best_optimizer_state = checkpoint['best_optimizer_state_dict']
        model.load_state_dict(best_model_state)
        #if continue_from == 'last':
        #    model.load_state_dict(best_model_state)
        optimizer.load_state_dict(best_optimizer_state) 
        best_val_loss = checkpoint['best_val_loss']
        if 'best_val_macrof1' in checkpoint:
            best_val_macrof1 = checkpoint['best_val_macrof1']
        best_epoch_info = checkpoint['best_epoch_info']
        current_lr = checkpoint['current_lr']
        current_patience = checkpoint['current_patience']  
        is_loss_decreasing = checkpoint['is_loss_decreasing']     
        metadata = checkpoint['metadata']
        info = checkpoint['best_epoch_info']
        history = checkpoint['history']
        print(f"Resumed from epoch {start_epoch-1}, best val loss: {best_val_loss:.4f}, best val loss: {best_val_macrof1:.4f}")
        
        if start_epoch >= epochs:
            print(f"Training already completed {epochs} epochs")
            return
    else:
        # Find initial LR if no checkpoint
        current_lr = update_lr(model, train_loader, val_loader, optimizer, criterion, start_lr=minimum_lr, end_lr=maximum_lr, num_iter=100)

    ### -------------- INNER TRAINING LOOP -------------------
    for epoch in range(start_epoch, epochs):

        print('--------------------------------')
        print(f'Training Epoch {epoch+1}/{epochs}')
        ### -------------- TRAINING -------------------
        torch.cuda.reset_peak_memory_stats()
        train_time = time.time()
        train_runner = EpochRunner('train', model, train_loader, criterion, num_classes=num_classes, 
                                   optimizer=optimizer, simulated_batch_size = simulated_batch_size, device = device)        
        train_runner.run(show_pred = show_batches)  # Training!
        train_loss, train_CE, train_dice, train_report, train_acc, train_cm = train_runner.get_metrics()
        #train_loss, train_CE, train_dice, train_report, train_acc, train_cm = run_epoch('train', model, train_loader, criterion, num_classes = num_classes, optimizer = optimizer, simulated_batch_size = simulated_batch_size, device = device, return_report = True, show_pred=show_batches)
        train_time = time.time()-train_time
        peak_train_memory = f"{torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB"
        torch.cuda.empty_cache()
        
        print(f'Train Loss: {train_loss}, {train_CE}, {train_dice}')
        print(f'Train Accuracy: {train_acc}')
        print(f'Train confusion matrix:')
        try:
            view.plot_confusion_matrix(train_cm)
        except:
            #TODO: fix is
            print('Error while plotting the confusuion matrix.')
        print(train_report)
        ### -------------- VALIDATING -------------------

        torch.cuda.reset_peak_memory_stats()
        val_time = time.time()
        #val_loss, val_CE, val_dice, val_report, val_acc, val_cm = run_epoch('validation', model, val_loader, criterion, num_classes = num_classes, optimizer = optimizer, simulated_batch_size = simulated_batch_size, device = device, return_report = True, show_pred=show_batches)
        val_runner = EpochRunner('val', model, val_loader, criterion, num_classes=num_classes, 
                                   optimizer=optimizer, simulated_batch_size = simulated_batch_size, device = device)
        val_runner.run(show_pred = show_batches)  # Training!        
        val_loss, val_CE, val_dice, val_report, val_acc, val_cm = val_runner.get_metrics()
        val_time = time.time()-val_time
        peak_val_memory = f"{torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB"
        torch.cuda.empty_cache()
        
        print(f'Val Loss: {val_loss}, {val_CE}, {val_dice}')
        ### updating is loss decreasing
        if len(history)>1:
            is_loss_decreasing = val_loss < history[-1]['val_loss']
        else:
            is_loss_decreasing = False
        if is_loss_decreasing:
            print('It is decreasing!')
        print(f'Val Accuracy: {val_acc}')
        print(f'Val confusion matrix:')
        try:
            view.plot_confusion_matrix(val_cm)
        except:
            #TODO: fix is
            print('Error while plotting the confusuion matrix.')
        print(val_report)
        
        ### ------------- ORGANIZING INFORMATON ------------
        info = {}   
        info['epoch']=epoch   
        info['lr']=current_lr
        info['patience']=current_patience 
        info['is_loss_decreasing']=is_loss_decreasing  
        
        #loss = float(loss.item() if isinstance(loss, torch.Tensor) else loss) if isinstance(loss, (torch.Tensor, np.floating, np.integer)) else float(loss)

        info['train_loss']=train_loss
        info['train_acc']=train_acc
        info['train_micro']=train_report.at['micro avg','f1-score']
        info['train_macro']=train_report.at['macro avg','f1-score']
        info['train_weighted']=train_report.at['weighted avg','f1-score']
        for i in range(num_classes):
            info[f'train_f1_C{i}']=train_report.at[f'Class {i}','f1-score']
        info['train_time']=train_time
        info['train_memory']=peak_train_memory
        info['train_report']=train_report
        info['train_CE']=train_CE
        info['train_dice']=train_dice
        info['train_confusion_matrix']=train_cm

        info['val_loss']=val_loss
        info['val_acc']=val_acc
        info['val_micro']=val_report.at['micro avg','f1-score']
        info['val_macro']=val_report.at['macro avg','f1-score']
        info['val_weighted']=val_report.at['weighted avg','f1-score']
        for i in range(num_classes):
            info[f'val_f1_C{i}']=val_report.at[f'Class {i}','f1-score']
        info['val_time']=val_time
        info['val_memory']=peak_val_memory
        info['val_report']=val_report
        info['val_CE']=val_CE
        info['val_dice']=val_dice
        info['val_confusion_matrix']=val_cm
        history.append(info)
        ### ---------------- SAVING IF BEST --------------

        
        improved = False
        if 'loss' in improvement_criterion:
            if val_loss <= best_val_loss * (1 + improvement_tolerance):
                improved = True
        if 'macrof1' in improvement_criterion:
            if info['val_macro'] >= best_val_macrof1 * (1 - improvement_tolerance/10):
                improved = True
        if improved: #if is decreasing(history[-1]['val_loss']-history[-1]['val_loss'])
            best_epoch = epoch
            best_val_loss = val_loss
            best_val_macrof1 = info['val_macro']
            current_patience = 0
            best_model_state = copy.deepcopy(model.state_dict())
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
            best_epoch_info = info
        else:
            current_patience+=1

        
        cols_to_save = ['epoch', 'lr', 'patience', 'train_loss', 'val_loss', 'train_acc', 'val_acc', 
                        'train_micro', 'val_micro', 'train_macro', 'val_macro', 'train_weighted', 'val_weighted']
        cols_to_save+= [f'train_f1_C{i}' for i in range(num_classes)]
        cols_to_save+= [f'val_f1_C{i}' for i in range(num_classes)]
        cols_to_save+= ['train_time', 'val_time', 'train_memory', 'val_memory', 'train_CE', 'val_CE', 'train_dice', 'val_dice']

        rows = []
        for info in history:
            # Combine base_dict with the subset of keys from the current dict in the list
            combined_row = {**metadata, **{key: info[key] for key in cols_to_save}}
            rows.append(combined_row)

        # Create DataFrame
        df = pd.DataFrame(rows)
        df.to_csv(csv_filename, index=False)

        ### ---------------- SETTING NEW LR IF PATIENCE MET --------------

        if current_patience >= patience:
            if not is_loss_decreasing:
                print(f"Patience exhausted at epoch {epoch}.")# Recalculating learning rate...")
                model.load_state_dict(best_model_state)
                optimizer.load_state_dict(best_optimizer_state)
                start_lr = max(minimum_lr, current_lr/100)
                end_lr = min(maximum_lr, current_lr)
                current_lr = update_lr(model, train_loader, val_loader, optimizer, criterion, 
                                    start_lr=start_lr, end_lr=end_lr, num_iter=50)

                current_patience = 0
                is_loss_decreasing = False
        ### ---------------- SAVING CHECKPOINT --------------        
        if save_to is not None:
            checkpoint = {
                "epoch":epoch,
                "best_epoch": best_epoch,
                "best_model_state_dict": best_model_state,
                "best_optimizer_state_dict": best_optimizer_state, 
                "best_val_loss": best_val_loss,
                "best_val_macrof1": best_val_macrof1,
                "best_epoch_info": best_epoch_info,
                "current_lr": current_lr,
                "current_patience": current_patience,
                "is_loss_decreasing": is_loss_decreasing,
                "metadata": metadata,
                "history": history, #info of all epochs
                }
            torch.save(checkpoint, save_to)
    ### ---------------- PLOTTING LOSSES CURVES --------------

    epochs = [info['epoch'] for info in history]           
    train_losses = [info['train_loss'] for info in history]           
    val_losses = [info['val_loss'] for info in history]           
    train_accs = [info['train_acc'] for info in history]           
    val_accs = [info['val_acc'] for info in history]           
    
    if save_subfolder:
        folder = os.path.join(working_dir, 'experimental_results', save_subfolder)
    else:
        folder = os.path.join(working_dir, 'experimental_results')
        
    png_filename = os.path.join(folder, 'loss_'+model_name+'.png')
    view.plot_metrics(history, 'loss', save_file=png_filename)
    #plot_losses(train_losses, val_losses, filename = png_filename)
    png_filename = os.path.join(folder, 'accuracy_'+model_name+'.png')
    view.plot_metrics(history, 'acc', save_file=png_filename)
    #plot_acc(train_accs, val_accs, filename = png_filename)
    png_filename = os.path.join(folder, 'macroF1_'+model_name+'.png')
    view.plot_metrics(history, 'macro', save_file=png_filename)
    


def plot_losses(train_losses, val_losses, title="Training and Validation Loss", xlabel="Epoch", ylabel="Loss", filename=None):
    """
    Plots the training and validation losses on the same plot, returns the figure, and optionally saves it to a file.

    Args:
        train_losses (list): List of training losses for each epoch.
        val_losses (list): List of validation losses for each epoch.
        title (str): Title of the plot (default: "Training and Validation Loss").
        xlabel (str): Label for the x-axis (default: "Epoch").
        ylabel (str): Label for the y-axis (default: "Loss").
        filename (str, optional): If provided, saves the plot to this file (e.g., "loss_plot.png").

    Returns:
        plt.Figure: The matplotlib figure object.
    """
    # Check if the lengths of the lists match
    if len(train_losses) != len(val_losses):
        raise ValueError("train_losses and val_losses must have the same length.")

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    epochs = range(1, len(train_losses) + 1)

    # Plot training and validation losses
    ax.plot(epochs, train_losses, label="Training Loss", marker="o", linestyle="-", color="blue")
    ax.plot(epochs, val_losses, label="Validation Loss", marker="o", linestyle="-", color="red")

    # Add labels, title, and legend
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)

    # Save the plot if a filename is provided
    if filename:
        plt.savefig(filename, bbox_inches="tight")
        print(f"Plot saved to {filename}")

    # Show the plot
    plt.show()

    # Return the figure object
    return fig


def plot_acc(train_acc, val_acc, title="Training and Validation Accuracy", xlabel="Epoch", ylabel="Accuracy", filename=None):
    """
    Plots the training and validation losses on the same plot, returns the figure, and optionally saves it to a file.

    Args:
        train_losses (list): List of training losses for each epoch.
        val_losses (list): List of validation losses for each epoch.
        title (str): Title of the plot (default: "Training and Validation Loss").
        xlabel (str): Label for the x-axis (default: "Epoch").
        ylabel (str): Label for the y-axis (default: "Loss").
        filename (str, optional): If provided, saves the plot to this file (e.g., "loss_plot.png").

    Returns:
        plt.Figure: The matplotlib figure object.
    """
    # Check if the lengths of the lists match
    if len(train_acc) != len(val_acc):
        raise ValueError("train_losses and val_losses must have the same length.")

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    epochs = range(1, len(train_acc) + 1)

    # Plot training and validation losses
    ax.plot(epochs, train_acc, label="Training Loss", marker="o", linestyle="-", color="blue")
    ax.plot(epochs, val_acc, label="Validation Loss", marker="o", linestyle="-", color="red")

    # Add labels, title, and legend
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)

    # Save the plot if a filename is provided
    if filename:
        plt.savefig(filename, bbox_inches="tight")
        print(f"Plot saved to {filename}")

    # Show the plot
    plt.show()

    # Return the figure object
    return fig

def save_results(file_path, model_name, epoch, loss_mode,
                 weighted, learning_rate, batch_size, 
                 train_loss, val_loss, 
                 val_acc, val_micro, val_macro, val_weighted, 
                 val_class_list,
                 status, model_filename,
                 train_time, val_time, peak_train_memory, peak_val_memory):
    file_exists = os.path.isfile(file_path)
    header = ['Model', 'Epoch', 'Loss method', 
              'Weighted loss', 'Learning rate', 'Batch size',
                'Train Loss', 'Val Loss', 
                'Val Accuracy', 'Val Micro F1', 'Val Macro F1', 'Val Weighted F1']
    val_c_headers = [f'Val C{i}' for i in range(len(val_class_list))]
    header.extend(val_c_headers)
    header.extend(['Status', 'Model path', 'Epoch Train Time', 'Epoch Validation Time', 'Train memory (MB)', 'Validation Memory (MB)'])

    values = [model_name, epoch, loss_mode, 
              weighted, learning_rate, batch_size, 
                         train_loss, val_loss, 
                         val_acc, val_micro, val_macro, val_weighted]
    values.extend(val_class_list)
    values.extend([status, model_filename, train_time, val_time, peak_train_memory, peak_val_memory])

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(values)

import cv2
import numpy as np

import numpy as np
import cv2

def apply_opening_multiclass_opencv(image, kernel=None):
    """
    Apply morphological opening to a multiclass image using OpenCV.

    Args:
        image (np.ndarray): Input 2D image of class indices.
        kernel (np.ndarray): Structuring element for the morphological operation.
                             Default is a 3x3 square element.
    Returns:
        np.ndarray: Image after applying opening operation (same shape and format as input).
    """
    # Define the structuring element if not provided
    if kernel is None:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    # Find unique classes
    unique_classes = np.unique(image)
    
    # Create an empty array for the result
    result = np.zeros_like(image, dtype=image.dtype)
    
    # Process each class separately
    for cls in unique_classes:
        # Create a binary mask for the current class
        binary_mask = (image == cls).astype(np.uint8)  # OpenCV requires binary images as uint8
        
        # Apply morphological opening to the binary mask
        opened_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        
        # Assign the opened mask back to the result
        result[opened_mask == 1] = cls  # Preserve the class index in the result

    return result



def test_model(model, checkpoint_path, dataloader, num_classes, device, loss_mode = 'CE', show_batches = True, set_name = '', subfolder = '', show_cm = 'simple'):
    working_dir = os.path.abspath('..')
    checkpoint_path_ = os.path.join(working_dir, 'models', subfolder, checkpoint_path)
    if os.path.exists(checkpoint_path_):
        checkpoint = torch.load(checkpoint_path_, weights_only=False)
        metadata = checkpoint['metadata']
        #print(checkpoint_path)
        #print(checkpoint)
        model.load_state_dict(checkpoint['best_model_state_dict'])
    else:
        raise ValueError(f"Erro na leitura do modelo {checkpoint_path_}.")
    criterion = CombinedLoss(loss_mode = loss_mode, weights = None, return_all_losses=True)

    torch.cuda.reset_peak_memory_stats()
    run_time = time.time()

    runner = EpochRunner('test', model, dataloader, criterion, num_classes=num_classes, 
                                optimizer=None, simulated_batch_size = dataloader.batch_size, device = device)
    runner.run(show_pred = show_batches)  
    loss, CE, dice, report, acc, cm = runner.get_metrics()
    run_time = time.time()-run_time
    peak_val_memory = f"{torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB"
    torch.cuda.empty_cache()
    print('_____________________________________')
    print(checkpoint_path)
    print(f'Loss: {loss}, {CE}, {dice}')
    print(f'Accuracy: {acc}')
    print(f'confusion matrix:')
    


    classes_list = [[2],[4], [5], [2,4,5]]
    #classes_list = [[2,4,5]]
    for cl in classes_list:
        str_cl = [str(c) for c in cl]
        save_ending = f'-CM-'+set_name+'.png'
        os.makedirs(os.path.join(working_dir, 'experimental_results', 'confusion_matrix', subfolder), exist_ok=True)
        save_to = os.path.join(working_dir, 'experimental_results', 'confusion_matrix', subfolder,checkpoint_path.replace('.pth', save_ending))
        
        cm = view.plot_confusion_matrix_simple(cm, save_to=save_to)
        
    return cm, report

def compute_iou_per_sample(pred, label, num_classes):
    """
    Calcula o IoU (Intersection over Union) por classe para cada item do batch e o IoU médio.

    Parâmetros:
    - pred: Tensor (N, C, W, H) contendo as predições (probabilidades), com N = batch size e C = número de classes.
    - label: Tensor (N, W, H) contendo os rótulos verdadeiros (inteiros de 0 a C-1).
    - num_classes: Número total de classes.

    Retorna:
    - iou_per_batch: Lista com um vetor de IoUs por classe para cada item no batch.
    - mean_iou_per_batch: Lista com o IoU médio por amostra no batch.
    """
    batch_size = pred.shape[0]
    iou_per_batch = []
    mean_iou_per_batch = []
    macro_iou_per_batch = []

    for i in range(batch_size):
        iou_per_class = []
        
        # Converte probabilidades em rótulos discretos usando argmax
        pred_i = pred[i].argmax(dim=0)  # (W, H)
        label_i = label[i]  # (W, H)

        total_intersection = 0
        total_union = 0

        for cls in range(num_classes):
            pred_cls = (pred_i == cls)  # Máscara booleana para a classe cls
            label_cls = (label_i == cls)  # Máscara booleana para a classe cls no ground truth

            intersection = (pred_cls & label_cls).sum().float()
            union = (pred_cls | label_cls).sum().float()

            total_intersection += intersection
            total_union += union

            if union == 0:
                iou_per_class.append(float('nan'))  # Ignora classes que não aparecem
            else:
                iou_per_class.append((intersection / union).item())

        # IoU médio para a amostra (não leva em conta as classes com union == 0)
        mean_iou = total_intersection / total_union if total_union != 0 else float('nan')
        mean_iou_per_batch.append(mean_iou)

        valid_iou = [v for v in iou_per_class if not math.isnan(v)]  # Remove 'nan' para cálculo
        if len(valid_iou) > 0:
            macro_iou_per_batch.append(np.mean(valid_iou))        
        else:
            macro_iou_per_batch.append(float('nan'))

        iou_per_batch.append(iou_per_class)

    return iou_per_batch, mean_iou_per_batch, macro_iou_per_batch




def flatten_masks(masks):
    """Flatten masks to shape [N, C], where N is the number of pixels."""
    return masks.view(masks.shape[0], masks.shape[1], -1).transpose(1, 2).reshape(-1, masks.shape[1])

def image_metrics(preds, labels, num_classes, epsilon=1e-7):
    """
    Calculate per-class IoU, Dice, Precision, Recall, and Accuracy for a single image.
    
    Args:
        preds (torch.Tensor): Predicted binary masks of shape [C, H, W].
        labels (torch.Tensor): Ground truth binary masks of shape [C, H, W].
        num_classes (int): Number of classes.
        epsilon (float): Small value to avoid division by zero.
    
    Returns:
        dict: A dictionary containing per-class metrics for the image.
    """
    # Flatten the predictions and labels
    preds_flat = preds.view(preds.shape[0], -1).transpose(0, 1)  # Shape: [N, C]
    labels_flat = labels.view(labels.shape[0], -1).transpose(0, 1)  # Shape: [N, C]
    
    # Initialize metrics dictionary
    metrics = {
        "iou": [],
        "dice": [],
        "precision": [],
        "recall": [],
        "accuracy": []
    }
    
    # Calculate metrics for each class
    for cls in range(num_classes):
        pred_cls = preds_flat[:, cls]  # Predictions for class cls
        label_cls = labels_flat[:, cls]  # Ground truth for class cls
        
        # Calculate TP, FP, FN
        TP = (pred_cls * label_cls).sum().float()  # True Positives
        FP = (pred_cls * (1 - label_cls)).sum().float()  # False Positives
        FN = ((1 - pred_cls) * label_cls).sum().float()  # False Negatives
        TN = ((1 - pred_cls) * (1 - label_cls)).sum().float()  # True Negatives
        
        # Calculate metrics
        iou = (TP + epsilon) / (TP + FP + FN + epsilon)
        dice = (2 * TP + epsilon) / (2 * TP + FP + FN + epsilon)
        precision = (TP + epsilon) / (TP + FP + epsilon)
        recall = (TP + epsilon) / (TP + FN + epsilon)
        accuracy = (TP + TN + epsilon) / (TP + TN + FP + FN + epsilon)
        
        # Append to metrics dictionary
        metrics["iou"].append(iou.item())
        metrics["dice"].append(dice.item())
        metrics["precision"].append(precision.item())
        metrics["recall"].append(recall.item())
        metrics["accuracy"].append(accuracy.item())
    
    return metrics
        



class Metrics:
    def __init__(self, num_classes):
        self.total_loss = 0.0
        self.total_samples = 0
        self.total_CE = 0
        self.total_dice = 0

        self.num_classes=num_classes
        self.all_preds = []
        self.all_labels = []
        self.valid = []

        self.TP = torch.zeros(num_classes)
        self.FP = torch.zeros(num_classes)
        self.FN = torch.zeros(num_classes)
        self.TN = torch.zeros(num_classes)
        self.support = torch.zeros(num_classes)
        self.cm = torch.zeros((num_classes, num_classes), dtype=int)

    def track_preds(self, truths, preds):
        #pred = preds>0.5
        preds = torch.argmax(preds, dim=1)
        for class_idx in range(self.num_classes):
            for i in range(preds.shape[0]):
                pred_ = preds[i]
                truths_ = truths[i]
                self.TP[class_idx] += ((pred_ == class_idx) & (truths_ == class_idx)).sum().item()
                self.FP[class_idx] += ((pred_ == class_idx) & (truths_ != class_idx)).sum().item()
                self.FN[class_idx] += ((pred_ != class_idx) & (truths_ == class_idx)).sum().item()
                self.TN[class_idx] += ((pred_ != class_idx) & (truths_ != class_idx)).sum().item()
                self.support[class_idx] += (truths_ == class_idx).sum().item()  # Number of instances per class
        predictions = preds.view(-1)  # Shape [B * W * H]
        labels = truths.view(-1)  # Shape [B * W * H]

        indices = labels * self.num_classes + predictions  # Convert 2D indices to 1D
        batch_cm = torch.bincount(indices, minlength=self.num_classes**2).reshape(self.num_classes, self.num_classes).cpu()
        self.cm += batch_cm


    def store_metrics(self, loss, CE, dice, batch_size):
        self.total_loss += loss * batch_size  # Multiply by valid pixels in this batch
        self.total_CE += CE * batch_size
        self.total_dice+= dice * batch_size
    def update_batch_count(self, batch_size):
        self.total_samples += batch_size

    def avg_metrics(self):
        avg_loss = self.total_loss / self.total_samples
        avg_CE = self.total_CE / self.total_samples
        avg_dice = self.total_dice / self.total_samples

        return avg_loss, avg_CE, avg_dice
    def class_report(self):
        metrics = {
        'precision': {},
        'recall': {},
        'f1-score': {},
        'support': {},
        'TP':self.TP.tolist(),
        'FP':self.FP.tolist(),
        'TN':self.TN.tolist(),
        'FN':self.FN.tolist(),
        }
    

        precision = self.TP / (self.TP + self.FP + 1e-8)
        recall = self.TP / (self.TP + self.FN + 1e-8)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

        metrics['precision'] = precision.tolist()
        metrics['recall'] = recall.tolist()
        metrics['f1-score'] = f1_score.tolist()
        metrics['support'] = [int(s) for s in self.support.tolist()]

        overall_accuracy = (self.TP.sum() + self.TN.sum()) / (
            self.TP.sum() + self.TN.sum() + self.FP.sum() + self.FN.sum()
        )

        # Weighted accuracy: Weight each class by the number of instances
        weights = self.support / self.support.sum()  # Weights based on the true class distribution
        weighted_precision = (weights * precision).sum()
        weighted_recall = (weights * recall).sum()
        weighted_f1_score = (weights * f1_score).sum()

        macro_precision = precision.mean()
        macro_recall = recall.mean()
        macro_f1_score = f1_score.mean()

        micro_precision = self.TP.sum().item() / (self.TP.sum().item() + self.FP.sum().item())
        micro_recall = self.TP.sum().item() / (self.TP.sum().item() + self.FN.sum().item())
        micro_f1_score = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)


        metrics['precision'].append(weighted_precision.item())
        metrics['recall'].append(weighted_recall.item())
        metrics['f1-score'].append(weighted_f1_score.item())
        metrics['precision'].append(macro_precision.item())
        metrics['recall'].append(macro_recall.item())
        metrics['f1-score'].append(macro_f1_score.item())
        metrics['precision'].append(micro_precision)
        metrics['recall'].append(micro_recall)
        metrics['f1-score'].append(micro_f1_score)
        metrics['support'].append(int(self.support.sum().item())) 
        metrics['support'].append(int(self.support.sum().item())) 
        metrics['support'].append(int(self.support.sum().item()))       
        metrics['TP']+=[0,0,0]
        metrics['TN']+=[0,0,0]
        metrics['FP']+=[0,0,0]
        metrics['FN']+=[0,0,0]       
        
        
        report = pd.DataFrame(metrics, index = [f'Class {i}' for i in range(self.num_classes)] + ['weighted avg', 'macro avg', 'micro avg'])
        confusion_matrix = self.cm
        return report, overall_accuracy, confusion_matrix 
    

