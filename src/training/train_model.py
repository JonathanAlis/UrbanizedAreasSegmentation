import itertools
import os
import torch
from tqdm import tqdm
import pandas as pd
import csv
import torch.nn.functional as F


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



def masked_cross_entropy_loss(pred, target, mask, weights = None):
    target = target.long()
    #print(weights)
    loss = F.cross_entropy(pred, target, reduction='none', weight=weights)  # Shape: (batch_size, height, width)
    loss = loss * mask.float()  # Only keep valid pixels where mask == True
    
    return loss.sum() / mask.sum()
    


def masked_dice_loss(pred, target, mask, num_classes, class_weights = None, epsilon=1e-6):
    #predit are the logits, not the probabilities
    target = target.long()
    #unique_values, counts = torch.unique(target, return_counts=True)
    #print(unique_values, counts)
    target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    pred_probs = torch.sigmoid(pred)  # pred are logits, then get the probabilities
    pred_flat = pred_probs.view(pred.shape[0], pred.shape[1], -1)
    target_flat = target_one_hot.view(target.shape[0], target_one_hot.shape[1], -1)
    
    # Flatten the mask and expand to match class dimension
    mask_flat = mask.view(mask.shape[0], 1, -1)  # Shape: (batch_size, 1, height*width)
    mask_flat = mask_flat.expand_as(pred_flat)    # Shape: (batch_size, num_classes, height*width)
    
    # Apply mask to both pred and target (only keep valid pixels)
    pred_flat_masked = pred_flat * mask_flat
    target_flat_masked = target_flat * mask_flat
    
    # Compute intersection and union
    dice_loss = 0
    for i in range(num_classes):

        
        input_i = pred_flat_masked[:, i]
        target_i = target_flat_masked[:, i]
        
        intersection = (input_i * target_i).sum(dim=-1)        
        total = input_i.sum(dim=-1) + target_i.sum(dim=-1)
        # Dice coefficient for class i
        dice_score = (2 * intersection + epsilon) / (total + epsilon)
        # Apply the class weight (if provided)
        if class_weights is not None:
            weight = class_weights[i]
            dice_loss += weight * (1 - dice_score)
        else:
            dice_loss += 1 - dice_score
    dice_loss = dice_loss / num_classes
    dice_loss = dice_loss.mean(dim=-1) #average by batches

    return dice_loss  



def combined_loss(preds, targets, mask, num_classes, mode = '', label_counts = None):
    # Compute Dice Loss
    
    if mode.lower().startswith('bce'):
        alpha, beta = 0, 1
    elif mode.lower().startswith('dice'):
        alpha, beta = 1, 0
    elif 'BCE'.lower() in mode.lower() and 'dice' in mode.lower():
        alpha, beta = 0.5, 0.5
    else:
        print('Undefined loss, choosing BCE.')
        alpha, beta = 0, 1        
    #print(label_counts)
    if label_counts is not None:
        device = preds.device
        class_weights = 1.0 / torch.tensor(list(label_counts.values())).float()
        class_weights = class_weights / class_weights.sum()
        class_weights = class_weights.to(device)
    else: 
        class_weights = None
    dice_loss = masked_dice_loss(preds, targets, mask, num_classes, class_weights)  # Compute Dice loss per class
    mean_dice_loss = dice_loss.mean()

    cross_entropy_loss = masked_cross_entropy_loss(preds, targets, mask, class_weights)
    

    # Combine the losses
    loss = alpha * mean_dice_loss + beta * cross_entropy_loss
    return loss, cross_entropy_loss, mean_dice_loss



def run_epoch(mode, model, dataloader, loss_mode='BCE', num_classes = None, optimizer=None, device=torch.device('cpu'), show_batches = 0, return_report = False, label_counts = None):
    # Alpha [0-1] is the weight of losses: 0 for cross entropy, 1 for dice
    print(f'Running an epoch on the {mode} mode.')
    if mode == 'train':
        model.train()
    else:
        model.eval()

    metrics_tracker = Metrics(num_classes=num_classes)

    # Disable gradient tracking for eval or test mode
    with torch.set_grad_enabled(mode == 'train'):
        for images, labels in tqdm(dataloader):
            #load and prepare the data
            #images, labels, nan_mask = prepare_batch(images, labels, device) #TODO: NaN           
            images=images.to(device)
            labels=labels.to(device)
            # Forward pass
            print(type(images))
            print(images.shape)
            print(images.dtype)  # Should be torch.float32 or torch.float64
            print(next(model.parameters()).dtype)  # Should match input_tensor.dtype
            logits = model(images)
            pred = torch.sigmoid(logits)
            metrics_tracker.track_preds(labels, pred)
            if show_batches>0:
                try:
                    if all(labels[i].sum()>0 for i in range(labels.shape[0])):
                        for i in range(labels.shape[0]): 
                            if labels[i].sum()>0:
                                view.plot_batch(images, labels, pred)
                                show_batches-=1
                                break
                except Exception as error:
                    print('Error while showing sample images...')
                    print(error)
            if mode == 'train' and label_counts is not None:
                loss, CE, dice = combined_loss(logits, labels, nan_mask, num_classes, mode = loss_mode, label_counts = label_counts)
            else:
                loss, CE, dice = combined_loss(logits, labels, nan_mask, num_classes, mode = loss_mode)
            
            if mode == 'train':
                # Backward pass and optimize, if training
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            metrics_tracker.store_metrics(loss.item(), CE.item(), dice.item(), batch_size=images.shape[0])
            
    avg_loss, avg_CE, avg_dice = metrics_tracker.avg_metrics()
    if return_report:
        report, acc = metrics_tracker.class_report()
        return avg_loss, avg_CE, avg_dice, report, acc

    return avg_loss, avg_CE, avg_dice



# Training loop
def train_model(model, train_loader, val_loader, epochs, loss_mode, optimizer, device, num_classes, patience = 0, label_counts = None, save_to = None, show_batches = 3):

    working_dir = os.path.abspath('..')
    save_to = os.path.join(working_dir, 'models', save_to)

    val_losses = []
    train_losses = []

    #continue from where it stoped:
    starting_epoch = 0
    if os.path.exists(save_to):
        print('Model is already saved')        
        checkpoint = torch.load(save_to, weights_only=False)
        metadata = checkpoint['metadata']
        best_val_loss = metadata['val_metrics']['loss']
        starting_epoch = metadata['epoch']
        
        if 'train_losses_history' not in metadata.keys() and 'train_losses_history' not in metadata.keys() :
            return

        train_losses = metadata['train_losses_history']
        val_losses = metadata['val_losses_history']
        
        if starting_epoch == epochs-1 or val_losses.index(min(val_losses)) <= starting_epoch - patience +1:
            return
        
        print('Loading it e continue training...')
        patience_count = 0
        model.load_state_dict(checkpoint['model_state_dict'])
    else:    
        best_val_loss = float('inf')    
        patience_count = patience+1
    
    
    if label_counts is not None:
        loss_mode+='w'
    print('LOSS mode:', loss_mode)

    print(f'Training for epoch {starting_epoch+1} until epoch {epochs}')
    
    
    for epoch in range(starting_epoch, epochs):
        print(f'Training Epoch {epoch+1}/{epochs}')
        train_loss, train_CE, train_dice, train_report, train_acc = run_epoch('train', model, train_loader, loss_mode, num_classes = num_classes, optimizer = optimizer, device = device, label_counts = label_counts, return_report = True)

        print(f'Train Loss: {train_loss}')
        print(f'Train Accuracy: {train_acc}')
        print(train_report)
        train_losses.append(train_loss)

        val_loss, val_CE, val_dice, val_report, val_acc = run_epoch('validation', model, val_loader, loss_mode, num_classes = num_classes, optimizer = optimizer, device = device, show_batches=show_batches, label_counts = None, return_report = True)
        print(f'Train Loss: {val_loss}')
        print(f'Train Accuracy: {val_acc}')
        print(val_report)
        val_losses.append(val_loss)
        
        status = ''
        # Save the model if validation loss decreases
        
        if epoch == epochs-1:
           status += ' last epoch. ending training.' 
        if val_loss >= best_val_loss:
            patience_count+=1
            if patience_count == patience:
                status += ' patience met. ending training.'
                break
        if val_loss < best_val_loss:
            status += ' current best val loss.'
            patience_count = 0
            best_val_loss = val_loss
            if save_to is not None:
                train_metrics = {
                    "report": train_report,
                    "loss": train_loss,
                    "accuracy": train_acc,
                    "CE": train_CE,
                    "dice": train_dice                   
                    }
                val_metrics = {
                    "report": val_report,
                    "loss": val_loss,
                    "accuracy": val_acc,
                    "CE": val_CE,
                    "dice": val_dice                   
                    }
                metadata = {
                    "filename": save_to,
                    "epoch": epoch,
                    "num_classes": num_classes,
                    "loss_mode" : loss_mode,
                    "train_metrics" : train_metrics,
                    "val_metrics" : val_metrics, 
                    "train_losses_history" : train_losses,
                    "val_losses_history" : val_losses,
                    
                    }
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "metadata": metadata
                    }
                torch.save(checkpoint, save_to)
        
        

        model_name = save_to.split('_')[0]
        i = model_name.rfind('/')
        j = model_name.find('_')
        model_name = model_name[i+1:j]
        
        val_micro = val_report.at['micro avg','f1-score']
        val_macro = val_report.at['macro avg','f1-score']
        val_weighted = val_report.at['weighted avg','f1-score']

        val_f1 = []
        for i in range(num_classes):
            val_f1.append(val_report.at[f'Class {i}','f1-score'])
            
        val_micro = val_report.at['micro avg','f1-score']
        
        validation_per_class_list = [f"{100*val_f1[0]:.2f}" for i in range(num_classes)]
        save_results(file_path=f"results_{model_name}.csv", 
                     model_name = model_name, 
                     epoch=epoch, 
                     loss_mode = loss_mode,
                     train_loss = f"{train_loss:.3f}", 
                     val_loss = f"{val_loss:.3f}",
                     #train_acc = f"{100*train_acc:.2f}", 
                     val_acc = f"{100*val_acc:.2f}", 
                     val_micro = f"{100*val_micro:.2f}",
                     val_macro = f"{100*val_macro:.2f}",
                     val_weighted = f"{100*val_weighted:.2f}",
                     val_class_list = validation_per_class_list,
                     #train_CE = f"{train_CE:.5f}", 
                     #val_CE = f"{val_CE:.5f}",
                     #train_dice = f"{train_dice:.5f}",
                     #val_dice = f"{val_dice:.5f}",
                     status = status,
                     model_filename = save_to
                     )
        


def save_results(file_path, model_name, epoch, loss_mode, 
                 train_loss, val_loss, 
                 val_acc, val_micro, val_macro, val_weighted, 
                 val_class_list,
                 status, model_filename):
    file_exists = os.path.isfile(file_path)
    header = ['Model', 'Epoch', 'Loss method', 
                'Train Loss', 'Val Loss', 
                'Val Accuracy', 'Val Micro F1', 'Val Macro F1', 'Val Weighted F1']
    val_c_headers = [f'Val C{i}' for i in range(len(val_class_list))]
    header.extend(val_c_headers)
    header.extend(['Status', 'Model path'])

    values = [model_name, epoch, loss_mode, 
                         train_loss, val_loss, 
                         val_acc, val_micro, val_macro, val_weighted]
    values.extend(val_class_list)
    values.extend([status, model_filename])

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(values)



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
        
    def track_preds(self, truths, preds, nan_mask = None):
        pred = preds>0.5
        preds = torch.argmax(preds, dim=1)
        nan_mask = nan_mask.squeeze()
        for class_idx in range(self.num_classes):
            for i in range(preds.shape[0]):
                #print('NAN MASK SUM:',nan_mask[i].sum(), 256*256-(nan_mask[i].sum()))

                pred_ = preds[i][nan_mask[i]]
                truths_ = truths[i][nan_mask[i]]
                self.TP[class_idx] += ((pred_ == class_idx) & (truths_ == class_idx)).sum().item()
                self.FP[class_idx] += ((pred_ == class_idx) & (truths_ != class_idx)).sum().item()
                self.FN[class_idx] += ((pred_ != class_idx) & (truths_ == class_idx)).sum().item()
                self.TN[class_idx] += ((pred_ != class_idx) & (truths_ != class_idx)).sum().item()
                self.support[class_idx] += (truths_ == class_idx).sum().item()  # Number of instances per class


    def store_metrics(self, loss, CE, dice, batch_size):
        self.total_loss += loss * batch_size  # Multiply by valid pixels in this batch
        self.total_CE += CE * batch_size
        self.total_dice+= dice * batch_size
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
        
        
        report = pd.DataFrame(metrics, index = [f'Class {i}' for i in range(self.num_classes)] + ['weighted avg', 'macro avg', 'micro avg'])
        
        return report, overall_accuracy
    

