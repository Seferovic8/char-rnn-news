#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import sys
import matplotlib.pyplot as plt


# In[6]:


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
class SaveModelCheckpoint:
    def __init__(self, path='model_checkpoint.pt'):
        self.path = path
    def __call__(self,val_loss, best_val_loss, train_loss, it, model, optimizer):
        if (val_loss<best_val_loss):
            torch.save({
                    'epoch': it,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    }, self.path)
            bestt_val_loss=val_loss

            print(f'{bcolors.OKGREEN}{bcolors.BOLD}Model saved at epoch: {it+1}, val_loss improved from: {best_val_loss:.4f} to: {val_loss:.4f}{bcolors.ENDC}')
        return bestt_val_loss


# In[5]:


def plot_loss_curves(train_losses,val_losses,train_losses2=None,val_losses2=None):
  if(train_losses2 is not None):
    train_losses = np.concatenate((train_losses,train_losses2))
    val_losses = np.concatenate((val_losses,val_losses2))
  plt.plot(train_losses,label='Train loss')
  plt.plot(val_losses,label='Val loss')
  plt.legend(); plt.show


# In[ ]:


def progress_bar(current_batch, total_batches,update_rate=10,validation=False):
    def get_progress_bar(current_batch,total_batches,length=50):
        completed_circles= current_batch//(total_batches//length)
        return '='*completed_circles + '.'*(length-completed_circles)
    
    if current_batch%update_rate==0:
        if validation:
            sys.stdout.write(f'\rValidation: Batch {current_batch}/{total_batches} - [{get_progress_bar(current_batch,total_batches)}]')
            sys.stdout.flush()
        else:
            sys.stdout.write(f'\rBatch {current_batch}/{total_batches} - [{get_progress_bar(current_batch,total_batches)}]')
            sys.stdout.flush() 
    return current_batch+1


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




