### Source: https://hanlab.mit.edu/courses/2024-fall-65940

import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

def train(
  model: nn.Module,
  dataloader: DataLoader,
  criterion: nn.Module,
  optimizer: Optimizer,
  scheduler: LambdaLR,
  callbacks = None
) -> None:
  model.train()

  for inputs, targets in tqdm(dataloader, desc='train', leave=False):
    # Move the data from CPU to GPU
    inputs = inputs.cuda()
    targets = targets.cuda()

    # Reset the gradients (from the last iteration)
    optimizer.zero_grad()

    # Forward inference
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Backward propagation
    loss.backward()

    # Update optimizer and LR scheduler
    optimizer.step()
    scheduler.step()

    if callbacks is not None:
        for callback in callbacks:
            callback()


def evaluate(
  model: nn.Module,
  dataloader: DataLoader,
  verbose=True,
) -> float:
  model = model.cuda()
  model.eval()

  num_samples = 0
  num_correct = 0

  with torch.no_grad():
    for inputs, targets in tqdm(dataloader, desc="eval", leave=False,
                              disable=not verbose):
      # Move the data from CPU to GPU
      inputs = inputs.cuda()
      targets = targets.cuda()

      # Inference
      outputs = model(inputs)

      # Convert logits to class indices
      outputs = outputs.argmax(dim=1)

      # Update metrics
      num_samples += targets.size(0)
      num_correct += (outputs == targets).sum()


  return (num_correct / num_samples * 100).item()