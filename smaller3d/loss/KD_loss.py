import torch
import torch.nn as nn
import torch.nn.functional as F

def kd_loss(student_outputs, teacher_outputs, target_labels, alpha=0.5, temperature=4.0):
    """
    Calculate the Knowledge Distillation Loss.
    
    Args:
    - student_outputs (torch.Tensor): [B, C, N] shaped tensor, student network outputs
    - teacher_outputs (torch.Tensor): [B, C, N] shaped tensor, teacher network outputs
    - target_labels (torch.Tensor): [B, C] shaped tensor, ground truth labels
    - alpha (float): weight for the KD loss, default 0.5
    - temperature (float): temperature for the softmax function, default 4.0
    
    Returns:
    - loss (torch.Tensor): scalar loss value
    """
    
    # Apply softmax with temperature to teacher and student outputs
    student_logits = F.softmax(student_outputs / temperature, dim=1)
    teacher_logits = F.softmax(teacher_outputs / temperature, dim=1)
    
    # Calculate the KL-divergence between teacher and student logits
    kl_div = nn.KLDivLoss(reduction="batchmean")(torch.log(student_logits), teacher_logits)
    
    # Calculate the Cross Entropy Loss for student outputs and target labels
    ce_loss = F.cross_entropy(student_outputs, target_labels)
    
    # Combine the losses using the alpha parameter
    loss = alpha * kl_div + (1 - alpha) * ce_loss
    
    return loss


class DistillationLoss(nn.Module):     
    def __init__(self, temperature=1):
        super().__init__()        
        self.temperature = temperature
    def forward(self, student_logits, teacher_logits):
        # Compute the soft targets from the teacher model        
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
        # Compute the output probabilities of the student model
        student_probs = F.softmax(student_logits / self.temperature, dim=1)        
        # Compute the cross-entropy loss between the soft targets and the output probabilities        
        loss_ce = nn.CrossEntropyLoss()(student_logits, torch.argmax(soft_targets, dim=1))
        # Compute the distillation loss using the gradients with respect to the logits
        loss_dist = torch.mean(torch.sum((student_logits - teacher_logits) **  2, dim=1)) / (2 * self.temperature ** 2)        
        # Compute the total loss as a linear combination of the cross-entropy loss and the distillation loss       
        loss = (1 - self.alpha) * loss_ce + self.alpha * loss_dist
        return loss