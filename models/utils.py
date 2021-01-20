from typing import Tuple

import torch

class SpanExtractor:
    
    @staticmethod
    def extract_argmax_score(start_scores: torch.Tensor, end_scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given a batch of start and end of span scores (logits), returns the
        highest scoring (start, end) indexes for each sample in the batch.
        
        Args:
            start_scores (torch.Tensor): start position scores of shape (B, T)
                of for each token position.
            end_scores (torch.Tensor): end position scores of shape (B, T)
                of for each token position.
        Returns:
            start_end_indexes (Tuple[torch.Tensor, torch.Tensor]): tuple of 
                tensors containing start and end indexes respectively for each
                sample.
        """
        
        span_start_idxs = torch.argmax(start_scores, axis=-1).cpu().detach()
        span_end_idxs = torch.argmax(end_scores, axis=-1).cpu().detach()
        return (span_start_idxs, span_end_idxs)
    
    @staticmethod
    def extract_most_probable(start_scores: torch.Tensor, end_scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given a batch of start and end of span scores (logits), returns the
        most probable (start, end) indexes for each sample in the batch
        such that end >= start.
        
        Args:
            start_scores (torch.Tensor): start position scores of shape (B, T)
                of for each token position.
            end_scores (torch.Tensor): end position scores of shape (B, T)
                of for each token position.
        Returns:
            start_end_indexes (Tuple[torch.Tensor, torch.Tensor]): tuple of 
                tensors containing start and end indexes respectively for each
                sample.
        """
        
        # extract shapes
        batch_dim, timestep_dim = start_scores.shape
        # compute marginal distributions for start and end
        start_probs = torch.nn.functional.softmax(start_scores, dim=1)
        end_probs = torch.nn.functional.softmax(end_scores, dim=1)
        # compute start_end joint distribution
        joint_dist_start_end = start_probs[:, :, None] @ end_probs[:, None, :]
        constrained_joint_dist = torch.triu(joint_dist_start_end)
        # compute the actual indexes
        flattened_distr_argmax = constrained_joint_dist.view(batch_dim, -1).argmax(1).view(-1, 1)
        start_end_idxs = torch.cat((flattened_distr_argmax // timestep_dim, flattened_distr_argmax % timestep_dim), dim=1).cpu().detach()
        return (start_end_idxs[:, 0], start_end_idxs[:, 1])