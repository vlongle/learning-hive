import torch


class OODSeparationLoss(torch.nn.Module):
    def __init__(self, delta=1.0, lambda_ood=1.0):
        """
        OOD Separation Loss to push away OOD data from task-specific data.
        :param delta: Margin threshold for the OOD separation.
        :param lambda_ood: Weighting factor for the OOD loss.
        """
        super().__init__()
        self.delta = delta
        self.lambda_ood = lambda_ood

    def forward(self, task_embeddings, ood_embeddings):
        """
        Compute the OOD separation loss.
        :param task_embeddings: Embeddings of the current task data.
        :param ood_embeddings: Embeddings of the OOD data.
        :return: OOD separation loss.
        """
        # Compute pairwise distance matrix between task and OOD embeddings
        dist_matrix = torch.cdist(task_embeddings, ood_embeddings, p=2)

        # Apply margin threshold
        margin_violations = torch.relu(self.delta - dist_matrix)

        # Compute mean of the violations
        ood_loss = margin_violations.mean()

        return self.lambda_ood * ood_loss
