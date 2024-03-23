import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class DiceLoss(_Loss):
    def __init__(
        self,
        log_loss: bool = True,
        smooth: float = 0.0,
        eps: float = 1e-7,
    ):
        """Compute Dice loss"""
        super().__init__()
        self.smooth = smooth
        self.eps = eps
        self.log_loss = log_loss

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        assert y_true.size(0) == y_pred.size(0)

        # Apply activations to get [0..1] class probabilities
        # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
        # extreme values 0 and 1
        y_pred = F.logsigmoid(y_pred).exp()

        # dimention 0 are images in a batch
        # dims = (1, 2, 3)

        predict, target = y_pred, y_true.type_as(y_pred)

        smooth = self.smooth
        eps = self.eps

        intersection = torch.nansum(predict * target)
        cardinality = torch.nansum(predict + target)
        dice = (2 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)

        if self.log_loss:
            loss = -torch.log(dice)
        else:
            loss = 1 - dice

        # mask = y_true.sum() > 0
        # loss *= mask.to(loss.dtype)
        # agg_loss = loss.mean()

        return loss


class FocalLoss(_Loss):
    def __init__(
        self,
        gamma=2.0,
        normalized: bool = False,
        eps=1e-6,
    ):
        """Compute Focal loss"""
        super().__init__()

        self.gamma = gamma
        self.normalized = normalized
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        assert y_true.size(0) == y_pred.size(0)

        batch_size = y_pred.size(0)

        predict, target = y_pred, y_true.type_as(y_pred)

        # reduce to two dimensions, dim-0 for the images in the batch
        predict = predict.view(batch_size, -1)
        target = target.view(batch_size, -1)

        logpt = F.binary_cross_entropy_with_logits(predict, target, reduction="none")
        pt = torch.exp(-logpt)

        assert pt.ndim == 2

        focal_term = (1.0 - pt).pow(self.gamma)

        loss = focal_term * logpt

        assert loss.ndim == 2

        if self.normalized:
            norm_factor = focal_term.sum(1).clamp_min(self.eps).view(batch_size, -1)
            loss /= norm_factor

        agg_loss = loss.mean()

        return agg_loss


def hough_transform(batch, threshold=50, return_coordinates=False):
    height, width = batch[0].squeeze().shape

    thetas = torch.arange(0, 180, 0.5)
    d = torch.sqrt(torch.tensor(width) ** 2 + torch.tensor(height) ** 2)
    rhos = torch.arange(-d, d, 3)

    cos_thetas = torch.cos(torch.deg2rad(thetas))
    sin_thetas = torch.sin(torch.deg2rad(thetas))

    hough_matrices = torch.Tensor(
        batch.shape[0], rhos.shape[0] - 1, thetas.shape[0] - 1
    )

    for i, img in enumerate(batch):
        img = img.squeeze()
        points = torch.argwhere(img > 0.5).type_as(cos_thetas)
        rho_values = torch.matmul(points, torch.stack((sin_thetas, cos_thetas)))

        accumulator, (theta_vals, rho_vals) = torch.histogramdd(
            torch.stack(
                (
                    torch.tile(thetas, (rho_values.shape[0],)),
                    rho_values.ravel(),
                )
            ).T,
            bins=[thetas, rhos],
        )

        accumulator = torch.transpose(accumulator, 0, 1)

        # testing, return only the first
        if return_coordinates:
            hough_lines = torch.argwhere(accumulator > threshold)
            rho_idxs, theta_idxs = hough_lines[:, 0], hough_lines[:, 1]
            hough_rhos, hough_thetas = rhos[rho_idxs], thetas[theta_idxs]
            hough_coordinates = torch.stack((hough_rhos, hough_thetas))
            return hough_coordinates

        hough_matrix = torch.where(accumulator > threshold, 1, 0)
        hough_matrices[i] = hough_matrix

    return hough_matrices


class SRLoss(_Loss):
    def __init__(self, loss_base: str = "dice", weight=0.5):
        """Dice loss for image segmentation task with Hough Transform constraint."""
        super(SRLoss, self).__init__()
        self.weight = weight

        assert loss_base in ["dice", "focal"]
        if loss_base == "dice":
            self.loss_fn = DiceLoss(log_loss=True)

        if loss_base == "focal":
            self.loss_fn = FocalLoss(normalized=True)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        assert y_true.size(0) == y_pred.size(0)

        predict, target = y_pred, y_true.type_as(y_pred)
        base_loss = self.loss_fn.forward(predict, target)

        # compute customized hough loss
        hough_predict = hough_transform(predict)
        hough_target = hough_transform(target)

        hough_loss = self.loss_fn.forward(hough_predict, hough_target)

        loss = self.weight * hough_loss + (1 - self.weight) * base_loss

        return loss
