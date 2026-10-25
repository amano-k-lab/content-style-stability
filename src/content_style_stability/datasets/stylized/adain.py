from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from .function import adaptive_instance_normalization
from .net import decoder, vgg


def crop_square_and_downsample(img, downsize_size=(224, 224)):
    """Crop to largest center square and then downsample."""

    width, height = img.size
    new_size = int(np.min((width, height)))
    left = int((width - new_size) / 2)
    top = int((height - new_size) / 2)
    right = left + new_size
    bottom = top + new_size
    img = img.crop((left, top, right, bottom))
    img.thumbnail(downsize_size, Image.LANCZOS)
    return img


def get_default_transforms(size, crop):
    """Apply a series of transformations."""

    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


class StyleLoader:
    """Generic (transfer-agnostic) loading of styles and converting them to tensors."""

    def __init__(
        self, style_transferer, style_img_file_list, rng, do_preprocessing=True
    ):
        assert isinstance(
            style_transferer, StyleTransferer
        ), "needs to be subclass of StyleTransferer"
        self.num_style_imgs = len(style_img_file_list)
        assert self.num_style_imgs >= 1, "please specify style image paths."
        self.style_transferer = style_transferer
        self.style_img_file_list = style_img_file_list
        self.rng = rng
        self.do_preprocessing = do_preprocessing

    def get_style_tensor_function(self, content_tensor):
        """Transfer randomly selected style onto content_tensor."""

        style_tensor = self.get_style_tensor(tensor_size=content_tensor.size())

        return self.style_transferer.transfer_tensor_to_tensor(
            style_tensor=style_tensor, content_tensor=content_tensor
        )

    def get_style_tensor(self, tensor_size):
        """Return a batch of style images in a tensor (randomly selected)."""
        minibatch_size = tensor_size[0]
        indices = self.rng.randint(low=0, high=self.num_style_imgs, size=minibatch_size)
        return self.get_style_tensor_from_indices(indices.tolist(), tensor_size[1:])

    def get_style_tensor_from_indices(self, indices, spatial_size=(3, 224, 224)):
        """Return a batch of style images for given indices.

        Args:
            indices: List of style image indices into style_img_file_list.
            spatial_size: (C, H, W) for output tensor.

        Returns:
            Tensor of shape (len(indices), C, H, W).
        """
        output_tensor = torch.FloatTensor(len(indices), *spatial_size).zero_()
        to_tensor = transforms.Compose([transforms.ToTensor()])

        for i, idx in enumerate(indices):
            style_image_path = self.style_img_file_list[idx]
            style_image = Image.open(style_image_path)
            if self.do_preprocessing:
                style_image = style_image.convert("RGB")
                style_image = crop_square_and_downsample(style_image, (224, 224))
            output_tensor[i, :, :, :] = to_tensor(style_image)

        device = getattr(self.style_transferer, "device", None)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return output_tensor.to(device)


class StyleTransferer(ABC):
    """Abstract class of a style transfer algorithm.

    Serves the purpose that the ImageNet training methods can
    use the abstract classes here so that even when the transfer algorithm
    is exchanged, one does not need to adjust anything.
    """

    def raise_default_error(self):
        raise NotImplementedError("This needs to be implemented by child class.")

    @abstractmethod
    def transfer_single_style(self, style_variable, content_variable):
        self.raise_default_error()

    @abstractmethod
    def transfer_tensor_to_tensor(self, style_tensor, content_tensor):
        self.raise_default_error()


class AdaIN(StyleTransferer):
    """AdaIN style transfer method from [Huang+, ICCV2017]."""

    def __init__(self, args):
        self.args = args

        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.decoder = decoder
        self.vgg = vgg

        self.decoder.eval()
        self.vgg.eval()

        self.decoder.load_state_dict(
            torch.load(self.args.decoder, map_location=self.device, weights_only=True)
        )
        self.vgg.load_state_dict(
            torch.load(self.args.vgg, map_location=self.device, weights_only=True)
        )
        self.vgg = nn.Sequential(*list(self.vgg.children())[:31])

        self.vgg.to(self.device)
        self.decoder.to(self.device)

        self.content_transforms = get_default_transforms(
            self.args.content_size, self.args.crop
        )
        self.style_transforms = get_default_transforms(
            self.args.style_size, self.args.crop
        )

        self.print_img_counter = 0

    def transfer_tensor_to_tensor(
        self, style_tensor, content_tensor, transform_style=False, output_to_cpu=False
    ):
        """Given style and content tensors (batches), transfer!

        Keyword arguments:
        style_tensor -- a torch.FloatTensor with size [B, C, W, H]
        content_tensor -- a torch.FloatTensor with size [B, C, W, H]
                          (where B = batch_size, and typically:
                           C = 3 (colour channels), W and H = 224)
        transform_style -- boolean indicating whether the style tensor should
                           be transformed (e.g., cropped)
        output_to_cpu -- a boolean indicating whether the return value should be
                         set to .cpu()

        Returns:
        output_tensor -- a torch.FloatTensor with size [B, C, W, H]
        """

        tensor_dimensionality = 4
        assert len(content_tensor.size()) == tensor_dimensionality, "size mismatch."
        assert len(style_tensor.size()) == tensor_dimensionality, "size mismatch."
        for i in range(tensor_dimensionality):
            assert content_tensor.size()[i] == style_tensor.size()[i], "size mismatch."

        batch_size = content_tensor.size()[0]

        style_tensor = style_tensor.to(self.device)
        content_tensor = content_tensor.to(self.device)
        output_tensor = torch.FloatTensor(content_tensor.size()).zero_().to(self.device)

        for i in range(batch_size):
            with torch.no_grad():
                # convert style to variable
                style = style_tensor[i, :, :, :]
                if transform_style:
                    style = self.style_transforms(style)
                style = style.unsqueeze(0)

                # convert content to variable
                content = content_tensor[i, :, :, :]
                content = content.unsqueeze(0)

            # call new_transfer_single_style
            output_tensor[i, :, :, :] = self.transfer_single_style(
                style_variable=style,
                content_variable=content,
                output_to_cpu=output_to_cpu,
            )

        # return output tensor
        return output_tensor

    def transfer_single_style(
        self, style_variable, content_variable, output_to_cpu=False
    ):
        """Given style and content: transfer!

        Keyword arguments:
        style_variable -- a torch.autograd.variable.Variable with the style image
        content_variable -- a torch.autograd.variable.Variable with the content image
        output_to_cpu -- a boolean indicating whether the return value should be
                         set to .cpu()

        Returns:
        output_tensor -- a torch.FloatTensor with the transferred image
        """

        output_tensor = self.transfer_helper(
            style_variable, content_variable, alpha=self.args.alpha
        ).data

        if output_to_cpu:
            output_tensor = output_tensor.cpu()

        return output_tensor

    def transfer_helper(self, style, content, alpha=1.0, interpolation_weights=None):
        """Helper method for style transfer.

        Keyword arguments:
        style -- a torch.autograd.variable.Variable with the style image
        content -- a torch.autograd.variable.Variable with the content image
        alpha --
        interpolation_weights --

        Returns:
        a torch.FloatTensor with the transferred image
        """

        assert 0.0 <= alpha <= 1.0
        content_f = self.vgg(content)
        style_f = self.vgg(style)
        if interpolation_weights:
            _, C, H, W = content_f.size()
            feat = torch.FloatTensor(1, C, H, W).zero_().to(self.device)
            base_feat = adaptive_instance_normalization(content_f, style_f)
            for i, w in enumerate(interpolation_weights):
                feat = feat + w * base_feat[i : i + 1]
            content_f = content_f[0:1]
        else:
            feat = adaptive_instance_normalization(content_f, style_f)
        feat = feat * alpha + content_f * (1 - alpha)
        return self.decoder(feat)
