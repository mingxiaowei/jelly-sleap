from sleap.nn.architectures import UNet
from sleap.nn.data.utils import ensure_list
from sleap.nn.architectures.unet import *
from sleap.nn.model import Model
from sleap.nn.heads import *
from sleap.nn.architectures.common import *

@attr.s(auto_attribs=True)
class UNetConfig:
    """UNet backbone configuration.

    Attributes:
        stem_stride: If not None, controls how many stem blocks to use for initial
            downsampling. These are useful for learned downsampling that is able to
            retain spatial information while reducing large input image sizes.
        max_stride: Determines the number of downsampling blocks in the network,
            increasing receptive field size at the cost of network size.
        output_stride: Determines the number of upsampling blocks in the network.
        filters: Base number of filters in the network.
        filters_rate: Factor to scale the number of filters by at each block.
        middle_block: If True, add an intermediate block between the downsampling and
            upsampling branch for additional processing for features at the largest
            receptive field size. This will not introduce an extra pooling step.
        up_interpolate: If True, use bilinear upsampling instead of transposed
            convolutions for upsampling. This can save computations but may lower
            overall accuracy.
        stacks: Number of repeated stacks of the network (excluding the stem).
    """

    stem_stride: Optional[int] = None
    max_stride: int = 16
    output_stride: int = 1
    filters: int = 64
    filters_rate: float = 2
    middle_block: bool = True
    up_interpolate: bool = False
    stacks: int = 1
    
@attr.s(auto_attribs=True)
class MaskModel:
    """SLEAP model that describes an architecture and output types.

    Attributes:
        backbone: An `Architecture` class that provides methods for building a
            tf.keras.Model given an input.
        heads: List of `Head`s that define the outputs of the network.
        keras_model: The current `tf.keras.Model` instance if one has been created.
    """

    backbone: UNet
    heads: MaskHead
    keras_model: Optional[tf.keras.Model] = None
   
    @property
    def maximum_stride(self) -> int:
        """Return the maximum stride of the model backbone."""
        return self.backbone.maximum_stride

    def make_model(self, input_shape: Tuple[int, int, int]) -> tf.keras.Model:
        """Create a trainable model by connecting the backbone with the heads.

        Args:
            input_shape: Tuple of (height, width, channels) specifying the shape of the
                inputs before preprocessing.

        Returns:
            An instantiated `tf.keras.Model`.
        """
        # Create input layer.
        x_in = tf.keras.layers.Input(input_shape, name="input")

        # Create backbone.
        x_main, x_mid = self.backbone.make_backbone(x_in=x_in)

        # Make sure main and intermediate feature outputs are lists.
        if type(x_main) != list:
            x_main = [x_main]
        if len(x_mid) > 0 and isinstance(x_mid[0], IntermediateFeature):
            x_mid = [x_mid]

        # Build output layers for each head.
        x_outs = []
        for output in self.heads:
            x_head = []
            if output.output_stride == self.backbone.output_stride:
                # The main output has the same stride as the head, so build output layer
                # from that tensor.
                for i, x in enumerate(x_main):
                    x_head.append(output.make_head(x))

            else:
                # Look for an intermediate activation that has the correct stride.
                for feats in zip(*x_mid):
                    # TODO: Test for this assumption?
                    assert all([feat.stride == feats[0].stride for feat in feats])
                    if feats[0].stride == output.output_stride:
                        for i, feat in enumerate(feats):
                            x_head.append(output.make_head(feat.tensor))
                        break

            if len(x_head) == 0:
                raise ValueError(
                    f"Could not find a feature activation for output at stride "
                    f"{output.output_stride}."
                )
            x_outs.extend(x_head)
        # TODO: Warn/error if x_main was not connected to any heads?

        # Create model.
        self.keras_model = tf.keras.Model(inputs=x_in, outputs=x_outs)
        return self.keras_model
