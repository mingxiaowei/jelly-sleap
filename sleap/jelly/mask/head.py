from sleap.nn.heads import *

class ClassMapsHead(Head):
    """Head for specifying class identity maps.

    Attributes:
        classes: List of string names of the classes.
        sigma: Spread of the class maps around each node.
        output_stride: Stride of the output head tensor. The input tensor is expected to
            be at the same stride.
        loss_weight: Weight of the loss term for this head during optimization.
    """

    classes: List[Text]
    sigma: float = 5.0
    output_stride: int = 1
    loss_weight: float = 1.0

    @property
    def channels(self) -> int:
        """Return the number of channels in the tensor output by this head."""
        return len(self.classes)

    @property
    def activation(self) -> str:
        """Return the activation function of the head output layer."""
        return "sigmoid"

    @classmethod
    def from_config(
        cls,
        config: ClassMapsHeadConfig,
        classes: Optional[List[Text]] = None,
    ) -> "ClassMapsHead":
        """Create this head from a set of configurations.

        Attributes:
            config: A `ClassMapsHeadConfig` instance specifying the head parameters.
            classes: List of string names of the classes that this head will predict.
                This must be set if the `classes` attribute of the configuration is not
                set.

        Returns:
            The instantiated head with the specified configuration options.
        """
        if config.classes is not None:
            classes = config.classes
        return cls(
            classes=classes,
            sigma=config.sigma,
            output_stride=config.output_stride,
            loss_weight=config.loss_weight,
        )