from pie.models.decoder import LinearDecoder
from ..utils.labels import CategoryEncoder


def Classifier(output_encoder: CategoryEncoder,
               input_size: int, highway_layers: int = 0,
               highway_act: str = "relu") -> LinearDecoder:
    return LinearDecoder(
        label_encoder=output_encoder,
        in_features=input_size,
        highway_layers=highway_layers,
        highway_act=highway_act
    )
