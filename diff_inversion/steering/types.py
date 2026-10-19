"""Shared annotations for the dictionaries passed between steering stages."""

from typing import Any, Literal, NotRequired, TypeAlias, TypedDict

from torch import Tensor

# YAML settings remain a dynamic mapping; tensor data has explicit structure below.
Config: TypeAlias = dict[str, Any]
Method: TypeAlias = Literal["DDIM", "MW"]
Representation: TypeAlias = Literal["CA", "SA", "block"]
Control: TypeAlias = Literal["true", "swap", "gaussian"]
LayerProfile: TypeAlias = Literal["all", "last36", "last12"]
Conditioning: TypeAlias = dict[str, Tensor]
LayerTensors: TypeAlias = dict[str, Tensor]
FeatureMaps: TypeAlias = dict[Representation, LayerTensors]
Offsets: TypeAlias = dict[int, LayerTensors]


class Source(TypedDict):
    id: str
    path: str
    label: Literal["positive", "negative"]
    split: NotRequired[str]


class ActivationRead(TypedDict):
    features: FeatureMaps  # Each tensor is [step, channel].
    moments: FeatureMaps  # Each tensor is [step], before the RMS square root.


class SourceRecord(Source):
    DDIM: ActivationRead
    MW: ActivationRead


class FeatureBundle(TypedDict):
    config: Config
    feature_source: str
    records: list[SourceRecord]


class DirectionFit(TypedDict):
    directions: dict[Representation, dict[Method, dict[str, dict[Control, Tensor]]]]
    reference_rms: FeatureMaps
    null_seed: int


class DirectionBundle(DirectionFit):
    layer_order: list[str]
    config: Config
    feature_source: str
    sources: list[Source]
