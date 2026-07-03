from typing import Literal, TypeAlias

WandbWatchLog: TypeAlias = Literal["gradients", "parameters", "all"]

WandbMode: TypeAlias = Literal["online", "offline", "disabled"]
