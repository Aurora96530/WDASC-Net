from .upernet import UPerHead, UPerHead_fa
from .segformer import SegFormerHead, SegFormerHead_fa
from .sfnet import SFHead
from .fpn import FPNHead
from .fapn import FaPNHead
from .fcn import FCNHead
from .condnet import CondHead
from .lawin import LawinHead

__all__ = ['UPerHead', 'UPerHead_fa', 'SegFormerHead', 'SegFormerHead_fa', 'SFHead', 'FPNHead', 'FaPNHead', 'FCNHead', 'CondHead', 'LawinHead']