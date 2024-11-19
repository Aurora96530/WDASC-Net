# from .segformer import SegFormer
# from .ddrnet import DDRNet
# from .ddrnet_official import DDRNet
from .fchardnet import FCHarDNet
from .sfnet import SFNet
from .bisenetv1 import BiSeNetv1
from .bisenetv2 import BiSeNetv2
from .bisenetv2_fa import BiSeNetv2_fa
from .lawin import Lawin

# added models
from .deeplabv3plus import DeeplabV3Plus, DeeplabV3Plus1
from .deeplabv2 import DeeplabV2
from .pspnet import PSPNet
from .upernet import UperNet
# from .sosnet_ablation import SOSNetBaseline, SOSNetSB, SOSNetDFEMABL
from .fast_scnn import FastSCNN
# from .ccnet import CCNet
from .topformer import TopFormer
from .segformer_fa import SegFormer_fa
from .pidnet import PIDNet
from .deeplabv3plus_fa import DeeplabV3Plus_fa
from .deeplabv3plus_mfa import DeeplabV3Plus_mfa
from .deeplabv3plus_mscg import DeeplabV3Plus_mscg
from .deeplabv3plus_fim import DeeplabV3Plus_fim
from .deeplabv3plus_maf import DeeplabV3Plus_maf
from .deeplabv3plus_fim import DeeplabV3Plus_cgl
from .upernet_fa import UperNet_fa

__all__ = [
    # 'SegFormer',
    'Lawin',
    'SFNet',
    'BiSeNetv1',
    'TopFormer',
    'SegFormer_fa',
    'PSPNet',
    'DeeplabV3Plus',
    'DeeplabV3Plus1',
    'DeeplabV2',
    'UperNet',
    # 'CCNet',
    # Standalone Models
    'FastSCNN',
    # 'DDRNet',
    'FCHarDNet',
    'BiSeNetv2',
    'BiSeNetv2_fa',
    'PIDNet',
    'DeeplabV3Plus_fa',
    'DeeplabV3Plus_mfa',
    'DeeplabV3Plus_mscg',
    'DeeplabV3Plus_fim',
    'DeeplabV3Plus_maf',
    'DeeplabV3Plus_cgl',
    'UperNet_fa'
]
