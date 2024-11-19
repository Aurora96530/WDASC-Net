from .ade20k import ADE20K
from .camvid import CamVid
from .cityscapes import CityScapes
from .pascalcontext import PASCALContext
from .cocostuff import COCOStuff
from .sunrgbd import SunRGBD
from .mapillary import MapillaryVistas
from .mhpv1 import MHPv1
from .mhpv2 import MHPv2
from .lip import LIP
from .cihp import CIHP, CCIHP
from .atr import ATR
from .suim import SUIM
from .helen import HELEN
from .lapa import LaPa
from .ibugmask import iBugMask
from .celebamaskhq import CelebAMaskHQ
from .facesynthetics import FaceSynthetics
# #################################
from .vaihingen import Vaihingen
from .uavid2020 import UAVid2020
from .isaid import ISAID
from .udd6 import UDD6
from .htht2022 import HTHT2022Coarse
from .aeroscapes import Aeroscapes
from .ddr import DDR
from .idrid import IDRiD
from .fgadr import FGADR
from .lesion import Lesion, Lesion1, Lesion2, Lesion_sigle, Lesion_new, Lesion_dwt1
from .lesion1 import Lesion_multi
from .lesion_dwt import Lesion_dwt
from .domain import Domain
# #################################

__all__ = [
    'CamVid',
    'CityScapes',
    'ADE20K',
    'MHPv1',
    'MHPv2',
    'LIP',
    'CIHP',
    'CCIHP',
    'ATR',
    'PASCALContext',
    'COCOStuff',
    'SUIM',
    'SunRGBD',
    'MapillaryVistas',
    'HELEN',
    'LaPa',
    'iBugMask',
    'CelebAMaskHQ',
    'FaceSynthetics',
    # #################
    'Vaihingen',
    'UAVid2020',
    'ISAID',
    'UDD6',
    'HTHT2022Coarse',
    'Aeroscapes',
    'DDR',
    'IDRiD',
    'FGADR',
    'Lesion1',
    'Lesion2',
    'Lesion',
    'Lesion_new',
    'Lesion_sigle',
    'Lesion_dwt',
    'Lesion_dwt1',
    'Domain'
    # #################
]
