from semseg.StructureAwareBrightnessAugmentation import Structure_Aware_Brightness_Augmentation
from semseg.slaug import LocationScaleAugmentation
from semseg.fourier import FDA_source_to_target_np
import numpy as np

def brightness_augmentation(image, mask):
    BrightnessAug = Structure_Aware_Brightness_Augmentation()
    # SABA = BrightnessAug.Global_Color_Augmentation(image.copy())
    # SABA = BrightnessAug.Global_Saturation_Augmentation(image.copy())
    # SABA = BrightnessAug.Local_Brightness_Augmentation(image.copy(), mask.copy().astype(np.int32))
    SABA = BrightnessAug.Global_Local_Color_Augmentation(image.copy(), mask.copy().astype(np.int32))
    return SABA
    # return SABA, GCA

def sl_augmentation(image, mask):
    location_scale = LocationScaleAugmentation(vrange=(0., 1.), background_threshold=0.01)
    # location_scale = LocationScaleAugmentation(vrange=(0., 255.), background_threshold=0.01)
    GLA = location_scale.Global_Location_Scale_Augmentation(image.copy())
    LLA = location_scale.Local_Location_Scale_Augmentation(image.copy(), mask.copy().astype(np.int32))
    return GLA, LLA

def fourier_augmentation_reverse(data, fda_beta=0.1):
    this_fda_beta = round(0.05+np.random.random() * fda_beta, 2)
    lowf_batch = data[::-1]
    fda_data = FDA_source_to_target_np(data, lowf_batch, L=this_fda_beta)
    return fda_data

def collate_fn_tr_styleaug(batch):
    images, labels = zip(*batch)
    images = np.stack(images, 0)
    labels = np.stack(labels, 0)
    data_dict = {'img': images, 'mask': labels}

    # GLA, LLA = sl_augmentation(data_dict['img'], data_dict['mask'])
    SABA= brightness_augmentation(data_dict['img'], data_dict['mask'])
    # SABA, GCA = brightness_augmentation(data_dict['img'], data_dict['mask'])
    data_dict['SABA'] = SABA
    # data_dict['GCA'] = GCA
    # data_dict['GLA'] = GLA
    # data_dict['LLA'] = LLA
    return data_dict

def collate_fn_ts(batch):
    images, labels = zip(*batch)
    images = np.stack(images, 0)
    labels = np.stack(labels, 0)
    data_dict = {'img': images, 'mask': labels}
    return data_dict