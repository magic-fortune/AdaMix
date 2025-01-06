import torch
from torch import nn
import torch.nn.functional as F

def get_max_indices(probabilities):
    # Get the shape of the input tensor
    B, C, H, W = probabilities.shape

    # Find the max value and its index along the channel dimension
    sum_values = torch.sum(probabilities, dim=1)

    # Find the max value and its index along the spatial dimensions
    max_values, spatial_indices = torch.max(sum_values.view(B, -1), dim=1)

    x_coords = spatial_indices // W
    y_coords = spatial_indices % W

    return x_coords, y_coords


def pert_region(pert_fn, img, x_coords, y_coords, f_x, f_y):
    img_p = img.clone()
    region_height = img.shape[-2] // f_x
    region_width = img.shape[-1] // f_y
    
    mask = torch.zeros(size=(img.shape[0], 1, img.shape[-2], img.shape[-1]))
    
    for b, (xx, yy) in enumerate(zip(x_coords, y_coords)):
        x_start = region_height * xx
        x_end = (xx + 1) * region_height
        y_start = region_width * yy
        y_end = (yy + 1) * region_width
        
        img_region = img[b, :, x_start:x_end, y_start:y_end]
        perturbed_region = pert_fn(img_region.unsqueeze(0)).squeeze(0)
        
        img_p[b, :, x_start:x_end, y_start:y_end] = perturbed_region
        mask[b, :, x_start:x_end, y_start:y_end] = 1
    
    return img_p, mask

def get_region(img, x_coords, y_coords, f_x, f_y):
    
    # return img
    
    region_height = img.shape[-2] // f_x
    region_width = img.shape[-1] // f_y
    img_region = torch.zeros(img.shape[0], img.shape[1], region_height, region_width)
    
    for b, (xx, yy) in enumerate(zip(x_coords, y_coords)):
        x_start = region_height * xx
        x_end = (xx + 1) * region_height
        y_start = region_width * yy
        y_end = (yy + 1) * region_width
        
        img_region[b, :, :, :] = img[b, :, x_start:x_end, y_start:y_end]
        
    return img_region


def exange_area(img_a, uimg_a, lab_a, plab_a, x_coords_a, y_coords_a, x_coords_ua, y_coords_ua, f_x, f_y):
    # feature -> image
    lab_a = lab_a.unsqueeze(1)
    plab_a = plab_a.unsqueeze(1)
    assert img_a.shape == uimg_a.shape
    img_a_ori = img_a.clone()
    uimg_a_ori = uimg_a.clone()
    lab_a_ori = lab_a.clone()
    plab_a_ori = plab_a.clone()
    region_height = img_a.shape[-2] // f_x
    region_width = img_a.shape[-1] // f_y
    
    mask_a = torch.ones(size=(img_a.shape[0], 1, img_a.shape[-2], img_a.shape[-1]), device=img_a.device, requires_grad=False)
    mask_ua = torch.ones(size=(img_a.shape[0], 1, img_a.shape[-2], img_a.shape[-1]), device=img_a.device, requires_grad=False)
    
    for b, (xx_l, yy_l, xx_u, yy_u) in enumerate(zip(x_coords_a, y_coords_a, x_coords_ua, y_coords_ua)):
        x_start_l = region_height * xx_l
        x_end_l = (xx_l + 1) * region_height
        y_start_l = region_width * yy_l
        y_end_l = (yy_l + 1) * region_width
        
        x_start_u = region_height * xx_u
        x_end_u = (xx_u + 1) * region_height
        y_start_u = region_width * yy_u
        y_end_u = (yy_u + 1) * region_width
        
        # exchange the area
        img_a_ori[b, :, x_start_l:x_end_l, y_start_l:y_end_l] = uimg_a[b, :, x_start_u:x_end_u, y_start_u:y_end_u]
        uimg_a_ori[b, :, x_start_u:x_end_u, y_start_u:y_end_u] = img_a[b, :, x_start_l:x_end_l, y_start_l:y_end_l]
        lab_a_ori[b, :, x_start_l:x_end_l, y_start_l:y_end_l] = plab_a[b, :, x_start_u:x_end_u, y_start_u:y_end_u] 
        plab_a_ori[b, :, x_start_u:x_end_u, y_start_u:y_end_u] = lab_a[b, :, x_start_l:x_end_l, y_start_l:y_end_l]
        
        # mask_a -> contains the area of img_a
        mask_a[b, :, x_start_l:x_end_l, y_start_l:y_end_l] = 0
        # mask_ua -> contains the area of uimg_a
        mask_ua[b, :, x_start_u:x_end_u, y_start_u:y_end_u] = 0

    
    return img_a_ori, uimg_a_ori, lab_a_ori, plab_a_ori, mask_a, mask_ua                   
    
if '__main__' == __name__:
    import torch
    image = torch.rand(size=(4, 1, 256, 256))
    image2 = torch.rand(size=(4, 1, 256, 256))
    img_region_l, img_region_u, mask_a, mask_ua = exange_area(image, image2, torch.tensor([14,  1,  8, 14,  7,  7]), torch.tensor([9, 4, 7, 2, 8, 8]), torch.tensor([7, 3, 8, 5, 1, 9]), torch.tensor([ 8, 14, 14,  1,  9, 14]),4, 4)
    # torch.Size([4, 1, 256, 256]) torch.Size([4, 1, 256, 256]) torch.Size([4, 1, 256, 256]) torch.Size([4, 1, 256, 256])
    print(img_region_l.shape, img_region_u.shape, mask_a.shape, mask_ua.shape)
    