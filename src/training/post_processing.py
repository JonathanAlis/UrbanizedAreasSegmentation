import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from skimage.morphology import opening, closing, disk
from scipy.special import softmax
#from osgeo import gdal

def smooth_logits(logits, method='gaussian', sigma=1):
    """Apply smoothing to logits using Gaussian or TVD."""
    if method == 'gaussian':
        return gaussian_filter(logits, sigma=sigma)
    elif method == 'tvd':
        from skimage.restoration import denoise_tv_chambolle
        return np.stack([denoise_tv_chambolle(logits[i], weight=0.1) for i in range(logits.shape[0])])
    return logits

def apply_dense_crf(image, logits):
    """Applies CRF to refine logits."""
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian

    num_classes, H, W = logits.shape
    d = dcrf.DenseCRF2D(W, H, num_classes)
    U = unary_from_softmax(logits)
    d.setUnaryEnergy(U)
    
    # Pairwise Gaussian filter
    d.addPairwiseEnergy(create_pairwise_gaussian((3, 3), (W, H)), compat=3)
    
    # Pairwise Bilateral filter
    d.addPairwiseEnergy(create_pairwise_bilateral((50, 50), (10, 10, 10), image, chdim=2), compat=5)

    Q = np.array(d.inference(10))
    return Q.reshape((num_classes, H, W))

def morphological_operations(mask, kernel_size=3):
    """Apply opening and closing to remove noise and smooth edges."""
    kernel = disk(kernel_size)
    cleaned = np.zeros_like(mask)
    
    for cls in np.unique(mask):
        binary = (mask == cls).astype(np.uint8)
        binary = opening(binary, kernel)
        binary = closing(binary, kernel)
        cleaned[binary > 0] = cls
    
    return cleaned

def apply_gdal_sieve(segmentation_mask, min_size=100):
    """Remove small isolated regions using GDAL Sieve."""
    driver = gdal.GetDriverByName("MEM")
    src_ds = driver.Create("", segmentation_mask.shape[1], segmentation_mask.shape[0], 1, gdal.GDT_Int32)
    src_ds.GetRasterBand(1).WriteArray(segmentation_mask)

    gdal.SieveFilter(src_ds.GetRasterBand(1), None, src_ds.GetRasterBand(1), threshold=min_size)

    return src_ds.GetRasterBand(1).ReadAsArray()

def full_segmentation_pipeline(image, logits, smooth_method='gaussian', sigma=1, min_size=100, kernel_size=3):
    """
    Full segmentation refinement pipeline:
    1. Smooth logits
    2. Apply CRF
    3. Convert to discrete classes
    4. Apply morphological opening & closing
    5. Remove small isolated regions using GDAL sieve
    6. Final morphological cleanup
    """
    # Step 1: Smooth logits
    logits = smooth_logits(logits, method=smooth_method, sigma=sigma)

    # Step 2: Apply CRF
    logits = apply_dense_crf(image, logits)

    # Step 3: Convert logits to class mask
    segmentation_mask = np.argmax(logits, axis=0)

    # Step 4: Morphological opening & closing
    segmentation_mask = morphological_operations(segmentation_mask, kernel_size=kernel_size)

    # Step 5: Remove small regions using GDAL sieve
    #segmentation_mask = apply_gdal_sieve(segmentation_mask, min_size=min_size)

    # Step 6: Final cleanup (optional)
    segmentation_mask = morphological_operations(segmentation_mask, kernel_size=kernel_size)

    return segmentation_mask



import numpy as np
import torch
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
from skimage.morphology import remove_small_objects, remove_small_holes
from tqdm import tqdm


import src.data.utils as utils
import src.data.preprocess_data as data


def remove_small(mask, min_size = 10):
    import numpy as np
    from skimage.morphology import remove_small_objects

    cleaned_mask = np.zeros_like(mask)
    for class_value in [1, 2, 3, 4]:
        binary_mask = (mask == class_value)
        cleaned_binary_mask = remove_small_objects(binary_mask, min_size=min_size)
        cleaned_binary_mask = remove_small_holes(cleaned_binary_mask, area_threshold=min_size)

        cleaned_mask[cleaned_binary_mask] = class_value
    return cleaned_mask

import cv2
import numpy as np

def refine_segmentation(mask, kernel_size=3):
    """
    Apply morphological opening followed by closing to smooth a multiclass segmentation mask.
    
    Args:
        mask (np.ndarray): The segmentation mask (H, W) with integer labels.
        kernel_size (int): Size of the structuring element.
    
    Returns:
        np.ndarray: The refined segmentation mask.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    refined_mask = np.zeros_like(mask)

    for class_id in np.unique(mask):
        if class_id == 0:  # Assuming 0 is the background, skip it if necessary
            continue
        class_mask = (mask == class_id).astype(np.uint8)
        opened = cv2.morphologyEx(class_mask, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        closed_again = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel)

        refined_mask[closed_again > 0] = class_id  # Restore class label

    return refined_mask


def apply_crf_batch(logits, image):
    batch_size = logits.shape[0]
    refined_preds = []
    for i in range(batch_size):
        # Convert to NumPy
        img_np = image[i, :3].permute(1, 2, 0).cpu().numpy()  # Use 3 channels (e.g., RGB or PCA)
        logits_np = logits[i].cpu().numpy()  # Convert logits to NumPy
        # Apply CRF
        refined_mask = apply_dense_crf(img_np, logits_np)
        # Append result
        refined_preds.append(refined_mask)
    # Convert back to tensor
    refined_preds = torch.tensor(refined_preds)  # Shape: [B, W, H]
    return refined_preds


import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

def apply_dense_crf_(image: np.ndarray, logits: np.ndarray, n_iter = 10, use_custom_compat: bool = False):
    """
    Applies DenseCRF post-processing to the given logits with optional custom compatibility rules.

    Args:
        image (np.ndarray): The input image in shape (H, W, 3) with values in [0, 255].
        logits (np.ndarray): The predicted logits of shape (num_classes, H, W).
        use_custom_compat (bool): Whether to use a custom compatibility matrix.

    Returns:
        np.ndarray: The refined segmentation map of shape (H, W).
    """
    H, W, _ = image.shape
    num_classes = logits.shape[0]

    # Convert logits to unary potentials
    unary = unary_from_softmax(logits)
    unary = np.ascontiguousarray(unary)

    # Initialize CRF
    d = dcrf.DenseCRF2D(W, H, num_classes)
    d.setUnaryEnergy(unary)

    # Default compatibility
    compat_matrix = np.ones((num_classes, num_classes), dtype=np.float32) * 10  # Default penalty

    if use_custom_compat:
        # Apply custom compatibility rules
        compat_matrix[0, 3] = 1  
        compat_matrix[3, 0] = 1  
        compat_matrix[1, 2] = 2  
        compat_matrix[2, 1] = 2
        compat_matrix[1, 4] = 2  
        compat_matrix[4, 1] = 2
        compat_matrix[2, 4] = 2  
        compat_matrix[4, 2] = 2

    # Add pairwise Gaussian term
    d.addPairwiseGaussian(sxy=(3, 3), compat=5)

    # Add pairwise bilateral term (appearance term using RGB image)
    d.addPairwiseBilateral(sxy=(10, 100), srgb=(15, 15, 15), rgbim=np.ascontiguousarray(image), compat=15)

    # Set custom compatibilities
    #d.setClassCompatibility(compat_matrix.astype(np.int32).flatten())

    # Run inference
    Q = d.inference(n_iter)  # 10 iterations
    refined = np.argmax(Q, axis=0).reshape(H, W)

    return refined, Q


import cv2

def image_to_RGB(image, pca_file = None):
    """
    Image must be [12, W, H]
    """
    if pca_file:
        rgb_img = torch.Tensor(data.apply_pca_weights(image, pca_file))
    else:
        rgb_img = image[[3, 2, 1], :, :]
    rgb_img = np.transpose(rgb_img, (1, 2, 0))  # shape: (3, W, H)

#    rgb_img = rgb_img.permute(1, 2, 0).detach().cpu().numpy()
    rgb_img-=np.min(rgb_img)
    rgb_img/=np.max(rgb_img)
    
    return (rgb_img*255).astype(np.uint8)

def enforce_containment(segmentation):
    """
    Enforce that class 3 must be inside class 4.

    Parameters:
    - segmentation: np.ndarray (H, W) with class labels.

    Returns:
    - Corrected segmentation map.
    """

    class_3_mask = (segmentation == 3).astype(np.uint8)
    class_4_mask = (segmentation == 4).astype(np.uint8)

    # Find connected components of class 3
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(class_3_mask, connectivity=8)

    for i in range(1, num_labels):  # Skip background (label 0)
        component_mask = (labels == i)

        # Check if this component overlaps with class 4
        if np.sum(component_mask & class_4_mask) == 0:
            segmentation[component_mask] = 4  # Change to class 4 if not inside

    return segmentation


import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from skimage.morphology import opening, closing, disk
#from osgeo import gdal

def smooth_logits(logits, method='gaussian', sigma=1):
    """Apply smoothing to logits using Gaussian or TVD."""
    if method == 'gaussian':
        return gaussian_filter(logits, sigma=sigma)
    elif method == 'tvd':
        from skimage.restoration import denoise_tv_chambolle
        return np.stack([denoise_tv_chambolle(logits[i], weight=0.1) for i in range(logits.shape[0])])
    return logits

def apply_dense_crf__(image, logits):
    """Applies CRF to refine logits."""
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian

    num_classes, H, W = logits.shape
    d = dcrf.DenseCRF2D(W, H, num_classes)
    U = unary_from_softmax(logits)
    d.setUnaryEnergy(U)
    
    # Pairwise Gaussian filter
    d.addPairwiseEnergy(create_pairwise_gaussian((3, 3), (W, H)), compat=3)
    
    # Pairwise Bilateral filter
    print(image.shape)
    d.addPairwiseEnergy(create_pairwise_bilateral((50, 50), (10, 10, 10), image, chdim=2), compat=5)
    #(sdims=(80, 80), schan=(13, 13, 13),img=img, chdim=2)

    Q = np.array(d.inference(10))
    refined = np.argmax(Q, axis=0).reshape(H, W)
    return Q.reshape((num_classes, H, W)), refined

def morphological_operations(mask, kernel_size=3):
    """Apply opening and closing to remove noise and smooth edges."""
    kernel = disk(kernel_size)
    cleaned = np.zeros_like(mask)
    
    for cls in np.unique(mask):
        binary = (mask == cls).astype(np.uint8)
        binary = opening(binary, kernel)
        binary = closing(binary, kernel)
        cleaned[binary > 0] = cls
    
    return cleaned



class ReconstructTile:
    def __init__(self, size = (10560, 10560), subtile_size = (10560//6, 10560//6), patch_size = 256, stride = 256, edge_removal = 8, num_classes = 5):
        self.tile = np.zeros(shape=size, dtype=np.float32)
        self.logits = np.zeros(shape=(num_classes, size[0], size[1]), dtype=np.float16)
        self.labels = -np.ones(shape=size, dtype=np.uint8)
        self.preds = -np.ones(shape=size, dtype=np.uint8)
        self.count = np.zeros(shape=size, dtype=np.int32)
        self.crf = -np.ones(shape=size, dtype=np.uint8)
        self.cleaned_preds = -np.ones(shape=size, dtype=np.uint8)
        self.cleaned_crf = -np.ones(shape=size, dtype=np.uint8)
        self.vi_completion = -np.ones(shape=size, dtype=np.uint8)
        self.image = np.ones(shape=(12, size[0], size[1]), dtype=np.float32)
        self.rgb_image = np.zeros(shape=(size[0], size[1], 3), dtype=np.uint8)
        self.ss = subtile_size
        self.ps = patch_size
        self.stride = stride
        self.overlap=patch_size - stride
        self.size = size
        self.edge_removal = edge_removal
    def add_batch(self, xs, ys, fs, logits, preds, labels, imgs):
        batch_size = preds.shape[0]
        #print(xs, ys)
        for i in range(batch_size):
            
            x = xs[i] #patch offset
            y = ys[i]
            f = fs[i]
            offx, offy = utils.subtile_offset(f)
            logits_patch = np.squeeze(logits[i].cpu().detach().numpy()).astype(np.float16)
            img = imgs[i].cpu().detach().numpy() #shape: (12, width, height)
            labels_patch = np.squeeze(labels[i].cpu().detach().numpy())
            
            if x<=self.edge_removal:
                left = 0
            else:
                left = self.edge_removal
            if x+self.ps>=self.ss[0]-self.edge_removal:
                right = self.ps
            else:
                right = self.ps-self.edge_removal
                
            if y<=self.edge_removal:
                up = 0
            else:
                up = self.edge_removal
            if y+self.ps>=self.ss[1]-self.edge_removal:
                down = self.ps
            else:
                down = self.ps-self.edge_removal
            
            y0 = y+offy+up
            y1 = y+offy+down
            x0 = x+offx+left
            x1 = x+offx+right
            self.count[y0:y1, x0:x1] +=np.ones(shape=(down-up, right-left), dtype=np.uint8)
            self.logits[:,y0:y1, x0:x1] += logits_patch[:,up:down,left:right]      
            self.labels[y0:y1, x0:x1] = labels_patch[up:down, left:right]
            self.image[:,y0:y1, x0:x1] = img[:,up:down, left:right]
            rgb_img = image_to_RGB(img[:,up:down, left:right])
            self.rgb_image[y0:y1, x0:x1,:] = rgb_img#
            
        return
    
    def set_pred(self):            
        self.logits = self.logits/self.count
        self.probs = softmax(self.logits)
        self.preds = np.argmax(self.probs, axis=0)
        
        
    def post_process(self, x_idx, y_idx):
        import numpy as np
        from scipy.ndimage import binary_opening, binary_closing, distance_transform_edt
        from skimage.morphology import disk


        
        ppps = self.ss[0] #post processing patch size

        x_range = list(range(0,self.size[0],self.ss[0]))
        y_range = list(range(0,self.size[1],self.ss[1]))

        print(x_range)
        print(y_range)

        x=x_range[x_idx]
        y=y_range[y_idx]
        labels = self.labels[x:x+self.ss[0], y:y+self.ss[1]]

        logits_patch = self.logits[:,x:x+self.ss[0], y:y+self.ss[1]]
        smooth_logits_patch = smooth_logits(logits_patch, method='gaussian', sigma=1)

        prob_patch = softmax(logits_patch)#self.probs[:,x:x+self.ss[0], y:y+self.ss[1]]
        rgb_patch = self.rgb_image[x:x+self.ss[0], y:y+self.ss[1],:]
        pred_patch = self.preds[x:x+self.ss[0], y:y+self.ss[1]]


        clean_pred = apply_dense_crf(rgb_patch, logits_patch)#(pred_patch, kernel_size=5))
        clean_pred = np.argmax(clean_pred, axis=0).reshape(self.ss)
        clean_noholes = remove_small(clean_pred, min_size = 50)
        clean_noholes_2 = refine_segmentation(clean_pred, kernel_size=5)
        noholes = remove_small(pred_patch, min_size = 50)
        noholes2 = refine_segmentation(pred_patch, kernel_size=5)
        rules = enforce_enclosure_rules(clean_noholes)



        return labels, pred_patch, clean_pred, clean_noholes, clean_noholes_2, noholes, noholes2, rules

        logits_patch = self.logits[:,i:i+self.ss[0], j:j+self.ss[1]]
        rgb_patch = self.rgb_image[i:i+self.ss[0], j:j+self.ss[1],:]
        pred_patch = self.preds[i:i+self.ss[0], j:j+self.ss[1]]
        

        for i in tqdm(range(0,self.size[0],self.ss[0])):
            for j in tqdm(range(0,self.size[1],self.ss[1]),leave=False):
                logits_patch = self.logits[:,i:i+self.ss[0], j:j+self.ss[1]]
                rgb_patch = self.rgb_image[i:i+self.ss[0], j:j+self.ss[1],:]
                pred_patch = self.preds[i:i+self.ss[0], j:j+self.ss[1]]
                clean = full_segmentation_pipeline(rgb_patch, logits_patch, smooth_method='gaussian', sigma=1, min_size=100, kernel_size=5)
                
                crf, Q = apply_dense_crf__(rgb_patch, logits_patch)#, use_custom_compat=True, n_iter=10)
                    
                #smooth = smooth_segmentation(pred_patch, kernel_size=5)
                clean_pred = apply_dense_crf(self.image, logits_patch)#(pred_patch, kernel_size=5))
                clean_crf = remove_small(crf, min_size = 50)
                vazios = enforce_enclosure_rules(pred_patch)

                return clean, crf, Q, clean_crf, clean_pred, vazios
                self.clean[i:i+self.ss[0], j:j+self.ss[1]] = clean
                self.crf[i:i+self.ss[0], j:j+self.ss[1]] = crf
                self.Q[i:i+self.ss[0], j:j+self.ss[1]] = Q
                self.cleaned_crf[i:i+self.ss[0], j:j+self.ss[1]] = clean_crf
                self.cleaned_preds[i:i+self.ss[0], j:j+self.ss[1]] = clean_pred
                self.vi_completion[i:i+self.ss[0], j:j+self.ss[1]] = vazios
                break
            break

import numpy as np
import cv2

def enforce_enclosure_rules(mask):
    """
    Modify the mask by ensuring:
    1. Class 3 must be inside class 4; if not, turn it into class 0.
    2. Class 0 inside class 4 should be converted into class 3.

    Args:
        mask (numpy.ndarray): 2D array representing the class labels.

    Returns:
        numpy.ndarray: Modified mask.
    """
    mask = mask.copy()  # Avoid modifying the original

    class_3 = (mask == 3).astype(np.uint8)  # Binary mask for class 3
    class_0 = (mask == 0).astype(np.uint8)  # Binary mask for class 0
    class_4 = (mask == 4).astype(np.uint8)  # Binary mask for class 4

    # Find connected components of class 3
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(class_3, connectivity=8)

    for i in range(1, num_labels):  # Skip background (label 0)
        component_mask = (labels == i).astype(np.uint8)

        # Dilate slightly and check if it expands outside class 4
        dilated = cv2.dilate(component_mask, np.ones((3,3), np.uint8))
        if np.any((dilated > 0) & (class_4 == 0)):  # If dilated part exits class 4
            mask[labels == i] = 0  # Convert to class 0

    # Find connected components of class 0
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(class_0, connectivity=8)

    for i in range(1, num_labels):  # Skip background (label 0)
        component_mask = (labels == i).astype(np.uint8)

        # Dilate slightly and check if it is fully inside class 4
        dilated = cv2.dilate(component_mask, np.ones((3,3), np.uint8))
        if np.all((dilated > 0) & (class_4 > 0)):  # If fully enclosed in class 4
            mask[labels == i] = 3  # Convert to class 3

    return mask
