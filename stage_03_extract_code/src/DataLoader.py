from PIL import Image
import tensorflow as tf
import albumentations as A
import numpy as np
from skimage.measure import label, regionprops

def get_aug(aug, min_area=0., min_visibility=0.):
    bbox_params = A.BboxParams(format='coco', min_area=min_area, min_visibility=min_visibility, label_fields=['category_id'])
    return A.Compose(aug, bbox_params)

def make_augmentation(output_size, is_validation):
    aug = None
    if is_validation:
        aug = get_aug([                      
              A.Resize(width=output_size[0], height=output_size[1], always_apply=True),
              A.Normalize(),
            ], min_visibility=0.1)
    else:
        aug = get_aug([
                A.RGBShift(p=0.1),
                A.OneOf([
                  A.RandomBrightnessContrast(p=0.5),            
                  A.HueSaturationValue(),
                  A.RandomGamma(p=0.25),
                  A.RandomBrightness(p=0.25),
                  A.Blur(blur_limit=2,p=0.25),
                ],p=0.01),

                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.05),

                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.05, rotate_limit=15,  border_mode=0,  p=0.2, value=(144.75479165, 137.70713403, 129.666091), mask_value=0.0 ),

                A.Resize(width=output_size[0], height=output_size[1], always_apply=True),

                A.Normalize(),
            ], min_visibility=0.1)
    return aug

class DataLoader(tf.keras.utils.Sequence):
    def __init__(self, 
                 dataset, 
                 batch_size,
                 augmentation_fn,
                 shuffle=True, 
                 output_size=(512,512), 
                 **kwargs):
        self.dataset = dataset
        self._len = len(self.dataset)
        self.indices = range(self._len)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._output_size = output_size
        self.aug_fn = augmentation_fn
        self.on_epoch_end()
        
    def __len__(self):        
        return self._len // self.batch_size

    def __getitem__(self, index):
        """ Generate one batch of data. """
        s = index * self.batch_size % self._len
        e = s + self.batch_size
        indices = self.indices[s:e]

        return self.__data_generator(indices)

    def on_epoch_end(self):
        """ Updates indices after each epoch. """
        if self.shuffle:
            self.indices = np.random.permutation(self._len)
            
    def augment(self, img, mask):
        label_image = label(mask)
        bboxes = []
        for region in regionprops(label_image):
            if region.area >= 100:
                minr, minc, maxr, maxc = region.bbox
                bboxes.append((minc, minr, maxc-minc, maxr-minr ))
                                
        if len(bboxes) == 0:
            #print ("no bboxes")
            bboxes = [ [0, 0, img.shape[1], img.shape[0]] ]            

        new_img = None
        new_mask = None
        try:
            annotations = {'image': img, 
                   "masks" : [mask],
                   'bboxes': bboxes,
                   #'cropping_bbox': [minc, minr,  maxc - minc , maxr - minr],
                   #'cropping_bbox': [0.1, 0.1, 0.2, 0.2],
                   'category_id' : [255] * len(bboxes)}
            
            augmented = self.aug_fn(**annotations)
            new_img = augmented['image']
            new_mask = augmented["masks"][0]
        except Exception as e:
            print(e)
            new_img = img
            new_mask = mask
        return new_img, new_mask
        
    def __data_generator(self, indices):
        # Init the matrix
        batch_images, batch_target = [], []
        for idx in indices:
            image_path, label_path = self.dataset[idx]
            image = np.array(Image.open(image_path).convert('RGB'))
            target = np.array(Image.open(label_path))

            ## Rescale masks from [0; 255] to [0; 1]
            target[target > 0] = 1
            target = target.astype('float32')
            
            image, target = self.augment(image, target)            
            image_shape = image.shape[:2]    

            # if shape of mask is not h*w*c
            if len(target.shape) != 3:
                ## the keras model require h*w*1
                target = np.expand_dims(target, axis=-1)
            
            batch_images.append(image)
            batch_target.append(target)
        
        if len(batch_images) < self.batch_size:
            pad_images = [np.zeros_like(batch_images[0]) 
                          for _ in range(self.batch_size-len(batch_images))]
            pad_target = [np.zeros_like(batch_target[0]) 
                          for _ in range(self.batch_size-len(batch_target))]
            batch_images.extend(pad_images)
            batch_target.extend(pad_target)

        return np.stack(batch_images), np.stack(batch_target)