import cv2

def create_descriptor(descriptor_name):
    if descriptor_name == 'BRIEF_LIB':
        return cv2.xfeatures2d.BriefDescriptorExtractor_create()
    elif descriptor_name == 'SELF_BRIEF':
        return None
    elif descriptor_name == 'BRISK':
        return cv2.BRISK_create()
    elif descriptor_name == 'FREAK':
        return cv2.xfeatures2d.FREAK.create()
    
def compute_descriptor(descriptor, img_rgb, keypoints):
    return descriptor.compute(img_rgb, keypoints)