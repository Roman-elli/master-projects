import cv2

def get_detector(detector_name, **kwargs):    
    if detector_name == 'SIFT':
        return cv2.SIFT_create(**kwargs)
    elif detector_name == 'ORB':
        return cv2.ORB_create(**kwargs)
    elif detector_name == 'KAZE':
        return cv2.KAZE_create(**kwargs)
    elif detector_name == 'FAST':
        return cv2.FastFeatureDetector_create(**kwargs)
    elif detector_name == 'SURF':
        return cv2.xfeatures2d.SURF_create(**kwargs)
    else:
        raise ValueError(f"Detetor {detector_name} não é suportado.")

def detect_keypoints(image, detector):
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
        
    keypoints = detector.detect(gray_image, None)
    return keypoints