import cv2
import numpy as np

def calculate_homography(kps1, kps2, matches, ransac_thresh=3.0):
    """
    Calcula a matriz de homografia entre duas imagens usando os pontos correspondentes e o RANSAC.
    """
    if len(matches) < 4:
        raise ValueError("São precisos pelo menos 4 matches para calcular a homografia.")
        
    # 1. Extrair as coordenadas (x, y) dos keypoints que deram match
    src_pts = np.float32([kps1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kps2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # 2. Estimar a matriz de Homografia (H) usando RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
    
    # 3. Converter a máscara do OpenCV para uma lista (1 = inlier, 0 = outlier)
    matches_mask = mask.ravel().tolist()
    
    return H, matches_mask, src_pts, dst_pts