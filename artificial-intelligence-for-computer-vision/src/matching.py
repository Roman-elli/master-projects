import cv2

def get_matcher(nome_descritor):
    """
    Retorna o BFMatcher configurado com a norma correta.
    """
    # Lista de descritores que geram strings binárias
    descritores_binarios = ['ORB', 'BRIEF_LIB', 'BRISK', 'FREAK']
    
    if nome_descritor in descritores_binarios:
        norm_type = cv2.NORM_HAMMING
    else:
        norm_type = cv2.NORM_L2
        
    # crossCheck=False é obrigatório quando usamos knnMatch
    return cv2.BFMatcher(norm_type, crossCheck=False)

def match_and_filter(matcher, des1, des2, ratio_thresh=0.8):
    """
    Encontra as correspondências e filtra usando o Lowe's ratio test.
    """
    # 1. Encontrar os 2 melhores matches para cada ponto (k=2)
    knn_matches = matcher.knnMatch(des1, des2, k=2)
    
    # 2. Aplicar o ratio test
    good_matches = []
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
            
    return good_matches