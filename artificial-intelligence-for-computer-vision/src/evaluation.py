import numpy as np
import cv2

def compute_reprojection_error(src_pts, dst_pts, H):
    """
    Transforma os src_pts usando H e calcula a distância euclidiana para os dst_pts.
    Retorna o erro médio e um array booleano indicando se o erro é < 3 pixels (correct match).
    """
    if H is None or len(src_pts) == 0:
        return 0.0, np.zeros(len(src_pts), dtype=bool)

    # Converter os pontos para o formato (N, 1, 2) necessário para a função perspectiveTransform
    src_pts_reshaped = src_pts.reshape(-1, 1, 2)
    
    # Aplicar a homografia aos pontos da imagem 1
    transformed_pts = cv2.perspectiveTransform(src_pts_reshaped, H)
    transformed_pts = transformed_pts.reshape(-1, 2)
    dst_pts_reshaped = dst_pts.reshape(-1, 2)
    
    # Calcular a distância Euclidiana (L2) entre o ponto transformado e o ponto real na img2
    errors = np.linalg.norm(transformed_pts - dst_pts_reshaped, axis=1)
    
    # Segundo o protocolo do guião, um match é correto se o erro < 3 pixels
    correct_matches_mask = errors < 3.0
    avg_error = np.mean(errors)
    
    return avg_error, correct_matches_mask

def compute_evaluation_metrics(num_features, num_putative, num_correct, num_correspondences_gt=None):
    """
    Calcula as fórmulas de avaliação oficiais da tabela.
    """
    metrics = {}
    
    # Putative Match Ratio = #N_putative / #N_features
    metrics['PMR'] = num_putative / num_features if num_features > 0 else 0
    
    # Precision = #N_correct / #N_putative
    metrics['Precision'] = num_correct / num_putative if num_putative > 0 else 0
    
    # Matching Score = #N_correct / #N_features
    metrics['Matching_Score'] = num_correct / num_features if num_features > 0 else 0
    
    # Recall = #N_correct / #N_correspondences (Se ground truth estiver disponível)
    if num_correspondences_gt and num_correspondences_gt > 0:
        metrics['Recall'] = num_correct / num_correspondences_gt
    else:
        metrics['Recall'] = None # Não é possível calcular sem Ground Truth
        
    return metrics