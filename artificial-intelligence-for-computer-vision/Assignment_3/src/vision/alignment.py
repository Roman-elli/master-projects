import cv2
import numpy as np
from insightface.utils import face_align

class FaceAligner:
    def __init__(self, crop_size: int = 112):
        """
        Inicializa o módulo de alinhamento facial.
        crop_size: O tamanho da imagem de saída (ex: 112 para modelos baseados em ArcFace/ResNet).
        """
        self.crop_size = crop_size

    def align(self, frame: np.ndarray, landmarks: list) -> np.ndarray:
        """
        Gera o recorte normalizado e alinhado do rosto usando os 5 pontos faciais[cite: 3, 4].
        
        :param frame: Imagem original bruta.
        :param landmarks: Lista de 5 pontos [[x1,y1], [x2,y2], ...] vindos da deteção.
        :return: Imagem do rosto recortada e alinhada.
        """
        # A função norm_crop calcula a matriz de transformação afim e aplica o warpAffine
        landmarks_array = np.array(landmarks, dtype=np.float32)
        aligned_face = face_align.norm_crop(frame, landmark=landmarks_array, image_size=self.crop_size)
        
        return aligned_face