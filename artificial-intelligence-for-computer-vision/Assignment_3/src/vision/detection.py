import cv2
import time
import numpy as np
from insightface.app import FaceAnalysis

class FaceDetector:
    def __init__(self, model_name: str = 'buffalo_l', det_thresh: float = 0.5, use_gpu: bool = False):
        """
        Inicializa o detetor SCRFD via InsightFace.
        O modelo 'buffalo_l' utiliza o SCRFD para a deteção facial.
        """
        ctx_id = 0 if use_gpu else -1
        # Restringimos o módulo apenas a 'detection' para poupar recursos na avaliação inicial
        self.app = FaceAnalysis(name=model_name, root='.', allowed_modules=['detection'])
        self.app.prepare(ctx_id=ctx_id, det_thresh=det_thresh)

    def detect(self, frame: np.ndarray):
        """
        Processa o frame e devolve as bounding boxes, landmarks e níveis de confiança[cite: 3].
        """
        start_time = time.perf_counter()
        faces = self.app.get(frame)
        inference_time = (time.perf_counter() - start_time) * 1000 # em milissegundos
        
        results = []
        for face in faces:
            results.append({
                'bbox': face.bbox.astype(int).tolist(),     # [x1, y1, x2, y2]
                'confidence': float(face.det_score),        # Confiança da deteção
                'landmarks': face.kps.astype(int).tolist()  # 5 pontos faciais para futuro alinhamento[cite: 4]
            })
            
        return results, inference_time