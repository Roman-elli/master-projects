import numpy as np
from emotiefflib.facial_analysis import EmotiEffLibRecognizer

LABEL_CANONICAL_MAP = {
    "anger": "Anger",
    "angry": "Anger",
    "contempt": "Contempt",
    "disgust": "Disgust",
    "fear": "Fear",
    "happiness": "Happiness",
    "happy": "Happiness",
    "neutral": "Neutral",
    "sadness": "Sadness",
    "sad": "Sadness",
    "surprise": "Surprise",
    "surprised": "Surprise",
}

def canonicalize_expression(label: str) -> str:
    return LABEL_CANONICAL_MAP.get(label.strip().lower(), label.strip())

class ExpressionRecognizer:
    def __init__(self, model_name: str = "enet_b0_8_best_afew", device: str = "cpu"):
        """
        Inicializa o reconhecedor de expressões faciais em ONNX (EmotiEffLib).
        As previsões representam configurações faciais visíveis e não estados emocionais internos.
        """
        self.model_name = model_name
        self.recognizer = EmotiEffLibRecognizer(
            engine="onnx",
            model_name=model_name,
            device=device,
        )
        idx_map = getattr(self.recognizer, "idx_to_emotion_class", None) or self.recognizer.idx_to_class
        self.classes = [canonicalize_expression(idx_map[i]) for i in sorted(idx_map.keys())]

    def predict(self, face_rgb: np.ndarray, allowed_classes: list[str] | None = None, top_k: int = 3):
        """
        Recebe um recorte facial RGB e devolve a expressão dominante, confiança e top-k probabilidades.
        """
        _, scores = self.recognizer.predict_emotions(face_rgb, logits=False)
        probs = np.asarray(scores, dtype=np.float32).squeeze()

        # Garantia extra: aplicar softmax caso o backend devolva logits
        if probs.min() < 0.0 or not np.isclose(probs.sum(), 1.0, atol=1e-2):
            exp_p = np.exp(probs - np.max(probs))
            probs = exp_p / exp_p.sum()

        prob_dict = {cls: float(p) for cls, p in zip(self.classes, probs)}

        # Harmonização de classes (ex.: remover Contempt ao avaliar no BTFER7)
        if allowed_classes is not None:
            allowed_canonical = [canonicalize_expression(c) for c in allowed_classes]
            prob_dict = {cls: prob_dict[cls] for cls in allowed_canonical if cls in prob_dict}
            total = sum(prob_dict.values())
            if total > 0:
                prob_dict = {cls: p / total for cls, p in prob_dict.items()}

        sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
        best_label, best_conf = sorted_probs[0]

        return {
            "label": best_label,
            "confidence": best_conf,
            "top_k": sorted_probs[:top_k],
            "probabilities": prob_dict,
        }