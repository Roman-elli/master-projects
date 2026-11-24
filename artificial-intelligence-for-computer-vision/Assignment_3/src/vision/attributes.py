import numpy as np
from deepface import DeepFace


def format_age_range(age) -> str:
    """
    Formata a idade aparente estimada num intervalo de 10 anos (ex.: '30-39').
    """
    if age is None:
        return "Unknown"
    age_int = int(round(float(age)))
    if age_int < 10:
        return "0-9"
    if age_int >= 70:
        return "70+"
    low = (age_int // 10) * 10
    return f"{low}-{low + 9}"


class AgeGenderEstimator:
    def __init__(self, gender_conf_thresh: float = 0.70):
        """
        Inicializa o estimador de idade aparente e classificação visual de género (DeepFace).
        Previsões de género com confiança inferior a gender_conf_thresh são marcadas como 'Uncertain'.
        """
        self.gender_conf_thresh = gender_conf_thresh

    def estimate(self, face_bgr: np.ndarray):
        """
        Estima idade aparente, faixa etária e categoria visual de género (definida pelo dataset)
        a partir de um recorte facial em formato BGR.
        """
        try:
            results = DeepFace.analyze(
                img_path=face_bgr,
                actions=["age", "gender"],
                enforce_detection=False,
                silent=True,
            )

            res = results[0] if isinstance(results, list) else results

            raw_age = res.get("age")
            age = int(round(float(raw_age))) if raw_age is not None else None
            age_range = format_age_range(age)

            dominant_gender = res.get("dominant_gender")
            gender_probs = res.get("gender", {})

            if isinstance(gender_probs, dict) and dominant_gender in gender_probs:
                conf = float(gender_probs[dominant_gender])
                if conf > 1.0:  # DeepFace devolve percentagens (0-100)
                    conf /= 100.0
            else:
                conf = 0.0

            gender_raw = "Male" if dominant_gender == "Man" else "Female"
            gender = gender_raw if conf >= self.gender_conf_thresh else "Uncertain"

            return {
                "age": age,
                "age_range": age_range,
                "gender": gender,
                "gender_raw": gender_raw,
                "gender_confidence": conf,
            }

        except Exception as e:
            print(f"Erro na estimativa com DeepFace: {e}")
            return {
                "age": None,
                "age_range": "Unknown",
                "gender": "Uncertain",
                "gender_raw": None,
                "gender_confidence": 0.0,
            }