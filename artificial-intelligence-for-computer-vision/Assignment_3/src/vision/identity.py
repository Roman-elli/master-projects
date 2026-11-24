import os
import glob
import json
from datetime import datetime
from typing import Dict, List, Optional

import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis


class IdentityRecognizer:
    def __init__(
        self,
        model_name: str = "buffalo_l",
        gallery_path: Optional[str] = "data/gallery/gallery.json",
        similarity_threshold: float = 0.40,
        max_identities: int = 19,
        use_gpu: bool = False,
    ):
        """
        Reconhecimento facial em conjunto fechado com rejeição de desconhecidos ('Unknown')
        baseado em embeddings normalizados ArcFace (InsightFace) e similaridade de cosseno.
        """
        self.model_name = model_name
        self.gallery_path = gallery_path
        self.similarity_threshold = float(similarity_threshold)
        self.max_identities = min(int(max_identities), 19)  # Requisito: menos de 20 identidades

        self.rec_model = self._load_arcface_model(model_name, use_gpu)
        self.gallery: Dict[str, dict] = {}
        self.load_gallery()

    def _load_arcface_model(self, model_name: str, use_gpu: bool):
        ctx_id = 0 if use_gpu else -1
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]

        # Tentar carregar diretamente o ficheiro ONNX do ArcFace já existente em ./models/<model_name>/
        model_dir = os.path.join(".", "models", model_name)
        onnx_candidates = glob.glob(os.path.join(model_dir, "w600k_*.onnx")) + glob.glob(
            os.path.join(model_dir, "*r50*.onnx")
        )
        if onnx_candidates:
            rec_model = insightface.model_zoo.get_model(onnx_candidates[0], providers=providers)
            rec_model.prepare(ctx_id=ctx_id)
            return rec_model

        # Fallback: descarregar/inicializar via FaceAnalysis se os pesos ainda não existirem
        app = FaceAnalysis(name=model_name, root=".", allowed_modules=["detection", "recognition"])
        app.prepare(ctx_id=ctx_id)
        return app.models["recognition"]

    def extract_embedding(self, aligned_face_bgr: np.ndarray) -> np.ndarray:
        """
        Extrai o vetor embedding ArcFace normalizado (L2) de um recorte facial alinhado (112x112 BGR).
        """
        if aligned_face_bgr.shape[:2] != (112, 112):
            aligned_face_bgr = cv2.resize(aligned_face_bgr, (112, 112))

        feat = self.rec_model.get_feat(aligned_face_bgr).flatten().astype(np.float32)
        norm = np.linalg.norm(feat)
        return feat / norm if norm > 0 else feat

    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Calcula a similaridade de cosseno entre dois embeddings normalizados.
        """
        n1, n2 = np.linalg.norm(emb1), np.linalg.norm(emb2)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(np.dot(emb1, emb2) / (n1 * n2))

    # ------------------------------------------------------------------
    # Gestão local da galeria (com consentimento e eliminação)
    # ------------------------------------------------------------------

    def load_gallery(self):
        self.gallery = {}
        if not self.gallery_path or not os.path.exists(self.gallery_path):
            return

        with open(self.gallery_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        for name, entry in raw.items():
            vec = np.asarray(entry["template"], dtype=np.float32)
            norm = np.linalg.norm(vec)
            self.gallery[name] = {
                "template": vec / norm if norm > 0 else vec,
                "num_samples": int(entry.get("num_samples", 1)),
                "consent": bool(entry.get("consent", True)),
                "enrolled_at": entry.get("enrolled_at", ""),
            }

    def save_gallery(self):
        if not self.gallery_path:
            return
        os.makedirs(os.path.dirname(self.gallery_path), exist_ok=True)
        serializable = {
            name: {
                "template": entry["template"].tolist(),
                "num_samples": entry["num_samples"],
                "consent": entry["consent"],
                "enrolled_at": entry["enrolled_at"],
            }
            for name, entry in self.gallery.items()
        }
        with open(self.gallery_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)

    def enrol_from_embeddings(
        self,
        name: str,
        embeddings: List[np.ndarray],
        consent_given: bool = True,
        save: bool = True,
    ):
        """
        Regista ou atualiza uma identidade a partir de vários embeddings (template médio normalizado).
        """
        name = name.strip()
        if not name or name.lower() == "unknown":
            raise ValueError("Nome de identidade inválido.")
        if not consent_given:
            raise PermissionError("O registo exige consentimento informado explícito do participante.")
        if not embeddings:
            raise ValueError("É necessário pelo menos um embedding válido para o registo.")
        if name not in self.gallery and len(self.gallery) >= self.max_identities:
            raise ValueError(f"Limite máximo da galeria atingido ({self.max_identities} identidades).")

        mean_vec = np.mean(np.stack(embeddings, axis=0), axis=0).astype(np.float32)
        norm = np.linalg.norm(mean_vec)
        template = mean_vec / norm if norm > 0 else mean_vec

        self.gallery[name] = {
            "template": template,
            "num_samples": len(embeddings),
            "consent": True,
            "enrolled_at": datetime.now().isoformat(timespec="seconds"),
        }
        if save:
            self.save_gallery()

    def enrol_identity(
        self,
        name: str,
        aligned_faces_bgr: List[np.ndarray],
        consent_given: bool = True,
        save: bool = True,
    ):
        """
        Extrai embeddings de múltiplos recortes alinhados (112x112 BGR) e regista o template médio.
        """
        embeddings = [self.extract_embedding(face) for face in aligned_faces_bgr]
        self.enrol_from_embeddings(name, embeddings, consent_given=consent_given, save=save)

    def remove_identity(self, name: str, save: bool = True) -> bool:
        if name in self.gallery:
            del self.gallery[name]
            if save:
                self.save_gallery()
            return True
        return False

    def clear_gallery(self, delete_file: bool = True):
        self.gallery.clear()
        if delete_file and self.gallery_path and os.path.exists(self.gallery_path):
            os.remove(self.gallery_path)

    def list_identities(self) -> List[str]:
        return sorted(self.gallery.keys())

    # ------------------------------------------------------------------
    # Identificação 1-para-N com rejeição de desconhecidos
    # ------------------------------------------------------------------

    def identify_embedding(self, embedding: np.ndarray, threshold: Optional[float] = None):
        """
        Compara um embedding com todos os templates da galeria e devolve a identidade ou 'Unknown'.
        """
        thresh = self.similarity_threshold if threshold is None else float(threshold)

        if not self.gallery:
            return {
                "identity": "Unknown",
                "best_match": None,
                "similarity": 0.0,
                "is_known": False,
                "all_scores": {},
            }

        scores = {
            name: self.cosine_similarity(embedding, entry["template"])
            for name, entry in self.gallery.items()
        }
        best_match, best_sim = max(scores.items(), key=lambda x: x[1])
        is_known = best_sim >= thresh

        return {
            "identity": best_match if is_known else "Unknown",
            "best_match": best_match,
            "similarity": best_sim,
            "is_known": is_known,
            "all_scores": scores,
        }

    def identify(self, aligned_face_bgr: np.ndarray, threshold: Optional[float] = None):
        emb = self.extract_embedding(aligned_face_bgr)
        return self.identify_embedding(emb, threshold=threshold)