import yaml
import os

class Config:
    _instance = None

    def __new__(cls, config_path="config.yaml"):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config(config_path)
        return cls._instance

    def _load_config(self, config_path):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Ficheiro de configuração não encontrado: {config_path}")
            
        with open(config_path, "r") as f:
            self._cfg = yaml.safe_load(f)

    # Propriedades de Deteção
    @property
    def det_model_name(self):
        return self._cfg.get("detection", {}).get("model_name", "buffalo_l")

    @property
    def det_thresh(self):
        return self._cfg.get("detection", {}).get("det_thresh", 0.5)

    # Propriedades de Avaliação (Paths)
    @property
    def wider_face_gt_path(self):
        return self._cfg.get("evaluation", {}).get("wider_face_gt", "")

    @property
    def wider_face_img_dir(self):
        return self._cfg.get("evaluation", {}).get("wider_face_img_dir", "")
    
    @property
    def utkface_img_dir(self):
        return self._cfg.get("evaluation", {}).get("utkface_img_dir", "")
    
    @property
    def btfer_img_dir(self):
        return self._cfg.get("evaluation", {}).get("btfer_img_dir", "")

    @property
    def expression_model_name(self):
        return self._cfg.get("attributes", {}).get("expression_model_name", "enet_b0_8_best_afew")
    
    @property
    def id_model_name(self):
        return self._cfg.get("identity", {}).get("model_name", "buffalo_l")

    @property
    def gallery_path(self):
        return self._cfg.get("identity", {}).get("gallery_path", "data/gallery/gallery.json")

    @property
    def gallery_eval_dir(self):
        return self._cfg.get("identity", {}).get("eval_dir", "data/datasets/gallery_eval")

    @property
    def id_similarity_thresh(self):
        return float(self._cfg.get("identity", {}).get("similarity_threshold", 0.40))

    @property
    def max_identities(self):
        return int(self._cfg.get("identity", {}).get("max_identities", 19))