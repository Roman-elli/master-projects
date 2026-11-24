import os
from collections import defaultdict
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    precision_recall_fscore_support,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
)
import numpy as np

AGE_GROUPS = ["0-12", "13-19", "20-39", "40-59", "60+"]


def get_age_group(age: int) -> str:
    if age <= 12:
        return "0-12"
    elif age <= 19:
        return "13-19"
    elif age <= 39:
        return "20-39"
    elif age <= 59:
        return "40-59"
    return "60+"


def parse_wider_face_val(gt_file_path: str, img_dir: str):
    """
    Lê o ficheiro wider_face_val_bbx_gt.txt e mapeia as anotações.
    Devolve um dicionário: { 'caminho_imagem': [[x, y, w, h], ...] }
    """
    annotations = {}
    with open(gt_file_path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        img_name = lines[i].strip()
        i += 1
        num_faces = int(lines[i].strip())
        i += 1

        bboxes = []
        if num_faces == 0:
            i += 1
        else:
            for _ in range(num_faces):
                data = list(map(int, lines[i].strip().split()[:4]))
                bboxes.append(data)
                i += 1

        img_path = os.path.join(img_dir, img_name)
        annotations[img_path] = bboxes

    return annotations


def compute_iou(boxA, boxB):
    """
    Calcula a Intersection over Union (IoU) de duas bounding boxes [x1, y1, x2, y2].
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea + 1e-5)


def evaluate_image_detections(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    Compara predições [x1, y1, x2, y2] com o ground truth [x, y, w, h] e devolve TP, FP e FN.
    """
    gt_xyxy = [[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in gt_boxes]

    tp = 0
    matched_gt = set()

    for pred in pred_boxes:
        best_iou = 0
        best_gt_idx = -1

        for i, gt in enumerate(gt_xyxy):
            if i in matched_gt:
                continue
            iou = compute_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        if best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_gt_idx)

    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


UTK_RACE_LABELS = {
    0: "White",
    1: "Black",
    2: "Asian",
    3: "Indian",
    4: "Others",
}


def evaluate_age_gender_metrics(
    y_true_age,
    y_pred_age,
    y_true_gender,
    y_pred_gender,
    y_pred_gender_filtered=None,
    y_true_race=None,
):
    """
    Calcula métricas globais e estratificadas para Idade Aparente (MAE, RMSE, MAE por faixa etária,
    exatidão de faixa etária e diferenças demográficas) e Género (global e apenas previsões confiantes).
    """
    mae = mean_absolute_error(y_true_age, y_pred_age)
    rmse = root_mean_squared_error(y_true_age, y_pred_age)

    # 1. MAE por faixa etária + exatidão de classificação em faixas etárias
    true_groups = [get_age_group(a) for a in y_true_age]
    pred_groups = [get_age_group(a) for a in y_pred_age]
    age_group_acc = accuracy_score(true_groups, pred_groups)

    grouped = defaultdict(lambda: {"true": [], "pred": []})
    for t_age, p_age, g in zip(y_true_age, y_pred_age, true_groups):
        grouped[g]["true"].append(t_age)
        grouped[g]["pred"].append(p_age)

    mae_by_group = {
        g: (mean_absolute_error(grouped[g]["true"], grouped[g]["pred"]), len(grouped[g]["true"]))
        for g in AGE_GROUPS
        if grouped[g]["true"]
    }

    # 2. MAE por género real (diferenças demográficas)
    gender_names = {0: "Male", 1: "Female"}
    mae_by_gender = {}
    for g_id, g_name in gender_names.items():
        idxs = [i for i, g in enumerate(y_true_gender) if g == g_id]
        if idxs:
            mae_by_gender[g_name] = (
                mean_absolute_error([y_true_age[i] for i in idxs], [y_pred_age[i] for i in idxs]),
                len(idxs),
            )

    # 3. Métricas demográficas por anotação de etnia do UTKFace (opcional)
    demo_by_race = {}
    if y_true_race is not None:
        for r_id, r_name in UTK_RACE_LABELS.items():
            idxs = [i for i, r in enumerate(y_true_race) if r == r_id]
            if idxs:
                r_mae = mean_absolute_error([y_true_age[i] for i in idxs], [y_pred_age[i] for i in idxs])
                r_g_acc = accuracy_score([y_true_gender[i] for i in idxs], [y_pred_gender[i] for i in idxs])
                demo_by_race[r_name] = {"mae": r_mae, "gender_acc": r_g_acc, "n": len(idxs)}

    # 4. Métricas de género globais (todas as previsões)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_gender, y_pred_gender, average=None, labels=[0, 1], zero_division=0
    )
    cm = confusion_matrix(y_true_gender, y_pred_gender, labels=[0, 1])

    # 5. Métricas de género após filtragem de baixa confiança (-1 = Uncertain)
    confident_metrics = None
    if y_pred_gender_filtered is not None:
        total = len(y_pred_gender_filtered)
        conf_idxs = [i for i, p in enumerate(y_pred_gender_filtered) if p in (0, 1)]
        n_uncertain = total - len(conf_idxs)
        if conf_idxs:
            c_true = [y_true_gender[i] for i in conf_idxs]
            c_pred = [y_pred_gender_filtered[i] for i in conf_idxs]
            c_p, c_r, c_f1, _ = precision_recall_fscore_support(
                c_true, c_pred, average=None, labels=[0, 1], zero_division=0
            )
            c_cm = confusion_matrix(c_true, c_pred, labels=[0, 1])
        else:
            c_p = c_r = c_f1 = [0.0, 0.0]
            c_cm = confusion_matrix([], [], labels=[0, 1])

        confident_metrics = {
            "n_confident": len(conf_idxs),
            "n_uncertain": n_uncertain,
            "uncertain_rate": n_uncertain / total if total > 0 else 0.0,
            "precision": c_p,
            "recall": c_r,
            "f1": c_f1,
            "confusion_matrix": c_cm,
        }

    return {
        "mae": mae,
        "rmse": rmse,
        "age_group_accuracy": age_group_acc,
        "mae_by_group": mae_by_group,
        "mae_by_gender": mae_by_gender,
        "demo_by_race": demo_by_race,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "confident_gender": confident_metrics,
    }

def evaluate_expression_metrics(y_true, y_pred, labels, top_n_confusions: int = 5):
    """
    Calcula Overall/Balanced Accuracy, Macro Precision/Recall/F1, Per-Class Recall,
    Matriz de Confusão e os principais pares de confusão.
    """
    overall_acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", labels=labels, zero_division=0
    )
    _, per_class_recall, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)

    confusions = [
        (int(cm[i, j]), labels[i], labels[j])
        for i in range(len(labels))
        for j in range(len(labels))
        if i != j and cm[i, j] > 0
    ]
    top_confusions = sorted(confusions, reverse=True)[:top_n_confusions]

    return {
        "overall_accuracy": overall_acc,
        "balanced_accuracy": bal_acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "per_class_recall": dict(zip(labels, per_class_recall)),
        "confusion_matrix": cm,
        "top_confusions": top_confusions,
        "classification_report": report,
    }
    
def parse_wider_face_val_detailed(gt_file_path: str, img_dir: str):
    """
    Lê as 10 colunas de anotação do WIDER FACE:
    x, y, w, h, blur, expression, illumination, invalid, occlusion, pose.
    """
    annotations = {}
    with open(gt_file_path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        img_name = lines[i].strip()
        i += 1
        num_faces = int(lines[i].strip())
        i += 1

        faces = []
        if num_faces == 0:
            i += 1
        else:
            for _ in range(num_faces):
                parts = list(map(int, lines[i].strip().split()[:10]))
                if len(parts) >= 10:
                    faces.append({
                        "bbox": parts[:4],         # [x, y, w, h]
                        "blur": parts[4],          # 0: clear, 1: normal, 2: heavy
                        "expression": parts[5],    # 0: typical, 1: exaggerate
                        "illumination": parts[6],  # 0: normal, 1: extreme
                        "invalid": parts[7],       # 0: valid, 1: invalid
                        "occlusion": parts[8],     # 0: none, 1: partial, 2: heavy
                        "pose": parts[9],          # 0: typical, 1: atypical
                    })
                i += 1

        img_path = os.path.join(img_dir, img_name)
        annotations[img_path] = faces

    return annotations


def evaluate_detection_robustness(
    preds_by_img: dict,
    gt_detailed: dict,
    iou_threshold: float = 0.5,
    small_thresh: int = 32,
    multi_face_min: int = 5,
):
    """
    Calcula o Recall (taxa de deteção) estratificado por condição difícil do WIDER FACE:
    rostos normais, rostos pequenos (< small_thresh px), pose atípica,
    oclusão parcial/pesada e cenas com múltiplos rostos (>= multi_face_min).
    Ignora anotações marcadas como invalid == 1 (protocolo padrão WIDER FACE).
    """
    categories = [
        "all_valid",
        "normal_size",
        "small_faces",
        "atypical_pose",
        "partial_occlusion",
        "heavy_occlusion",
        "multiple_faces",
    ]
    stats = {c: {"tp": 0, "total": 0} for c in categories}

    for img_path, pred_boxes in preds_by_img.items():
        gt_faces = [
            f for f in gt_detailed.get(img_path, [])
            if f["invalid"] == 0 and f["bbox"][2] > 0 and f["bbox"][3] > 0
        ]
        gt_xyxy = [[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in (f["bbox"] for f in gt_faces)]

        matched_gt = set()
        for pred in pred_boxes:
            best_iou, best_idx = 0.0, -1
            for idx, gt in enumerate(gt_xyxy):
                if idx in matched_gt:
                    continue
                iou = compute_iou(pred, gt)
                if iou > best_iou:
                    best_iou, best_idx = iou, idx
            if best_iou >= iou_threshold:
                matched_gt.add(best_idx)

        is_multi = len(gt_faces) >= multi_face_min
        for idx, face in enumerate(gt_faces):
            hit = 1 if idx in matched_gt else 0
            w, h = face["bbox"][2], face["bbox"][3]

            face_cats = ["all_valid"]
            face_cats.append("small_faces" if min(w, h) < small_thresh else "normal_size")
            if face["pose"] == 1:
                face_cats.append("atypical_pose")
            if face["occlusion"] == 1:
                face_cats.append("partial_occlusion")
            elif face["occlusion"] == 2:
                face_cats.append("heavy_occlusion")
            if is_multi:
                face_cats.append("multiple_faces")

            for c in face_cats:
                stats[c]["tp"] += hit
                stats[c]["total"] += 1

    return {
        c: {
            "recall": stats[c]["tp"] / stats[c]["total"] if stats[c]["total"] > 0 else 0.0,
            "tp": stats[c]["tp"],
            "total": stats[c]["total"],
        }
        for c in categories
    }
    
def calibrate_identity_threshold(known_scores, unknown_scores, thresholds=None):
    """
    Calibra o limiar de similaridade de cosseno exclusivamente no conjunto de validação.
    Minimiza a soma FAR + FRR (com preferência por menor FAR em caso de empate para controlar falsos positivos).
    """
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 201)

    known_arr = np.asarray(known_scores, dtype=np.float32)
    unknown_arr = np.asarray(unknown_scores, dtype=np.float32)

    far_list, frr_list = [], []
    best_thresh = 0.40
    best_err = float("inf")
    best_far = float("inf")

    for t in thresholds:
        far = float(np.mean(unknown_arr >= t)) if len(unknown_arr) > 0 else 0.0
        frr = float(np.mean(known_arr < t)) if len(known_arr) > 0 else 0.0
        far_list.append(far)
        frr_list.append(frr)

        total_err = far + frr
        if total_err < best_err - 1e-9 or (abs(total_err - best_err) <= 1e-9 and far < best_far):
            best_err = total_err
            best_far = far
            best_thresh = float(t)

    diffs = np.abs(np.asarray(far_list) - np.asarray(frr_list))
    eer_idx = int(np.argmin(diffs))
    eer = 0.5 * (far_list[eer_idx] + frr_list[eer_idx])

    return {
        "best_threshold": best_thresh,
        "eer": float(eer),
        "eer_threshold": float(thresholds[eer_idx]),
        "thresholds": np.asarray(thresholds),
        "far_curve": np.asarray(far_list),
        "frr_curve": np.asarray(frr_list),
    }


def evaluate_identity_metrics(y_true, y_pred_open, y_pred_closed, similarities, thresholds=None):
    """
    Avalia reconhecimento de identidade em conjunto fechado + rejeição de desconhecidos ('Unknown'):
    Closed-Set Rank-1, Open-Set Rank-1 (DIR), FRR, False ID Rate, FAR, Unknown Rejection Rate
    e curvas de sensibilidade ao limiar.
    """
    known_idxs = [i for i, y in enumerate(y_true) if y != "Unknown"]
    unknown_idxs = [i for i, y in enumerate(y_true) if y == "Unknown"]

    n_known = len(known_idxs)
    n_unknown = len(unknown_idxs)

    closed_rank1 = (
        sum(1 for i in known_idxs if y_pred_closed[i] == y_true[i]) / n_known if n_known > 0 else 0.0
    )
    open_rank1 = (
        sum(1 for i in known_idxs if y_pred_open[i] == y_true[i]) / n_known if n_known > 0 else 0.0
    )
    frr = (
        sum(1 for i in known_idxs if y_pred_open[i] == "Unknown") / n_known if n_known > 0 else 0.0
    )
    false_id_rate = (
        sum(1 for i in known_idxs if y_pred_open[i] != "Unknown" and y_pred_open[i] != y_true[i]) / n_known
        if n_known > 0
        else 0.0
    )
    far = (
        sum(1 for i in unknown_idxs if y_pred_open[i] != "Unknown") / n_unknown if n_unknown > 0 else 0.0
    )
    unknown_rejection = 1.0 - far if n_unknown > 0 else 0.0

    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)

    far_curve, frr_curve, dir_curve = [], [], []
    for t in thresholds:
        far_t = (
            sum(1 for i in unknown_idxs if similarities[i] >= t) / n_unknown if n_unknown > 0 else 0.0
        )
        frr_t = (
            sum(1 for i in known_idxs if similarities[i] < t) / n_known if n_known > 0 else 0.0
        )
        dir_t = (
            sum(1 for i in known_idxs if similarities[i] >= t and y_pred_closed[i] == y_true[i]) / n_known
            if n_known > 0
            else 0.0
        )
        far_curve.append(far_t)
        frr_curve.append(frr_t)
        dir_curve.append(dir_t)

    return {
        "n_known": n_known,
        "n_unknown": n_unknown,
        "closed_set_rank1": closed_rank1,
        "open_set_rank1": open_rank1,
        "frr": frr,
        "false_id_rate": false_id_rate,
        "far": far,
        "unknown_rejection_rate": unknown_rejection,
        "thresholds": np.asarray(thresholds),
        "far_curve": np.asarray(far_curve),
        "frr_curve": np.asarray(frr_curve),
        "dir_curve": np.asarray(dir_curve),
    }