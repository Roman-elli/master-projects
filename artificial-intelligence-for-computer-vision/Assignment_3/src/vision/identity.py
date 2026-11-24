import numpy as np  # garante que numpy está importado no topo do evaluation.py


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

    # Ponto aproximado de Equal Error Rate (EER)
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
    - Closed-Set Rank-1 (sem limiar, apenas nos registados)
    - Open-Set Rank-1 / DIR (com limiar, apenas nos registados)
    - FRR (registados rejeitados como Unknown) e False ID Rate (registados trocados por outra pessoa)
    - FAR (desconhecidos aceites como alguém da galeria) e Unknown Rejection Rate (1 - FAR)
    - Curvas de sensibilidade ao limiar (FAR, FRR, Open-Set Rank-1)
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

    # Curvas de sensibilidade no conjunto de teste
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