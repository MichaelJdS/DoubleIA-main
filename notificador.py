"""
BLAZE DOUBLE AI — NOTIFICADOR TELEGRAM v5.0 (Pantheon)
Exibe: micro-regime, 14 experts, D-S conflict, banca_level, Oracle.
"""
import json
import urllib.request
import urllib.parse
import logging
import os
from datetime import datetime

log = logging.getLogger("notificador")

TELEGRAM_TOKEN  = os.getenv("TELEGRAM_TOKEN", "").strip()
TELEGRAM_CHATID = os.getenv("TELEGRAM_CHATID", "").strip()

COR_NOME  = {0: "BRANCO",  1: "VERMELHO", 2: "PRETO",  None: "—"}
COR_EMOJI = {0: "⚪",      1: "🔴",       2: "⚫",     None: "❓"}

ACTION_PT = {
    "enter":         "🎯 ENTRAR",
    "wait":          "⏳ AGUARDAR",
    "block":         "🚫 BLOQUEADO",
    "freeze":        "❄️ FREEZE",
    "gale_1":        "🔥 GALE 1",
    "gale_2":        "🔥🔥 GALE 2",
}

def configured() -> bool:
    return bool(TELEGRAM_TOKEN and TELEGRAM_CHATID)

def escape_html(texto: str) -> str:
    if texto is None:
        return ""
    return str(texto).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

def enviar(texto: str) -> bool:
    if not configured():
        log.info("Telegram não configurado.")
        return False
    try:
        url  = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = urllib.parse.urlencode({
            "chat_id":                  TELEGRAM_CHATID,
            "text":                     texto,
            "parse_mode":               "HTML",
            "disable_web_page_preview": True,
        }).encode()
        req = urllib.request.Request(url, data=data, method="POST")
        with urllib.request.urlopen(req, timeout=8) as resp:
            payload = json.loads(resp.read())
            ok = payload.get("ok", False)
            if not ok:
                log.warning("Telegram falhou: %s", payload)
            return ok
    except Exception as e:
        log.warning("Telegram erro: %s", e)
        return False

def notificar_sinal(signal: dict, regime: dict, patterns: list, probs: dict,
                    features: dict = None) -> bool:
    """
    Notificação completa do Pantheon Engine v3.0.
    Exibe: micro-regime, D-S conflict, vote_count, banca_level, Kelly.
    """
    action = signal.get("action", "wait")
    color  = signal.get("color")
    conf   = float(signal.get("confidence", 0) or 0)
    reason = escape_html(signal.get("reason", ""))
    kelly  = float(signal.get("kelly", 0) or 0)

    if action not in ("enter", "gale_1", "gale_2", "monitor_white"):
        return False

    feat = features or {}

    emoji  = COR_EMOJI.get(color, "❓")
    nome   = COR_NOME.get(color, "—")
    act_pt = ACTION_PT.get(action, action)

    p_red   = round(float(probs.get("red",   0) or 0) * 100, 1)
    p_black = round(float(probs.get("black", 0) or 0) * 100, 1)
    p_white = round(float(probs.get("white", 0) or 0) * 100, 1)

    # Regime + micro
    regime_name = escape_html(regime.get("label") or regime.get("name") or "Normal")
    micro       = escape_html(feat.get("micro_regime") or
                              regime.get("micro_regime") or "–")

    # D-S
    ds_k        = float(feat.get("ds_conflict", 0) or 0)
    ds_str      = f"{ds_k:.3f}" + (" ⚠️" if ds_k > 0.35 else " ✅")

    # Experts
    vote_count  = int(signal.get("vote_count", feat.get("vote_count", 0)) or 0)
    votes_json  = signal.get("votes_json") or feat.get("votes_json", "[]")
    votes_str   = ""
    try:
        votes  = json.loads(votes_json) if isinstance(votes_json, str) else votes_json
        active = [v for v in votes if isinstance(v, dict) and v.get("vote") is not None]
        if active:
            votes_str = " ".join(
                f"{'🔴' if v['vote']==1 else '⚫'}"
                for v in active[:14]
            )
    except Exception:
        pass

    # Threshold + banca
    threshold  = float(signal.get("threshold_used", 0) or 0)
    banca      = feat.get("banca_level", "NORMAL")
    banca_icon = {"NORMAL": "🟢", "ALERT": "🟡", "LOCKDOWN": "🔴"}.get(banca, "⚪")

    timestamp = datetime.now().strftime("%d/%m %H:%M:%S")

    msg = (
        f"<b>⚡ PANTHEON v3.0 — SINAL</b>\n"
        f"<code>{timestamp}</code>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>{act_pt} {emoji} {nome}</b>\n"
        f"📊 Edge: <b>{conf*100:.1f}%</b>"
        + (f" | Thr: <b>{threshold*100:.1f}%</b>" if threshold else "") + "\n"
        f"🌀 Micro-regime: <code>{micro}</code>\n"
        f"⚛️ D-S Conflito: <b>{ds_str}</b>\n"
        f"🗳️ Experts ({vote_count}/14): <code>{votes_str}</code>\n"
        + (f"💰 Kelly: <b>{kelly:.2f}%</b>\n" if kelly > 0 else "")
        + f"{banca_icon} Banca: <b>{banca}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🎲 🔴{p_red}% ⚫{p_black}% ⚪{p_white}%\n"
        f"📝 <i>{reason[:140]}</i>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<i>Pantheon v3.0 | 14 Experts | D-S Fusion | Oracle RL</i>"
    )
    return enviar(msg)


def notificar_auto_mute(losses: int, rounds: int, banca_level: str = "ALERT"):
    banca_icon = {"NORMAL": "🟢", "ALERT": "🟡", "LOCKDOWN": "🔴"}.get(banca_level, "⚪")
    return enviar(
        f"<b>🔇 BANCA PROTECTION ATIVADO</b>\n"
        f"{banca_icon} Nível: <b>{banca_level}</b>\n"
        f"Derrotas consecutivas: <b>{losses}</b>\n"
        f"Silêncio por <b>{rounds} rounds</b>.\n"
        f"<i>Pantheon recalibrando threshold e Oracle...</i>"
    )

def notificar_mute_encerrado(wr_recente: str):
    return enviar(
        f"<b>🔊 PROTEÇÃO ENCERRADA</b>\n"
        f"Performance recuperada: <b>{wr_recente}</b>\n"
        f"<i>Pantheon voltando a operar normalmente.</i>"
    )

def notificar_regime_shift(old_regime: str, new_regime: str, freeze_rounds: int):
    return enviar(
        f"<b>❄️ REGIME SHIFT DETECTADO</b>\n"
        f"<code>{escape_html(old_regime)}</code> → <code>{escape_html(new_regime)}</code>\n"
        f"HERMES/BOCPD detectou mudança estrutural.\n"
        f"Freeze de <b>{freeze_rounds} rounds</b> ativado.\n"
        f"<i>Aguarde novo ciclo estável...</i>"
    )

def notificar_inicio():
    return enviar(
        "<b>⚡ PANTHEON ENGINE v3.0 ATIVO</b>\n"
        "14 Experts | Dempster-Shafer Fusion | Oracle Q-Learning\n"
        "SYBIL • CHAOS • HERMES • ATLAS • TITAN\n"
        "18 Micro-Regimes | Correlation Guard | Regime-Shift Freeze\n"
        "Aguardando sinais operacionais..."
    )

def notificar_parada(total: int):
    return enviar(
        f"<b>⚡ Pantheon v3.0 encerrado</b>\n"
        f"Total coletado: <b>{int(total)}</b> rodadas."
    )

def notificar_status(texto: str):
    return enviar(f"<b>📡 Status Pantheon</b>\n{escape_html(texto)}")

if __name__ == "__main__":
    print("TELEGRAM_TOKEN:",  "OK" if TELEGRAM_TOKEN  else "NÃO CONFIGURADO")
    print("TELEGRAM_CHATID:", "OK" if TELEGRAM_CHATID else "NÃO CONFIGURADO")