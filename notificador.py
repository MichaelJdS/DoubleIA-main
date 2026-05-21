"""
BLAZE DOUBLE AI — NOTIFICADOR TELEGRAM v4.0 (Leviathan)
Exibe: regime, votos do ensemble, threshold adaptativo, auto-mute.
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

COR_NOME  = {0: "BRANCO",   1: "VERMELHO",  2: "PRETO",  None: "—"}
COR_EMOJI = {0: "⚪",       1: "🔴",        2: "⚫",     None: "❓"}

ACTION_PT = {
    "enter":             "🎯 ENTRAR",
    "enter_conditional": "⚡ ENTRAR CONDICIONAL",
    "wait":              "⏳ AGUARDAR",
    "block":             "🚫 BLOQUEADO",
    "monitor_white":     "🔍 MONITORAR BRANCA",
    "gale_1":            "🔥 GALE 1",
    "gale_2":            "🔥🔥 GALE 2",
}

def configured() -> bool:
    return bool(TELEGRAM_TOKEN and TELEGRAM_CHATID)

def escape_html(texto: str) -> str:
    if texto is None:
        return ""
    return str(texto).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

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

def notificar_sinal(signal: dict, regime: dict, patterns: list, probs: dict) -> bool:
    """
    Monta e envia notificação rica do Leviathan Engine.
    Exibe: regime, ensemble votes, threshold, kelly.
    """
    action = signal.get("action", "wait")
    color  = signal.get("color")
    conf   = float(signal.get("confidence", 0) or 0)
    reason = escape_html(signal.get("reason", ""))
    kelly  = float(signal.get("kelly", 0) or 0)

    if action not in ("enter", "enter_conditional", "monitor_white", "gale_1", "gale_2"):
        return False

    emoji     = COR_EMOJI.get(color, "❓")
    nome      = COR_NOME.get(color, "—")
    act_pt    = ACTION_PT.get(action, action)

    p_red   = round(float(probs.get("red",   0) or 0) * 100, 1)
    p_black = round(float(probs.get("black", 0) or 0) * 100, 1)
    p_white = round(float(probs.get("white", 0) or 0) * 100, 1)

    # Regime
    regime_name  = escape_html(regime.get("label") or regime.get("name") or "Normal")
    regime_str   = float(regime.get("strength", 0) or 0)

    # Votos do ensemble (novo no Leviathan)
    votes_json = signal.get("votes_json", [])
    votes_str = ""
    if votes_json:
        try:
            votes = (
                json.loads(votes_json)
                if isinstance(votes_json, str)
                else votes_json
            )
            active = [v for v in votes if isinstance(v, dict) and v.get("vote") is not None]
            if active:
                votes_str = " | ".join(
                    f"{'🔴' if v['vote']==1 else '⚫'} {escape_html(v['label'][:25])}"
                    for v in active[:3]
                )
        except Exception:
            pass

    # Threshold
    threshold_used = float(signal.get("threshold_used", 0) or 0)
    threshold_str  = f"{threshold_used*100:.1f}%" if threshold_used else ""

    timestamp = datetime.now().strftime("%d/%m %H:%M:%S")

    msg = (
        f"<b>⚡ LEVIATHAN ENGINE — SINAL</b>\n"
        f"<code>{timestamp}</code>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>{act_pt} {emoji} {nome}</b>\n"
        f"📊 Confiança: <b>{conf*100:.1f}%</b>"
        + (f" | Threshold: <b>{threshold_str}</b>" if threshold_str else "") + "\n"
        f"📈 Regime: <b>{regime_name}</b> ({regime_str*100:.0f}%)\n"
        + (f"🗳️ Ensemble:\n<code>{votes_str}</code>\n" if votes_str else "")
        + (f"💰 Kelly Recomendado: <b>{kelly:.2f}%</b>\n" if kelly > 0 else "")
        + f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🎲 Probs — 🔴{p_red}% ⚫{p_black}% ⚪{p_white}%\n"
        f"📝 <i>{reason[:120]}</i>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<i>Leviathan v1.0 | Auto-Adaptativo</i>"
    )
    return enviar(msg)

def notificar_auto_mute(losses: int, rounds: int):
    return enviar(
        f"<b>🔇 AUTO-MUTE ATIVADO</b>\n"
        f"{losses} derrotas consecutivas detectadas.\n"
        f"Sistema em silêncio por <b>{rounds} rounds</b>.\n"
        f"<i>Leviathan recalibrando threshold...</i>"
    )

def notificar_mute_encerrado(wr_recente: str):
    return enviar(
        f"<b>🔊 AUTO-MUTE ENCERRADO</b>\n"
        f"Performance recuperada: <b>{wr_recente}</b>\n"
        f"<i>Leviathan voltando a operar normalmente.</i>"
    )

def notificar_inicio():
    return enviar(
        "<b>⚡ Leviathan Engine v1.0 ATIVO</b>\n"
        "Ensemble Voting | Regime Detection | Adaptive Threshold\n"
        "Aguardando sinais operacionais..."
    )

def notificar_parada(total: int):
    return enviar(f"<b>Sistema encerrado</b>\nTotal coletado: <b>{int(total)}</b> rodadas.")

def notificar_status(texto: str):
    return enviar(f"<b>Status do Sistema</b>\n{escape_html(texto)}")

if __name__ == "__main__":
    print("TELEGRAM_TOKEN:", "OK" if TELEGRAM_TOKEN else "NÃO CONFIGURADO")
    print("TELEGRAM_CHATID:", "OK" if TELEGRAM_CHATID else "NÃO CONFIGURADO")