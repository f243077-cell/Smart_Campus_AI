"""
ui.py
-----------------------------------------------------------------------
Smart Campus AI - "Campus Line / Forest & Beige" look for the Streamlit UI.

The five request types are drawn as five transit lines and the AI
modules as the stations they stop at. This file only builds HTML/SVG
strings and the page CSS; app.py decides what to show and every result
still comes from the shared AI modules.
-----------------------------------------------------------------------
"""

import html
import math

import numpy as np

import config

# -------------------------------------------------------------------
# Design tokens
# -------------------------------------------------------------------
LIGHT_TOKENS = {
    "bg": "#F6F1E6", "panel": "#FFFCF6", "panel-2": "#F2EBDC", "ink": "#17251D", "ink-2": "#4A5549",
    "ink-3": "#7D867B", "rule": "#DED2BA", "rule-2": "#EBE3D2", "brand": "#1F4D36", "brand-2": "#2D6A4B",
    "brand-ink": "#F6F1E6", "side-bg": "#E3D6BA", "side-ink": "#17251D", "side-ink-2": "#45503F",
    "accent": "#A8406C", "ok": "#1D6B42", "bad": "#A8321F", "warn": "#7A5C0E", "ok-bg": "#DDEBDD",
    "bad-bg": "#F5DCD3", "warn-bg": "#F1E6C6", "l-nav": "#1B7A84", "l-elig": "#2E5A99", "l-book": "#6A4B9E",
    "l-urg": "#B23A2A", "l-full": "#1F4D36", "s0": "#1B7A84", "s1": "#2E5A99", "s2": "#6A4B9E", "s3": "#A8406C",
    "shadow": "0 1px 0 rgba(31,77,54,.06),0 12px 28px -18px rgba(31,77,54,.35)", "focus": "#A8406C",
    # CSP constraint colours keep the meaning of the project's CSP diagram
    "c-ne": "#C0392B", "c-exam": "#7B3294", "c-sup": "#C25A00", "c-prec": "#1F4E9C",
}
# Dark: the same forest line on a night ground (lighter line colours keep 4.5:1 contrast).
DARK_TOKENS = {
    "bg": "#101813", "panel": "#18221C", "panel-2": "#1F2B24", "ink": "#ECE6D8", "ink-2": "#B9BCAF",
    "ink-3": "#8C9489", "rule": "#34433A", "rule-2": "#27342C", "brand": "#86C79F", "brand-2": "#A2D6B5",
    "brand-ink": "#0F1A13", "side-bg": "#151E18", "side-ink": "#ECE6D8", "side-ink-2": "#B9BCAF",
    "accent": "#E58DB0", "ok": "#7BD3A0", "bad": "#F29580", "warn": "#E6C46A", "ok-bg": "#1D3727",
    "bad-bg": "#3F221B", "warn-bg": "#3A321A", "l-nav": "#5BBCC5", "l-elig": "#86AAE8", "l-book": "#B9A0E6",
    "l-urg": "#F29580", "l-full": "#86C79F", "s0": "#5BBCC5", "s1": "#86AAE8", "s2": "#B9A0E6", "s3": "#E58DB0",
    "shadow": "0 1px 0 rgba(0,0,0,.25),0 12px 28px -18px rgba(0,0,0,.8)", "focus": "#E58DB0",
    "c-ne": "#FF7B72", "c-exam": "#D2A8FF", "c-sup": "#FFA657", "c-prec": "#79C0FF",
}

# Request type -> line name and line colour variable
LINES = {
    "Navigation_Only": {"label": "Navigation", "var": "var(--l-nav)"},
    "Eligibility_Check": {"label": "Eligibility", "var": "var(--l-elig)"},
    "Booking_or_Scheduling": {"label": "Booking", "var": "var(--l-book)"},
    "Urgent_Service_Request": {"label": "Urgent service", "var": "var(--l-urg)"},
    "Full_Service_Request": {"label": "Full service", "var": "var(--l-full)"},
}
STATION_NAMES = {"Preprocessing": "Input check", "Router": "Router", "ANN": "Priority (ANN)",
                 "Logic": "Logic / KB", "CSP": "Scheduler (CSP)", "Search": "Search",
                 "Response": "Final response"}
STATUS_TONES = {"COMPLETED": "ok", "APPROVED": "ok", "DISPATCHED": "ok", "REJECTED": "bad", "INVALID": "bad",
                "ERROR": "bad", "UNSCHEDULABLE": "warn"}
TONE_VARS = {"ok": "var(--ok)", "bad": "var(--bad)", "warn": "var(--warn)"}
PRIORITY_CLASS = {"Urgent": "p-urgent", "High": "p-high", "Normal": "p-normal", "Low": "p-low"}
FIXED_BOOKINGS = ("AI Lab class (DrKhan)", "Reading group", "Admin meeting")

ICON_PLAY = '<svg viewBox="0 0 20 20" aria-hidden="true" style="fill:currentColor"><path d="M6 4l10 6-10 6z"/></svg>'


def esc(value):
    """
    Function: esc
    Purpose : HTML-escape any value for safe display.
    """
    return html.escape("" if value is None else str(value))


def nice(name):
    """
    Function: nice
    Purpose : Show identifiers like Main_Gate as 'Main Gate'.
    """
    return esc(str(name).replace("_", " "))


def line_of(request_type):
    """
    Function: line_of
    Purpose : Line info for a request type (grey line for unknown types).
    """
    return LINES.get(request_type, {"label": request_type or "Unknown", "var": "var(--ink-3)"})


# -------------------------------------------------------------------
# Page CSS
# -------------------------------------------------------------------
_CSS = r"""
:root{__TOKENS__}
html,body,.stApp,[data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--ink)}
.stApp{font-family:'Poppins',system-ui,sans-serif;font-variant-numeric:tabular-nums}
[data-testid="stHeader"]{background:transparent}
[data-testid="stMainBlockContainer"],.block-container{padding-top:2.2rem;padding-bottom:4rem;max-width:1360px}
.stApp ::selection{background:var(--brand);color:var(--brand-ink)}
h1,h2,h3{font-family:'Poppins',system-ui,sans-serif!important}

/* ---------- sidebar: the nav is a vertical line of stations ---------- */
section[data-testid="stSidebar"]{background:var(--side-bg)!important;border-right:1px solid var(--rule)}
section[data-testid="stSidebar"] > div{background:var(--side-bg)}
.brand{display:flex;align-items:center;gap:12px;padding:0 6px 6px}
.brand-mark{width:38px;height:38px;border-radius:50%;border:6px solid var(--brand);position:relative;flex:none;animation:spin 14s linear infinite;box-sizing:border-box}
.brand-mark::after{content:'';position:absolute;left:-10px;right:-10px;top:50%;height:6px;margin-top:-3px;background:var(--accent);border-radius:3px}
.brand b{display:block;font-size:17px;font-weight:700;letter-spacing:-.02em;line-height:1.15;color:var(--side-ink)}
.brand span{font-size:12px;color:var(--side-ink-2)}
.st-key-page [data-testid="stWidgetLabel"]{display:none}
.st-key-page{width:100%!important}
.st-key-page [role="radiogroup"] > div{width:100%}
.st-key-page [role="radiogroup"]{position:relative;display:flex;flex-direction:column;gap:2px}
.st-key-page [role="radiogroup"]::before{content:'';position:absolute;left:24px;top:22px;bottom:22px;width:4px;border-radius:2px;background:color-mix(in srgb,var(--brand) 22%,transparent)}
.st-key-page label[data-testid="stRadioOption"]{position:relative;display:flex;align-items:center;min-height:44px;margin:0;padding:0 12px 0 48px;border-radius:12px;cursor:pointer;color:var(--side-ink-2);transition:background .2s ease,color .2s ease;animation:slideIn .4s cubic-bezier(.2,.8,.2,1) both}
.st-key-page label[data-testid="stRadioOption"] > div > div:first-child{display:none}
.st-key-page label[data-testid="stRadioOption"]::before{content:'';position:absolute;left:17px;top:50%;width:18px;height:18px;margin-top:-9px;border-radius:50%;background:var(--panel);border:4px solid var(--ink-3);box-sizing:border-box;transition:transform .25s cubic-bezier(.3,1.6,.5,1),border-color .2s,background .2s;z-index:1}
.st-key-page label[data-testid="stRadioOption"] p{font-size:14px;font-weight:500;color:inherit;transition:transform .25s cubic-bezier(.2,.8,.2,1)}
.st-key-page label[data-testid="stRadioOption"]:hover{background:color-mix(in srgb,var(--brand) 7%,transparent);color:var(--side-ink)}
.st-key-page label[data-testid="stRadioOption"]:hover::before{transform:scale(1.25);border-color:var(--brand)}
.st-key-page label[data-testid="stRadioOption"]:hover p{transform:translateX(4px)}
.st-key-page label[data-testid="stRadioOption"][data-selected="true"]{color:var(--brand);background:var(--panel);box-shadow:var(--shadow)}
.st-key-page label[data-testid="stRadioOption"][data-selected="true"] p{font-weight:600}
.st-key-page label[data-testid="stRadioOption"][data-selected="true"]::before{background:var(--brand);border-color:var(--brand);animation:ping 2s ease-out infinite}
.side-h{font-size:12px;font-weight:600;color:var(--side-ink-2);padding:0 6px;margin:6px 0 2px}
section[data-testid="stSidebar"] [class*="st-key-sidebar_"] button{justify-content:flex-start;text-align:left;min-height:44px;border-radius:12px;border:1px solid var(--rule);background:color-mix(in srgb,var(--panel) 55%,transparent);color:var(--side-ink);transition:transform .2s ease,background .2s ease,border-color .2s ease}
section[data-testid="stSidebar"] [class*="st-key-sidebar_"] button:hover{transform:translateX(3px);background:var(--panel);border-color:color-mix(in srgb,var(--brand) 40%,var(--rule));color:var(--side-ink)}
section[data-testid="stSidebar"] [class*="st-key-sidebar_"] button p{font-size:12.5px;font-weight:600}
.side-note{font-size:12px;color:var(--side-ink-2);padding:0 6px}

/* ---------- buttons, inputs, tabs ---------- */
.stButton button,.stDownloadButton button,.stFormSubmitButton button{min-height:44px;border-radius:999px!important;padding:0 18px;font-weight:600;border:1.5px solid var(--brand);background:transparent;color:var(--brand);transition:background .2s ease,color .2s ease,transform .12s ease,box-shadow .2s ease}
.stButton button p,.stDownloadButton button p,.stFormSubmitButton button p{font-weight:600;font-size:14px}
.stButton button:hover,.stDownloadButton button:hover,.stFormSubmitButton button:hover{background:var(--brand);color:var(--brand-ink);border-color:var(--brand);box-shadow:0 8px 18px -10px rgba(31,77,54,.7)}
.stButton button:active{transform:scale(.97)}
button[data-testid="stBaseButton-primary"],button[data-testid="stBaseButton-primaryFormSubmit"]{background:var(--brand)!important;color:var(--brand-ink)!important;border-color:var(--brand)!important}
button[data-testid="stBaseButton-primary"]:hover,button[data-testid="stBaseButton-primaryFormSubmit"]:hover{background:var(--brand-2)!important;border-color:var(--brand-2)!important}
.st-key-qchips .stButton button{min-height:40px;border-color:var(--rule);color:var(--ink);background:var(--panel);font-weight:500}
.st-key-qchips .stButton button p{font-weight:500;font-size:12.5px;white-space:nowrap;overflow:visible}
.st-key-qchips .stButton button{padding:0 10px}
.st-key-qchips .stButton button:hover{border-color:var(--brand);background:var(--brand);color:var(--brand-ink);transform:translateY(-2px)}
[data-baseweb="select"] > div,[data-baseweb="input"],[data-baseweb="base-input"],.stNumberInput [data-baseweb="input"],.stTextInput [data-baseweb="input"]{border-radius:12px!important;background:var(--panel)!important;border-color:var(--rule)!important;min-height:44px}
[data-baseweb="select"] > div:hover,[data-baseweb="input"]:hover{border-color:var(--ink-3)!important}
[data-baseweb="select"] > div:focus-within,[data-baseweb="input"]:focus-within{border-color:var(--brand)!important;box-shadow:0 0 0 4px color-mix(in srgb,var(--brand) 16%,transparent)}
[data-testid="stWidgetLabel"] p{font-size:13px;font-weight:600;color:var(--ink)}
.stSlider [role="slider"]{background:var(--brand)!important;box-shadow:0 0 0 4px color-mix(in srgb,var(--brand) 18%,transparent)!important}
[data-testid="stSliderThumbValue"]{color:var(--brand)!important;font-weight:700;font-family:'Poppins',sans-serif}
[data-testid="stForm"]{border:1px solid var(--rule-2)!important;border-radius:16px;background:var(--panel);padding:22px;box-shadow:var(--shadow)}
.stTabs [role="tablist"]{gap:6px;flex-wrap:wrap;margin-bottom:6px;border:none!important;box-shadow:none!important}
.stTabs [role="tablist"]::after,.stTabs [role="tablist"]::before{display:none}
.stTabs .react-aria-SelectionIndicator{display:none!important}
.stTabs [data-testid="stTab"]{min-height:40px;height:auto;padding:0 16px;border-radius:999px;border:1.5px solid var(--rule);background:transparent;transition:all .2s;display:flex;align-items:center;cursor:pointer}
.stTabs [data-testid="stTab"] p{font-size:13.5px;font-weight:600;color:var(--ink-2)}
.stTabs [data-testid="stTab"]:hover{border-color:var(--brand)}
.stTabs [data-testid="stTab"]:hover p{color:var(--brand)}
.stTabs [data-testid="stTab"][aria-selected="true"]{background:var(--brand);border-color:var(--brand)}
.stTabs [data-testid="stTab"][aria-selected="true"] p{color:var(--brand-ink)}
[data-testid="stExpander"] details{border-radius:14px;border-color:var(--rule-2);background:var(--panel)}
[data-testid="stFileUploaderDropzone"]{border:2px dashed var(--rule)!important;border-radius:16px;background:var(--panel-2)!important;transition:border-color .2s,background .2s}
[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--brand)!important;background:color-mix(in srgb,var(--brand) 5%,var(--panel-2))!important}
[data-testid="stDataFrame"]{border:1px solid var(--rule-2);border-radius:12px;overflow:hidden}
[data-testid="stAlert"]{border-radius:14px}
.stProgress > div > div > div > div{background:repeating-linear-gradient(45deg,var(--brand) 0 12px,var(--brand-2) 12px 24px)!important}

/* segmented controls and pills picked up as line cards / chips */
[data-testid="stButtonGroup"] button{min-height:38px;padding:0 12px;border-radius:999px!important;border:1px solid var(--rule-2);background:var(--panel-2);color:var(--ink-2);font-weight:500}
[data-testid="stButtonGroup"] button[kind$="Active"],[data-testid="stButtonGroup"] button[aria-checked="true"],[data-testid="stButtonGroup"] button[aria-pressed="true"]{background:var(--brand)!important;color:var(--brand-ink)!important;border-color:var(--brand)!important;font-weight:600}
[data-testid="stButtonGroup"] button:hover{color:var(--ink);border-color:var(--brand)}

/* request type / demo pickers: radio styled as line cards */
.st-key-req_type,.st-key-demo_pick{width:100%!important}
.st-key-req_type [role="radiogroup"] > div,.st-key-demo_pick [role="radiogroup"] > div{display:flex}
.st-key-req_type label[data-testid="stRadioOption"],.st-key-demo_pick label[data-testid="stRadioOption"]{width:100%}
.st-key-req_type [role="radiogroup"],.st-key-demo_pick [role="radiogroup"]{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.st-key-demo_pick [role="radiogroup"]{grid-template-columns:repeat(auto-fill,minmax(230px,1fr))}
.st-key-req_type label[data-testid="stRadioOption"],.st-key-demo_pick label[data-testid="stRadioOption"]{position:relative;display:flex;flex-direction:column;align-items:flex-start;gap:12px;margin:0;min-height:44px;padding:16px;border-radius:14px;border:1.5px solid var(--rule);background:var(--panel);cursor:pointer;transition:transform .2s cubic-bezier(.2,.8,.2,1),box-shadow .2s,border-color .2s;overflow:hidden}
.st-key-req_type label[data-testid="stRadioOption"] > div > div:first-child,.st-key-demo_pick label[data-testid="stRadioOption"] > div > div:first-child{display:none}
.st-key-req_type label[data-testid="stRadioOption"] p{font-size:14.5px;font-weight:600;color:var(--ink)}
.st-key-req_type label[data-testid="stRadioOption"]::after{content:'';display:block;width:100%;height:16px;background:radial-gradient(circle,var(--panel) 0 3px,var(--lc) 3.2px 6px,transparent 6.4px) 0 0/25% 16px repeat-x,linear-gradient(var(--lc),var(--lc)) 0 6px/100% 4px no-repeat;border-radius:2px}
.st-key-req_type label[data-testid="stRadioOption"]:hover,.st-key-demo_pick label[data-testid="stRadioOption"]:hover{transform:translateY(-3px);box-shadow:var(--shadow);border-color:var(--lc)}
.st-key-req_type label[data-testid="stRadioOption"][data-selected="true"],.st-key-demo_pick label[data-testid="stRadioOption"][data-selected="true"]{border-color:var(--lc);box-shadow:0 0 0 3px color-mix(in srgb,var(--lc) 22%,transparent)}
.st-key-req_type [role="radiogroup"] > div:nth-child(1) label,.st-key-demo_pick [role="radiogroup"] > div:nth-child(1) label,.st-key-demo_pick [role="radiogroup"] > div:nth-child(2) label{--lc:var(--l-nav)}
.st-key-req_type [role="radiogroup"] > div:nth-child(2) label,.st-key-demo_pick [role="radiogroup"] > div:nth-child(3) label,.st-key-demo_pick [role="radiogroup"] > div:nth-child(4) label{--lc:var(--l-elig)}
.st-key-req_type [role="radiogroup"] > div:nth-child(3) label,.st-key-demo_pick [role="radiogroup"] > div:nth-child(5) label,.st-key-demo_pick [role="radiogroup"] > div:nth-child(6) label{--lc:var(--l-book)}
.st-key-req_type [role="radiogroup"] > div:nth-child(4) label,.st-key-demo_pick [role="radiogroup"] > div:nth-child(7) label{--lc:var(--l-urg)}
.st-key-req_type [role="radiogroup"] > div:nth-child(5) label,.st-key-demo_pick [role="radiogroup"] > div:nth-child(8) label{--lc:var(--l-full)}
.st-key-demo_pick [role="radiogroup"] > div:nth-child(9) label{--lc:var(--ink-3)}
.st-key-demo_pick label[data-testid="stRadioOption"]{padding:16px 16px 16px 22px;min-height:84px}
.st-key-demo_pick label[data-testid="stRadioOption"]::before{content:'';position:absolute;left:0;top:0;bottom:0;width:6px;background:var(--lc);transform:scaleY(.35);transform-origin:top;transition:transform .3s cubic-bezier(.2,.8,.2,1)}
.st-key-demo_pick label[data-testid="stRadioOption"]:hover::before,.st-key-demo_pick label[data-testid="stRadioOption"][data-selected="true"]::before{transform:scaleY(1)}
.st-key-demo_pick label[data-testid="stRadioOption"] p{font-size:14px;font-weight:600;line-height:1.35;color:var(--ink)}
.st-key-req_type label[data-testid="stRadioOption"] p:last-child{font-size:12px;font-weight:500;color:var(--ink-2)}
.st-key-demo_pick label[data-testid="stRadioOption"] p:first-child{font-size:12px;font-weight:500;color:var(--ink-2)}
.st-key-req_type [data-testid="stWidgetLabel"],.st-key-demo_pick [data-testid="stWidgetLabel"]{display:none}

/* ---------- shared surfaces ---------- */
.sc *{box-sizing:border-box}
.sc{font-family:'Poppins',system-ui,sans-serif;color:var(--ink);font-size:15px;line-height:1.5}
.sc h1,.sc h2,.sc h3,.sc p{margin:0;padding:0}
.sc h2{font-size:18px;font-weight:600;letter-spacing:-.01em;color:var(--ink)}
.sc h3{font-size:15px;font-weight:600;color:var(--ink)}
.muted{color:var(--ink-2)}.small{font-size:13px}
.top h1{font-size:30px;font-weight:700;letter-spacing:-.025em;line-height:1.15;color:var(--ink)}
.top p{max-width:62ch;color:var(--ink-2);margin-top:4px}
.panel{background:var(--panel);border:1px solid var(--rule-2);border-radius:16px;padding:22px;box-shadow:var(--shadow);animation:rise .45s cubic-bezier(.2,.8,.2,1) both}
.panel-head{display:flex;flex-wrap:wrap;gap:8px 16px;align-items:baseline;justify-content:space-between;margin-bottom:14px}
.row{display:flex;flex-wrap:wrap;gap:20px;align-items:stretch}
.grow{flex:1 1 320px;min-width:0}.grow3{flex:3 1 520px;min-width:0}.grow2{flex:2 1 460px;min-width:0}
.col{display:flex;flex-direction:column;gap:14px}
/* status and priority markers: a station symbol plus coloured text */
.pill{display:inline-flex;align-items:center;gap:8px;font-size:13px;font-weight:600;white-space:nowrap}
.pill::before{content:'';flex:none;width:13px;height:13px;border-radius:50%;border:3px solid currentColor;box-sizing:border-box}
.pill.tone-ok::before{background:radial-gradient(circle,currentColor 0 2px,transparent 2.6px)}
.pill.tone-bad::before{background:linear-gradient(135deg,transparent 38%,currentColor 38% 62%,transparent 62%)}
.pill.tone-warn::before{border-width:2.5px;border-style:dashed}
.tone-ok{color:var(--ok)}.tone-bad{color:var(--bad)}.tone-warn{color:var(--warn)}
.p-urgent{color:var(--bad)}.p-high{color:var(--accent)}.p-normal{color:var(--l-elig)}.p-low{color:var(--ink-2)}
.pill.p-urgent::before{width:14px;height:14px;background:currentColor}
.pill.p-high::before{width:12px;height:12px;background:radial-gradient(circle,currentColor 0 1.6px,transparent 2.2px)}
.pill.p-normal::before{width:10px;height:10px;border-width:2.5px}
.pill.p-low::before{width:8px;height:8px;border-width:2px}
/* home */
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:14px}
.tile{padding:18px 20px;border-radius:16px;background:var(--panel);border:1px solid var(--rule-2);box-shadow:var(--shadow);transition:transform .2s ease;animation:rise .5s both}
.tile:hover{transform:translateY(-3px)}
.tile .num{display:block;font-size:38px;font-weight:700;letter-spacing:-.03em;line-height:1;color:var(--ink)}
.tile.ok .num{color:var(--ok)}.tile.bad .num{color:var(--bad)}.tile.warn .num{color:var(--warn)}.tile.brand .num{color:var(--brand)}
.tile .lab{display:block;font-size:13px;color:var(--ink-2);margin-top:6px}
.stack{display:flex;height:18px;border-radius:9px;overflow:hidden;background:var(--panel-2);gap:2px}
.stack .sseg{display:block;height:100%;background:currentColor;animation:grow .9s cubic-bezier(.2,.8,.2,1) both;transform-origin:left}
.legend{display:flex;flex-wrap:wrap;gap:8px 16px;margin-top:12px;font-size:13px}
.legend span{display:inline-flex;align-items:center;gap:6px}
.legend i{width:10px;height:10px;border-radius:3px;background:currentColor}
.hbar{display:grid;grid-template-columns:120px minmax(0,1fr) 40px;gap:10px;align-items:center;min-height:30px;font-size:13.5px}
.hbar .track2{height:10px;border-radius:5px;background:var(--panel-2);overflow:hidden}
.hbar .fill{display:block;height:100%;border-radius:5px;animation:grow .9s cubic-bezier(.2,.8,.2,1) both;transform-origin:left}
.network .netline{display:grid;grid-template-columns:130px minmax(0,1fr);gap:12px;align-items:center;min-height:42px;padding:4px 8px;border-radius:10px;transition:opacity .25s,background .25s}
.network:hover .netline{opacity:.35}
.network .netline:hover{opacity:1;background:var(--panel-2)}
.network .netline b{font-size:13px;font-weight:600;display:flex;align-items:center;gap:8px}
.network .netline b i{width:18px;height:6px;border-radius:3px;background:var(--lc)}
.mini{position:relative;display:flex;justify-content:space-between;align-items:center;height:16px}
.mini::before{content:'';position:absolute;left:0;right:0;top:6px;height:4px;border-radius:2px;background:var(--lc)}
.mini i{position:relative;width:12px;height:12px;border-radius:50%;background:var(--panel);border:3px solid var(--lc)}
.netline:hover .mini i{animation:hop .45s cubic-bezier(.3,1.6,.5,1) both;animation-delay:var(--d)}
.mini-names{display:flex;justify-content:space-between;font-size:10.5px;color:var(--ink-2)}
.facts{display:flex;flex-wrap:wrap;gap:10px}
.fact{display:flex;align-items:baseline;gap:6px;padding:10px 16px;border-radius:999px;background:var(--panel-2);border:1px solid var(--rule-2)}
.fact b{font-size:20px;font-weight:700;color:var(--brand)}
.empty{display:flex;flex-direction:column;align-items:center;text-align:center;gap:12px;padding:46px 20px}
.empty .art{width:220px;height:70px;position:relative}
.empty .art i{position:absolute;left:0;right:0;height:5px;border-radius:3px;animation:grow 1.4s cubic-bezier(.2,.8,.2,1) both;transform-origin:left}
.empty h2{font-size:21px}
.empty p{max-width:44ch;color:var(--ink-2)}
/* tables */
.tablebox{overflow:auto;border-radius:12px;border:1px solid var(--rule-2)}
.tablebox.scroll{max-height:420px}
.sc table{width:100%;border-collapse:collapse;font-size:13.5px;margin:0;border:none}
.sc th{text-align:left;font-size:12px;font-weight:600;color:var(--ink-2);padding:10px 12px;background:var(--panel-2);white-space:nowrap;border:none;position:sticky;top:0}
.sc td{padding:10px 12px;border:none;border-top:1px solid var(--rule-2);vertical-align:middle;color:var(--ink)}
.sc tr.stream{animation:rowIn .45s cubic-bezier(.2,.8,.2,1) both}
.sc tr:hover td{background:var(--panel-2)}
.sc tr:hover td:first-child{box-shadow:inset 4px 0 0 var(--lc,var(--brand))}
.idcell{display:inline-flex;align-items:center;gap:8px;font-weight:600}
.idcell i{width:12px;height:12px;border-radius:50%;border:3px solid var(--lc);background:var(--panel)}
/* result: header, line, stations */
.result{display:flex;flex-direction:column;gap:18px}
.rhead{display:flex;flex-wrap:wrap;gap:12px 20px;align-items:center;justify-content:space-between}
.rhead h2{font-size:24px;font-weight:700;letter-spacing:-.02em}
.rbadge{display:inline-flex;align-items:center;gap:8px;padding:4px 12px 4px 6px;border-radius:999px;background:var(--panel-2);font-size:13px;font-weight:600}
.rbadge i{width:22px;height:8px;border-radius:4px;background:var(--line)}
.caption{font-size:14.5px;font-weight:600;min-height:22px;animation:fadeIn .5s both}
.trackbox{overflow-x:auto;margin:0 -8px;padding:0 8px}
.trackin{position:relative;height:150px;min-width:760px;margin:6px 70px 0}
.track{position:absolute;left:0;right:0;top:26px;height:10px;border-radius:5px;background:repeating-linear-gradient(90deg,var(--rule) 0 14px,transparent 14px 22px)}
.travel{position:absolute;left:0;top:26px;height:10px;border-radius:5px;background:var(--line);animation:travel var(--dur) cubic-bezier(.45,0,.2,1) both}
.train{position:absolute;top:13px;height:36px;padding:0 12px;transform:translateX(-50%);border-radius:18px;background:var(--ink);color:var(--panel);display:flex;align-items:center;font-size:11px;font-weight:700;white-space:nowrap;z-index:3;box-shadow:0 10px 18px -8px rgba(23,37,29,.7);animation:trainrun var(--dur) cubic-bezier(.45,0,.2,1) both}
.train::after{content:'';position:absolute;right:-5px;top:50%;width:10px;height:10px;margin-top:-5px;border-radius:50%;background:var(--accent);box-shadow:0 0 12px var(--accent)}
.stn{position:absolute;top:17px;width:28px;height:28px;margin-left:-14px;border-radius:50%;background:var(--panel);border:5px solid var(--rule);z-index:2;box-sizing:border-box;animation:stnOn .35s cubic-bezier(.3,1.6,.5,1) both}
.stn.ran{border-color:var(--line)}
.stn.failed{border-color:var(--bad);background:var(--bad);animation:stnFail .45s ease both}
.stn.skipped{border-style:dashed;opacity:.7;animation:none}
.stn.pending{animation:none}
.buffer{position:absolute;top:10px;width:10px;height:42px;margin-left:20px;border-radius:3px;background:var(--bad);z-index:2;animation:drop .45s cubic-bezier(.2,.9,.3,1.3) both}
.slabel{position:absolute;top:62px;width:150px;margin-left:-75px;text-align:center;display:flex;flex-direction:column;gap:2px;animation:rise .45s cubic-bezier(.2,.8,.2,1) both}
.slabel b{font-size:13.5px;font-weight:600}
.slabel span{font-size:12px;color:var(--ink-2);min-height:36px}
.slabel.pending{opacity:.45;animation:none}
.slabel.failed span{color:var(--bad);font-weight:600}
.slabel.skipped span{font-style:italic}
.status-sign{display:inline-flex;align-items:center;gap:12px;padding:10px 22px 10px 12px;border-radius:999px;background:currentColor;font-size:20px;font-weight:700;animation:arrive .6s cubic-bezier(.2,.9,.3,1.2) both}
.status-sign span{color:var(--panel)}
.status-sign i{width:26px;height:26px;border-radius:50%;border:5px solid var(--panel);box-sizing:border-box}
.keys{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}
.kv{padding:14px 16px;border-radius:14px;background:var(--panel-2);border-top:4px solid var(--line);animation:rise .5s both}
.kv.none{border-top-color:var(--rule)}
.kv small{display:block;font-size:12px;color:var(--ink-2)}
.kv b{display:block;font-size:19px;font-weight:700;line-height:1.3;word-break:break-word}
.kv.bad b{color:var(--bad)}.kv.none b{color:var(--ink-3)}
.pbar{display:grid;grid-template-columns:70px minmax(0,1fr) 48px;gap:10px;align-items:center;font-size:13.5px;margin-bottom:8px}
.pbar .t{height:12px;border-radius:6px;background:var(--panel-2);overflow:hidden}
.pbar .f{display:block;height:100%;border-radius:6px;background:var(--ink-3);animation:grow 1s cubic-bezier(.2,.8,.2,1) both;transform-origin:left}
.pbar.top .f{background:var(--line,var(--brand))}
.pbar.top span:first-child{font-weight:700}
.feats{display:grid;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));gap:8px}
.feat{padding:8px 10px;border-radius:10px;background:var(--panel-2);font-size:12px}
.feat b{display:block;font-size:14px}
.feat .t{display:block;height:4px;border-radius:2px;background:var(--rule);margin-top:4px;overflow:hidden}
.feat .t i{display:block;height:100%;background:var(--brand);animation:grow .8s both;transform-origin:left}
.check{padding:12px 14px;border-radius:12px;background:var(--panel-2);display:flex;flex-direction:column;gap:6px;margin-bottom:10px}
.check .ch{display:flex;justify-content:space-between;gap:10px;font-weight:600;font-size:14px}
.check.ok .v{color:var(--ok)}.check.no .v{color:var(--bad)}
.pstep{display:flex;gap:10px;align-items:baseline;font-size:13px;animation:rise .4s cubic-bezier(.2,.8,.2,1) both}
.pstep b{font-weight:600}
.pstep span{font-size:11.5px;color:var(--ink-2);padding:1px 8px;border-radius:999px;background:var(--panel);white-space:nowrap}
.pstep.isfact span{background:var(--ok-bg);color:var(--ok)}
.miss{font-size:13px;color:var(--bad);font-weight:500}
.chips{display:flex;flex-wrap:wrap;gap:6px}
.ochip{display:inline-flex;align-items:center;gap:6px;padding:4px 10px 4px 4px;border-radius:999px;background:var(--panel-2);font-size:12.5px;animation:pop2 .35s cubic-bezier(.3,1.6,.5,1) both}
.ochip i{display:inline-flex;align-items:center;justify-content:center;width:20px;height:20px;border-radius:50%;background:var(--brand);color:var(--brand-ink);font-size:10.5px;font-style:normal;font-weight:700}
.ochip.cur{background:var(--accent);color:#fff}
.expl{margin:0;padding-left:22px;max-width:78ch}.expl li+li{margin-top:6px}
.expl li{animation:rise .4s both}
/* map */
.map{width:100%;height:auto;display:block;--line:var(--brand)}
.map .medge{stroke:var(--rule);stroke-width:6;fill:none;stroke-linecap:round}
.map .mseg{stroke:var(--line);stroke-width:9;fill:none;stroke-linecap:round;stroke-dasharray:1;stroke-dashoffset:1;animation:draw .35s cubic-bezier(.4,0,.2,1) forwards}
.map .mw{font-size:11px;font-weight:600;fill:var(--ink-3)}
.map .mwbg{fill:var(--panel)}
.map .mnode{fill:var(--panel);stroke:var(--ink-2);stroke-width:4}
.map .mnode.ex{stroke:var(--line);fill:color-mix(in srgb,var(--line) 22%,var(--panel))}
.map .mnode.onpath{fill:var(--line);stroke:var(--ink)}
.map .mnode.cur{stroke:var(--accent);stroke-width:5}
.map .mnode.end{stroke:var(--ink)}
.map .ring{fill:none;stroke:var(--accent);stroke-width:3;transform-box:fill-box;transform-origin:center;animation:ringsvg 1.2s ease-out infinite}
.map .mlabel{font-size:12px;font-weight:500;fill:var(--ink-2)}
.map .mord{fill:var(--ink)}
.map .mordt{fill:var(--panel);font-size:10px;font-weight:700}
.map .pop{transform-box:fill-box;transform-origin:center;animation:popsvg .4s cubic-bezier(.3,1.6,.5,1) both}
.maplegend{display:flex;flex-wrap:wrap;gap:8px 16px;font-size:12.5px;color:var(--ink-2);margin-top:10px}
.maplegend span{display:inline-flex;align-items:center;gap:6px}
.maplegend i{width:14px;height:14px;border-radius:50%;border:3px solid var(--ink-2);background:var(--panel);box-sizing:border-box}
.maplegend i.ex{border-color:var(--line);background:color-mix(in srgb,var(--line) 22%,var(--panel))}
.maplegend i.cur{border-color:var(--accent)}
.maplegend i.p{width:22px;height:6px;border-radius:3px;border:none;background:var(--line)}
/* navigation */
.algocard{display:flex;flex-direction:column;gap:10px;padding:18px;border-radius:14px;background:var(--panel);border:1.5px solid var(--rule-2);animation:rise .4s both;transition:transform .2s,box-shadow .2s}
.algocard:hover{transform:translateY(-3px);box-shadow:var(--shadow)}
.algocard.best{border-color:var(--brand);box-shadow:0 0 0 3px color-mix(in srgb,var(--brand) 16%,transparent)}
.algocard .nums{display:flex;gap:18px}
.algocard .nums b{display:block;font-size:24px;font-weight:700;line-height:1.1}
.algocard .nums span{font-size:12px;color:var(--ink-2)}
.bestflag{font-size:11.5px;font-weight:700;padding:2px 10px;border-radius:999px;background:var(--brand);color:var(--brand-ink)}
.cmp{display:grid;grid-template-columns:60px minmax(0,1fr) 34px;gap:10px;align-items:center;font-size:13px;min-height:28px}
.cmp .t{height:10px;border-radius:5px;background:var(--panel-2);overflow:hidden}
.cmp .t i{display:block;height:100%;border-radius:5px;background:var(--brand);animation:grow .6s cubic-bezier(.2,.8,.2,1) both;transform-origin:left}
.cmp .t i.c{background:var(--accent)}
.warnbox{padding:12px 16px;border-radius:12px;background:var(--warn-bg);color:var(--warn);font-weight:600;font-size:14px;animation:shake .4s ease}
.alert{display:flex;flex-direction:column;gap:4px;padding:14px 18px;border-radius:14px;background:var(--bad-bg);color:var(--bad);animation:shake .4s ease}
.alert b{font-weight:600}.alert p{color:var(--ink);font-size:13.5px}
.okmsg{padding:10px 14px;border-radius:12px;background:var(--ok-bg);color:var(--ok);font-weight:500;font-size:13.5px;animation:rise .3s both}
/* priority network */
.nn{width:100%;height:auto;display:block}
.nn .e.pos{stroke:var(--brand)}.nn .e.neg{stroke:var(--l-urg)}
.nn .e{fill:none;animation:fadeIn .6s both}
.nn .n{stroke:var(--brand);stroke-width:2}
.nn .n.win{stroke:var(--accent);stroke-width:4}
.nn text{font-size:14px;font-weight:600;fill:var(--ink-2)}
.nn text.v{fill:var(--ink);font-size:11px;font-weight:700}
.predword{font-size:34px;font-weight:700;letter-spacing:-.02em;line-height:1.1;animation:pop2 .4s cubic-bezier(.3,1.6,.5,1)}
.bigstat{display:flex;flex-direction:column}
.bigstat b{font-size:32px;font-weight:700;letter-spacing:-.02em;color:var(--brand)}
.chart{width:100%;height:auto;display:block;overflow:visible}
.chart .grid{stroke:var(--rule-2);stroke-width:1}
.chart .ln{fill:none;stroke-width:2.5;stroke-linecap:round;stroke-linejoin:round;stroke-dasharray:1;stroke-dashoffset:1;animation:draw 1.2s ease forwards}
.chart text{font-size:9px;fill:var(--ink-2)}
.cm{display:grid;gap:4px}
.cmcell{display:flex;align-items:center;justify-content:center;min-height:42px;border-radius:8px;background:color-mix(in srgb,var(--brand) calc(var(--a) * 85%),var(--panel-2));color:var(--ink);font-weight:600;font-size:13px;animation:pop2 .4s cubic-bezier(.3,1.6,.5,1) both}
.cmcell.hi{color:var(--brand-ink)}
.cmcell.diag{outline:2px solid var(--accent);outline-offset:-2px}
.cmlab{font-size:11.5px;color:var(--ink-2);display:flex;align-items:center}
/* knowledge base */
.verdict{font-size:40px;font-weight:700;letter-spacing:-.03em;line-height:1}
.verdict.ok{color:var(--ok)}.verdict.no{color:var(--bad)}
.kbline{display:flex;flex-wrap:wrap;gap:4px 12px;padding:10px 4px;border-bottom:1px solid var(--rule-2);font-size:13.5px;animation:fadeIn .4s both}
.kbline:hover{background:var(--panel-2)}
.kbline b{font-weight:600}
.kbline .rn{display:inline-flex;align-items:center;justify-content:center;min-width:38px;padding:1px 8px;border-radius:999px;background:var(--brand);color:var(--brand-ink);font-size:11.5px;font-weight:700;height:fit-content}
.scrollbox{max-height:520px;overflow-y:auto;padding-right:6px}
/* scheduling */
.stat{padding:16px 18px;border-radius:14px;background:var(--panel-2);animation:rise .5s both}
.stat b{display:block;font-size:30px;font-weight:700;color:var(--brand);line-height:1.1}
.stat small{display:block;font-size:12.5px;color:var(--ink-2)}
.cgraph{width:100%;height:auto;display:block}
.cgraph .ge{stroke-width:3;fill:none}
.cgraph .ge.students{stroke:var(--c-ne)}
.cgraph .ge.examiner{stroke:var(--c-exam)}
.cgraph .ge.supervisor{stroke:var(--c-sup)}
.cgraph .ge.before{stroke:var(--c-prec);stroke-width:4}
.cgraph .elabel{font-size:11px;font-weight:600}
.cgraph .elabelbg{fill:var(--panel)}
.cgraph .gn{stroke-width:3}
.cgraph text{fill:var(--ink)}
.cgraph text.sub{fill:var(--ink-2);font-size:11px}
.cgraph g.node{transform-box:fill-box;transform-origin:center;animation:popsvg .45s cubic-bezier(.3,1.6,.5,1) both}
.lg{display:inline-flex;align-items:center;gap:8px;font-size:12.5px;color:var(--ink-2)}
.lg::before{content:'';width:28px;height:4px;border-radius:2px;background:var(--c-ne)}
.lg.examiner::before{background:var(--c-exam)}
.lg.supervisor::before{background:var(--c-sup)}
.lg.before::before{background:var(--c-prec)}
.lg.slot::before{width:14px;height:14px;border-radius:4px;background:var(--sc)}
.bgridbox{overflow-x:auto}
.bgrid{display:grid;grid-template-columns:minmax(150px,1.3fr) repeat(4,minmax(96px,1fr));gap:4px;min-width:640px}
.bgrid .hd{font-size:12px;font-weight:600;color:var(--ink-2);padding:6px 8px}
.bgrid .rm{padding:8px;font-size:13px;font-weight:600;line-height:1.25}
.bgrid .rm small{display:block;font-weight:400;font-size:11.5px;color:var(--ink-2)}
.bcell{padding:8px;border-radius:8px;font-size:12px;line-height:1.25;animation:pop2 .35s cubic-bezier(.3,1.6,.5,1) both;transition:transform .15s}
.bcell:hover{transform:scale(1.04)}
.bcell.free{border:1.5px dashed var(--rule);color:var(--ink-3)}
.bcell.exam{background:color-mix(in srgb,var(--l-elig) 16%,var(--panel));color:var(--l-elig);font-weight:600}
.bcell.fixed{background:var(--panel-2);color:var(--ink-2)}
.bcell.session{background:var(--brand);color:var(--brand-ink);font-weight:600}
.code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px;background:var(--panel-2);padding:1px 6px;border-radius:6px}

/* ---------- motion ---------- */
@keyframes spin{to{transform:rotate(360deg)}}
@keyframes rise{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
@keyframes rowIn{from{opacity:0;transform:translateX(-8px)}to{opacity:1;transform:none}}
@keyframes fadeIn{from{opacity:0}to{opacity:1}}
@keyframes slideIn{from{opacity:0;transform:translateX(-10px)}to{opacity:1;transform:none}}
@keyframes grow{from{transform:scaleX(0)}to{transform:scaleX(1)}}
@keyframes pop2{0%{transform:scale(.6);opacity:0}100%{transform:scale(1);opacity:1}}
@keyframes popsvg{0%{transform:scale(.3);opacity:0}100%{transform:scale(1);opacity:1}}
@keyframes ringsvg{0%{transform:scale(1);opacity:.9}100%{transform:scale(2.2);opacity:0}}
@keyframes hop{0%,100%{transform:translateY(0)}45%{transform:translateY(-6px)}}
@keyframes ping{0%{box-shadow:0 0 0 0 color-mix(in srgb,var(--brand) 45%,transparent)}100%{box-shadow:0 0 0 12px transparent}}
@keyframes shake{0%,100%{transform:translateX(0)}25%{transform:translateX(-5px)}75%{transform:translateX(5px)}}
@keyframes drop{from{opacity:0;transform:translateY(-16px)}to{opacity:1;transform:none}}
@keyframes arrive{from{opacity:0;transform:translateX(-30px) scale(.9)}to{opacity:1;transform:none}}
@keyframes travel{from{width:0}}
@keyframes trainrun{from{left:0}}
@keyframes draw{to{stroke-dashoffset:0}}
@keyframes stnOn{0%{transform:scale(.4);border-color:var(--rule)}100%{transform:scale(1.05)}}
@keyframes stnFail{0%{transform:scale(.4)}40%{transform:translateX(-5px)}70%{transform:translateX(5px)}100%{transform:none}}
@media (prefers-reduced-motion: reduce){.stApp *,.stApp *::before,.stApp *::after{animation-duration:.01ms!important;animation-delay:0s!important;animation-iteration-count:1!important;transition-duration:.01ms!important}}
@media (max-width: 720px){.top h1{font-size:25px}.keys{grid-template-columns:1fr}.trackin{margin:6px 40px 0}}
"""


def page_css(theme):
    """
    Function: page_css
    Purpose : The whole stylesheet with the design tokens of the theme.
    """
    tokens = DARK_TOKENS if theme == "dark" else LIGHT_TOKENS
    token_text = "".join(f"--{name}:{value};" for name, value in tokens.items())
    return ('<link rel="preconnect" href="https://fonts.googleapis.com">'
            '<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap" '
            'rel="stylesheet"><style>' + _CSS.replace("__TOKENS__", token_text) + "</style>")


def wrap(inner):
    """
    Function: wrap
    Purpose : Every HTML block sits in a .sc root so the design styles apply.
    """
    return f'<div class="sc">{inner}</div>'


# -------------------------------------------------------------------
# Small building blocks
# -------------------------------------------------------------------
def brand(subtitle="Request network"):
    """
    Function: brand
    Purpose : Logo mark and product name at the top of the sidebar.
    """
    return wrap(f'<div class="brand"><span class="brand-mark" aria-hidden="true"></span>'
                f'<span><b>Smart Campus AI</b><span>{esc(subtitle)}</span></span></div>')


def page_header(title, intro):
    """
    Function: page_header
    Purpose : Page title and one-line introduction.
    """
    return wrap(f'<header class="top"><h1>{esc(title)}</h1><p>{esc(intro)}</p></header>')


def panel(inner_html, title=None, meta=None, css_class="", label=None):
    """
    Function: panel
    Purpose : White card with an optional heading and a muted note on the
              right. inner_html must already be escaped.
    """
    head = ""
    if title or meta:
        meta_html = f'<span class="muted small">{esc(meta)}</span>' if meta else ""
        head = f'<div class="panel-head"><h2>{esc(title or "")}</h2>{meta_html}</div>'
    aria_label = f' aria-label="{esc(label or title)}"' if (label or title) else ""
    return f'<section class="panel {css_class}"{aria_label}>{head}{inner_html}</section>'


def status_pill(status):
    """
    Function: status_pill
    Purpose : Status text with a station symbol (ok / rejected / warning).
    """
    tone = STATUS_TONES.get(status, "warn")
    return f'<span class="pill tone-{tone}">{esc(status)}</span>'


def priority_pill(priority):
    """
    Function: priority_pill
    Purpose : Priority text with a dot whose size follows the priority.
    """
    if not priority or priority == "-":
        return '<span class="muted">-</span>'
    return f'<span class="pill {PRIORITY_CLASS.get(priority, "p-low")}">{esc(priority)}</span>'


def empty_state(title, text, colours=("var(--l-nav)", "var(--l-elig)", "var(--l-book)", "var(--l-full)")):
    """
    Function: empty_state
    Purpose : Friendly placeholder card shown before there is any data.
    """
    line_bars = "".join(f'<i style="top:{6 + 16 * index}px;background:{colour};animation-delay:{index * 0.1:.1f}s"></i>'
                        for index, colour in enumerate(colours))
    return panel(f'<div class="empty"><div class="art" aria-hidden="true">{line_bars}</div>'
                 f'<h2>{esc(title)}</h2><p>{esc(text)}</p></div>')


def mini_line(stops, colour, show_names=True):
    """
    Function: mini_line
    Purpose : Small transit line with one dot per stop (and stop names).
    """
    dots = "".join(f'<i style="--d:{index * 0.06:.2f}s"></i>' for index in range(len(stops)))
    names = "".join(f"<span>{esc(stop)}</span>" for stop in stops)
    return (f'<div style="--lc:{colour}"><div class="mini" aria-hidden="true">{dots}</div>'
            + (f'<div class="mini-names">{names}</div>' if show_names else "") + "</div>")


def line_stops(request_type, pipelines):
    """
    Function: line_stops
    Purpose : Stop names of a request type's line: Input, Router, its AI
              modules in pipeline order, then Answer.
    """
    return ["Input", "Router"] + list(pipelines.get(request_type, [])) + ["Answer"]


def table(headers, rows, scroll=False):
    """
    Function: table
    Purpose : Plain HTML table. rows are (cells, line_colour) pairs; cells
              are already-escaped HTML strings.
    """
    head = "".join(f"<th>{esc(header)}</th>" for header in headers)
    body = []
    for index, (cells, colour) in enumerate(rows):
        delay = min(index, 20) * 0.03
        cell_html = "".join(f"<td>{cell}</td>" for cell in cells)
        body.append(f'<tr class="stream" style="--lc:{colour};animation-delay:{delay:.2f}s">{cell_html}</tr>')
    return (f'<div class="tablebox{" scroll" if scroll else ""}"><table><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def id_cell(text, colour):
    """
    Function: id_cell
    Purpose : Request id with a station dot in its line colour.
    """
    return f'<span class="idcell" style="--lc:{colour}"><i aria-hidden="true"></i>{esc(text)}</span>'


# -------------------------------------------------------------------
# Home
# -------------------------------------------------------------------
def tiles(items):
    """
    Function: tiles
    Purpose : Row of big-number tiles.
    Input   : list of (number, label, tone) with tone ok/bad/warn/brand/''
    """
    cells = "".join(f'<div class="tile {tone}" style="animation-delay:{index * 0.06:.2f}s">'
                    f'<span class="num">{esc(number)}</span><span class="lab">{esc(label)}</span></div>'
                    for index, (number, label, tone) in enumerate(items))
    return f'<div class="tiles">{cells}</div>'


def network(pipelines):
    """
    Function: network
    Purpose : The five request lines with their stops (routing table of
              the Request Router).
    """
    rows = []
    for request_type, line in LINES.items():
        rows.append(f'<div class="netline" style="--lc:{line["var"]}"><b><i aria-hidden="true"></i>{esc(line["label"])}</b>'
                    f'{mini_line(line_stops(request_type, pipelines), line["var"])}</div>')
    return panel("".join(rows), "The five lines", "Hover a line to trace its stops", "grow2 network")


def facts_panel(facts, note):
    """
    Function: facts_panel
    Purpose : Counts of locations, rules, rooms and slots plus a note.
    """
    chips = "".join(f'<span class="fact"><b>{esc(number)}</b><span class="muted small">{esc(label)}</span></span>'
                    for number, label in facts)
    return panel(f'<div class="col"><div class="facts">{chips}</div><p class="muted small">{esc(note)}</p></div>',
                 "On the network", css_class="grow")


def status_stack(counts):
    """
    Function: status_stack
    Purpose : One stacked bar of request counts per status, with a legend.
    """
    total = max(sum(counts.values()), 1)
    segments, legend = [], []
    for status, count in counts.items():
        tone = STATUS_TONES.get(status, "warn")
        segments.append(f'<span class="sseg tone-{tone}" style="width:{100 * count / total:.2f}%" '
                        f'title="{esc(status)}: {count}"></span>')
        legend.append(f'<span class="tone-{tone}"><i aria-hidden="true"></i>'
                      f'<span style="color:var(--ink)">{esc(status)} {count}</span></span>')
    return f'<div class="stack">{"".join(segments)}</div><div class="legend">{"".join(legend)}</div>'


def hbars(items, label_width=120):
    """
    Function: hbars
    Purpose : Horizontal bar chart.
    Input   : list of (label, value, colour)
    """
    largest = max([value for _, value, _ in items] + [1])
    return "".join(f'<div class="hbar" style="grid-template-columns:{label_width}px minmax(0,1fr) 40px">'
                   f'<span>{esc(label)}</span><span class="track2"><span class="fill" '
                   f'style="width:{100 * value / largest:.1f}%;background:{colour};animation-delay:{index * 0.08:.2f}s">'
                   f'</span></span><b style="text-align:right">{esc(value)}</b></div>'
                   for index, (label, value, colour) in enumerate(items))


# -------------------------------------------------------------------
# Request result: the train on its line
# -------------------------------------------------------------------
def result_caption(response, stages):
    """
    Function: result_caption
    Purpose : One sentence saying where the request stopped or arrived.
    """
    failed_stage = next(((module, summary) for module, state, summary in stages if state == "error"), None)
    stations_run = sum(1 for _, state, _ in stages if state == "ok")
    if failed_stage:
        return f"Stopped at {STATION_NAMES.get(failed_stage[0], failed_stage[0])}: {failed_stage[1]}"
    return f"Arrived at the final response after {stations_run + 1} stops: {response['status']}."


def short_text(text, limit=60):
    """
    Function: short_text
    Purpose : Shorten a long station note; the full text stays in the
              caption, the tooltip and the explanation.
    """
    text = str(text)
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def result_line(response, stages):
    """
    Function: result_line
    Purpose : Result header (request id, line badge, status sign) and the
              animated line: the train runs from station to station; skipped
              stops are dashed, a failure ends at a red buffer stop.
    """
    line = line_of(response["request_type"])
    tone = STATUS_TONES.get(response["status"], "warn")
    stations = list(stages) + [("Response", "ok", "answer combined and logged")]
    failed_index = next((index for index, (_, state, _) in enumerate(stations) if state == "error"), None)
    if failed_index is not None:
        stations = [(module, "pending" if index > failed_index else state,
                     summary if index <= failed_index else "not reached")
                    for index, (module, state, summary) in enumerate(stations)]
    station_count = len(stations)
    end_index = failed_index if failed_index is not None else station_count - 1
    seconds_per_stop = 0.45
    station_classes = {"ok": "ran", "error": "failed", "skip": "skipped", "pending": "pending"}
    label_classes = {"ok": "", "error": "failed", "skip": "skipped", "pending": "pending"}
    parts = []
    for index, (module, state, summary) in enumerate(stations):
        left = 100 * index / (station_count - 1)
        delay = index * seconds_per_stop
        parts.append(f'<div class="stn {station_classes[state]}" style="left:{left:.2f}%;animation-delay:{delay:.2f}s"></div>')
        if state == "error":
            parts.append(f'<div class="buffer" style="left:{left:.2f}%;animation-delay:{delay + 0.2:.2f}s"></div>')
        parts.append(f'<div class="slabel {label_classes[state]}" style="left:{left:.2f}%;animation-delay:{delay:.2f}s">'
                     f'<b>{esc(STATION_NAMES.get(module, module))}</b><span title="{esc(summary)}">{esc(short_text(summary))}</span></div>')
    end_left = 100 * end_index / (station_count - 1)
    duration = max(end_index, 1) * seconds_per_stop
    sign = (f'<span class="status-sign tone-{tone}" style="animation-delay:{duration:.2f}s">'
            f'<i aria-hidden="true"></i><span>{esc(response["status"])}</span></span>')
    head = (f'<div class="rhead"><div style="display:flex;flex-direction:column;gap:6px">'
            f'<h2>{esc(response["request_id"])}</h2><span class="rbadge"><i aria-hidden="true"></i>'
            f'{esc(line["label"])} line · {esc(response["user_id"])} · {esc(response["role"])}</span></div>{sign}</div>')
    track = (f'<div class="trackbox"><div class="trackin" style="--dur:{duration:.2f}s">'
             f'<div class="track"></div><div class="travel" style="width:{end_left:.2f}%"></div>'
             f'{"".join(parts)}<div class="train" style="left:{end_left:.2f}%" aria-hidden="true">'
             f'{esc(response["request_id"])}</div></div></div>')
    caption = (f'<p class="caption" style="animation-delay:{duration:.2f}s" aria-live="polite">'
               f'{esc(result_caption(response, stages))}</p>')
    return (f'<section class="result" style="--line:{line["var"]}" aria-label="Request result">'
            f'<div class="panel" style="display:flex;flex-direction:column;gap:18px">{head}{caption}{track}</div></section>')


def key_results(response):
    """
    Function: key_results
    Purpose : The four headline outputs of the Final Response Layer:
              priority (ANN), permission (KB), assignment (CSP), route cost.
    """
    search = response["details"].get("search", {})
    items = [("Priority (ANN)", response["priority"]),
             ("Permission (KB)", response["authorization"]),
             ("Assignment (CSP)", response["assignment"]),
             ("Route cost", search.get("cost"))]
    cells = []
    for index, (label, value) in enumerate(items):
        empty = value in (None, "", "-")
        css_class = "none" if empty else ("bad" if value == "DENIED" else "")
        cells.append(f'<div class="kv {css_class}" style="animation-delay:{0.1 + index * 0.08:.2f}s"><small>{esc(label)}</small>'
                     f'<b>{"-" if empty else esc(value)}</b></div>')
    return f'<div class="keys">{"".join(cells)}</div>'


def probability_bars(probabilities):
    """
    Function: probability_bars
    Purpose : One bar per priority class; the predicted class is highlighted.
    """
    best_class = max(probabilities, key=lambda name: float(probabilities[name]))
    rows = []
    for class_name in config.PRIORITY_CLASSES:
        value = float(probabilities[class_name])
        rows.append(f'<div class="pbar{" top" if class_name == best_class else ""}"><span>{esc(class_name)}</span>'
                    f'<span class="t"><span class="f" style="width:{100 * value:.1f}%"></span></span>'
                    f'<b style="text-align:right">{value:.0%}</b></div>')
    return "".join(rows)


def feature_chips(features):
    """
    Function: feature_chips
    Purpose : The scaled ANN inputs x1..x7 with a small bar each.
    """
    cells = "".join(f'<div class="feat" style="animation-delay:{index * 0.05:.2f}s">'
                    f'<span class="muted">x{index + 1} {esc(name)}</span><b>{float(value):.2f}</b>'
                    f'<span class="t"><i style="width:{100 * min(max(float(value), 0), 1):.0f}%"></i></span></div>'
                    for index, (name, value) in enumerate(features.items()))
    return f'<div class="feats">{cells}</div>'


def proof_steps(proof, missing):
    """
    Function: proof_steps
    Purpose : Backward-chaining proof as one row per step, tagged with the
              rule used (or 'fact'), followed by any missing premises.
    """
    parts = []
    for index, step in enumerate(proof):
        is_fact = step.endswith("is a known fact")
        if " by " in step and not is_fact:
            goal, _, rest = step.partition(" by ")
            tag = rest.split(" ", 1)[0]
        else:
            goal, tag = step.replace(" is a known fact", ""), "fact"
        parts.append(f'<div class="pstep{" isfact" if is_fact else ""}" style="animation-delay:{index * 0.06:.2f}s">'
                     f'<b>{esc(goal)}</b><span>{esc(tag)}</span></div>')
    parts.extend(f'<span class="miss">Missing: {esc(reason)}</span>' for reason in missing)
    return "".join(parts)


def logic_checks(checks):
    """
    Function: logic_checks
    Purpose : Each Logic/KB check of a request with its verdict and proof.
    """
    cells = []
    for label, result in checks:
        proved = result["entailed"]
        cells.append(f'<div class="check {"ok" if proved else "no"}"><div class="ch"><span>{esc(label)}: '
                     f'<span class="code">{esc(result["query"])}</span></span>'
                     f'<span class="v">{"Proved" if proved else "Not proved"}</span></div>'
                     f'{proof_steps(result["proof"], result["missing"])}</div>')
    return "".join(cells)


def order_chips(order, current=None):
    """
    Function: order_chips
    Purpose : Numbered chips for the order a search expanded nodes.
    """
    return '<div class="chips">' + "".join(
        f'<span class="ochip{" cur" if name == current else ""}" style="animation-delay:{min(index, 20) * 0.03:.2f}s">'
        f'<i>{index + 1}</i>{nice(name)}</span>' for index, name in enumerate(order)) + "</div>"


def explanation(sentences):
    """
    Function: explanation
    Purpose : Numbered explanation from the Final Response Layer.
    """
    items = "".join(f'<li style="animation-delay:{index * 0.07:.2f}s">{esc(sentence)}</li>'
                    for index, sentence in enumerate(sentences))
    return panel(f'<ol class="expl">{items}</ol>', "Explanation")


# -------------------------------------------------------------------
# Campus map (SVG)
# -------------------------------------------------------------------
MAP_W, MAP_H = 760, 360
# Label placement (dx, dy, anchor) where the default "below the stop" would collide.
MAP_LABELS = {"Bus_Stop": (-16, 5, "end"), "Main_Gate": (-16, 5, "end"), "Library": (-12, -12, "end"),
              "Student_Services": (12, 24, "start"), "Medical_Center": (0, 28, "middle")}


def _map_xy(x_value, y_value):
    """
    Function: _map_xy
    Purpose : Campus coordinates (from the project's campus graph) to SVG
              pixels; y grows upward on the campus and downward in SVG.
    """
    return 90 + x_value * 60, 318 - y_value * 52


def _edge_path(start, end, other_points):
    """
    Function: _edge_path
    Purpose : SVG path of one edge: straight unless another building sits
              on it (e.g. Admin_Block - Exam_Hall), then bent upward.
    Returns : (path_text, label_point)
    """
    (x1, y1), (x2, y2) = start, end
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy) or 1
    for (point_x, point_y) in other_points:
        along = ((point_x - x1) * dx + (point_y - y1) * dy) / length ** 2
        distance_to_line = abs(dx * (point_y - y1) - dy * (point_x - x1)) / length
        if 0.05 < along < 0.95 and distance_to_line < 12:
            normal_x, normal_y = -dy / length, dx / length
            if normal_y > 0:
                normal_x, normal_y = -normal_x, -normal_y
            control_x, control_y = (x1 + x2) / 2 + normal_x * 34, (y1 + y2) / 2 + normal_y * 34
            middle = (0.25 * x1 + 0.5 * control_x + 0.25 * x2, 0.25 * y1 + 0.5 * control_y + 0.25 * y2)
            return f"M{x1:.1f},{y1:.1f} Q{control_x:.1f},{control_y:.1f} {x2:.1f},{y2:.1f}", middle
    return f"M{x1:.1f},{y1:.1f} L{x2:.1f},{y2:.1f}", ((x1 + x2) / 2, (y1 + y2) / 2)


def campus_map(campus_graph, path=None, expanded=None, current=None, line="var(--brand)", label="Campus map"):
    """
    Function: campus_map
    Purpose : Campus graph as an SVG transit map. The route is drawn
              segment by segment; explored nodes are tinted and numbered in
              the order the search expanded them; the current node rings.
    """
    path = path or []
    expanded = list(expanded or [])
    points = {name: _map_xy(*xy) for name, xy in campus_graph.coordinates.items()}
    route_index = {frozenset(pair): index for index, pair in enumerate(zip(path[:-1], path[1:]))}
    base_edges, route_segments, weights = [], [], []
    for node_a, node_b, cost in campus_graph.edges:
        others = [point for name, point in points.items() if name not in (node_a, node_b)]
        path_text, middle = _edge_path(points[node_a], points[node_b], others)
        base_edges.append(f'<path class="medge" d="{path_text}"/>')
        weights.append(f'<rect class="mwbg" x="{middle[0] - 9:.1f}" y="{middle[1] - 8:.1f}" width="18" height="15" rx="4"/>'
                       f'<text class="mw" x="{middle[0]:.1f}" y="{middle[1] + 4:.1f}" text-anchor="middle">{cost}</text>')
        edge_key = frozenset((node_a, node_b))
        if edge_key in route_index:
            index = route_index[edge_key]
            if path[index] != node_a:            # draw the segment in travel direction
                path_text = _edge_path(points[node_b], points[node_a], others)[0]
            route_segments.append((index, f'<path class="mseg" pathLength="1" d="{path_text}" '
                                          f'style="animation-delay:{0.15 + index * 0.3:.2f}s"/>'))
    route_segments = [segment for _, segment in sorted(route_segments)]
    nodes, labels = [], []
    order_index = {name: index for index, name in enumerate(expanded)}
    for name, (x_value, y_value) in points.items():
        css_class, radius = "mnode", 9
        if name in path:
            css_class += " onpath"
        elif name in order_index:
            css_class += " ex"
        if path and name in (path[0], path[-1]):
            css_class += " end"
            radius = 12
        if name == current:
            css_class += " cur"
        delay = order_index[name] * 0.04 if name in order_index else 0
        nodes.append(f'<circle class="{css_class} pop" cx="{x_value:.1f}" cy="{y_value:.1f}" r="{radius}" '
                     f'style="animation-delay:{delay:.2f}s"><title>{nice(name)}</title></circle>')
        if name == current:
            nodes.append(f'<circle class="ring" cx="{x_value:.1f}" cy="{y_value:.1f}" r="{radius}"/>')
        if name in order_index:
            nodes.append(f'<g class="pop" style="animation-delay:{delay:.2f}s">'
                         f'<circle class="mord" cx="{x_value + 13:.1f}" cy="{y_value - 13:.1f}" r="9"/>'
                         f'<text class="mordt" x="{x_value + 13:.1f}" y="{y_value - 9.5:.1f}" text-anchor="middle">'
                         f'{order_index[name] + 1}</text></g>')
        offset_x, offset_y, anchor = MAP_LABELS.get(name, (0, -18 if y_value < 90 else 26, "middle"))
        labels.append(f'<text class="mlabel" x="{x_value + offset_x:.1f}" y="{y_value + offset_y:.1f}" '
                      f'text-anchor="{anchor}">{nice(name)}</text>')
    return (f'<svg class="map" viewBox="0 0 {MAP_W} {MAP_H}" role="img" aria-label="{esc(label)}" style="--line:{line}">'
            f'{"".join(base_edges)}{"".join(route_segments)}{"".join(weights)}{"".join(nodes)}{"".join(labels)}</svg>')


def map_legend(walk=False, line="var(--brand)"):
    """
    Function: map_legend
    Purpose : Legend under a campus map (route map or search walkthrough).
    """
    if walk:
        return (f'<div class="maplegend" style="--line:{line}"><span><i class="ex"></i>Explored, numbered in order</span>'
                '<span><i class="cur"></i>Expanding now</span><span><i class="p"></i>Final path</span></div>')
    return (f'<div class="maplegend" style="--line:{line}"><span><i class="p"></i>Route</span><span><i></i>Building</span>'
            '<span style="font-weight:600">3</span><span>walking cost</span></div>')


# -------------------------------------------------------------------
# Priority network and training charts
# -------------------------------------------------------------------
def nn_diagram(mlp, feature_vector, input_names, output_names):
    """
    Function: nn_diagram
    Purpose : The 7-4-3-4 MLP (inputs x1..x7 in the fixed order of the
              project diagram) with the activations of one input. Edge
              colour is the weight's sign, thickness its strength; the
              predicted priority node rings.
    """
    activations = mlp.forward(np.asarray(feature_vector, dtype=float).reshape(1, -1))
    layer_sizes = [len(layer[0]) for layer in activations]
    width, height = 720, 320
    layer_x = np.linspace(170, width - 90, len(layer_sizes))
    positions = []
    for layer, size in enumerate(layer_sizes):
        layer_y = np.linspace(30, height - 30, size) if size > 1 else [height / 2]
        positions.append([(layer_x[layer], y_value) for y_value in layer_y])
    edges = []
    for layer, weight_matrix in enumerate(mlp.weights):
        strongest = float(np.abs(weight_matrix).max()) or 1.0
        for source, (x1, y1) in enumerate(positions[layer]):
            for target, (x2, y2) in enumerate(positions[layer + 1]):
                weight = float(weight_matrix[source, target])
                strength = abs(weight) / strongest
                edges.append(f'<line class="e {"pos" if weight >= 0 else "neg"}" x1="{x1:.1f}" y1="{y1:.1f}" '
                             f'x2="{x2:.1f}" y2="{y2:.1f}" stroke-width="{0.6 + 3.4 * strength:.2f}" '
                             f'stroke-opacity="{0.15 + 0.6 * strength:.2f}" style="animation-delay:{layer * 0.15:.2f}s"/>')
    winner = int(np.argmax(activations[-1][0]))
    nodes = []
    for layer, layer_positions in enumerate(positions):
        values = activations[layer][0]
        for index, (x_value, y_value) in enumerate(layer_positions):
            value = float(values[index])
            amount = min(abs(value), 1.0)
            is_winner = layer == len(positions) - 1 and index == winner
            text_colour = "var(--brand-ink)" if amount > 0.55 else "var(--ink)"
            nodes.append(f'<circle class="n{" win" if is_winner else ""}" cx="{x_value:.1f}" cy="{y_value:.1f}" r="16" '
                         f'style="fill:color-mix(in srgb,var(--brand) {amount * 100:.0f}%,var(--panel))"/>'
                         f'<text class="v" x="{x_value:.1f}" y="{y_value + 3.5:.1f}" text-anchor="middle" '
                         f'style="fill:{text_colour}">{value:.2f}</text>')
            if layer == 0:
                nodes.append(f'<text x="{x_value - 24:.1f}" y="{y_value + 4:.1f}" text-anchor="end">'
                             f'x{index + 1} {esc(input_names[index])}</text>')
            if layer == len(positions) - 1:
                nodes.append(f'<text x="{x_value + 24:.1f}" y="{y_value + 4:.1f}" '
                             f'style="{"fill:var(--accent)" if is_winner else ""}">{esc(output_names[index])}</text>')
    return (f'<svg class="nn" viewBox="0 0 {width} {height}" role="img" '
            f'aria-label="Neural network 7-4-3-4 with current activations; predicted {esc(output_names[winner])}">'
            f'{"".join(edges)}{"".join(nodes)}</svg>')


def line_chart(series, label, y_max=None):
    """
    Function: line_chart
    Purpose : Small SVG line chart over epochs.
    Input   : series = list of (values, colour, name)
    """
    width, height = 300, 110
    longest = max((len(values) for values, _, _ in series), default=1)
    top_value = y_max if y_max is not None else (max((max(values) for values, _, _ in series if len(values)),
                                                     default=1) or 1)
    grid = "".join(f'<line class="grid" x1="0" y1="{y_value}" x2="{width}" y2="{y_value}"/>'
                   for y_value in (0, height / 2, height))
    lines, legend = [], []
    for values, colour, name in series:
        if len(values) < 2:
            continue
        points = " ".join(f"{width * index / (longest - 1):.1f},{height - height * min(value / top_value, 1):.1f}"
                          for index, value in enumerate(values))
        lines.append(f'<polyline class="ln" pathLength="1" points="{points}" style="stroke:{colour}"/>')
        legend.append(f'<span style="color:{colour}"><i aria-hidden="true"></i>'
                      f'<span style="color:var(--ink)">{esc(name)}</span></span>')
    axis = (f'<text x="0" y="{height + 12}">1</text><text x="{width}" y="{height + 12}" text-anchor="end">'
            f'{longest} epochs</text><text x="-2" y="-1" text-anchor="start">{top_value:g}</text>')
    return (f'<svg class="chart" viewBox="-4 -8 308 132" role="img" aria-label="{esc(label)}">{grid}{"".join(lines)}{axis}</svg>'
            f'<div class="legend">{"".join(legend)}</div>')


def confusion(matrix, labels):
    """
    Function: confusion
    Purpose : Confusion matrix as a shaded grid (rows actual, columns
              predicted); correct predictions on the diagonal are outlined.
    """
    matrix = np.asarray(matrix)
    largest = matrix.max() or 1
    cells = ["<span></span>"] + [f'<span class="cmlab" style="justify-content:center">Pred {esc(name)}</span>'
                                 for name in labels]
    for row, row_label in enumerate(labels):
        cells.append(f'<span class="cmlab">{esc(row_label)}</span>')
        for column in range(len(labels)):
            amount = matrix[row, column] / largest
            cells.append(f'<span class="cmcell{" diag" if row == column else ""}{" hi" if amount > 0.55 else ""}" '
                         f'style="--a:{amount:.3f};animation-delay:{(row * len(labels) + column) * 0.03:.2f}s">'
                         f'{matrix[row, column]}</span>')
    return (f'<div class="cm" style="grid-template-columns:76px repeat({len(labels)},minmax(0,1fr))">'
            f'{"".join(cells)}</div>')


# -------------------------------------------------------------------
# Navigation comparison
# -------------------------------------------------------------------
def algo_cards(results, best_name):
    """
    Function: algo_cards
    Purpose : One card per search algorithm with cost, stops, nodes
              expanded and the path found.
    """
    cards = []
    for index, result in enumerate(results):
        is_best = result["algorithm"] == best_name
        flag = '<span class="bestflag">Best route</span>' if is_best else ""
        cards.append(f'<div class="algocard grow{" best" if is_best else ""}" '
                     f'style="animation-delay:{index * 0.08:.2f}s;flex-basis:220px">'
                     f'<div style="display:flex;justify-content:space-between;align-items:center;gap:8px">'
                     f'<h3>{esc(result["algorithm"])}</h3>{flag}</div>'
                     f'<div class="nums"><div><b>{esc(result["cost"])}</b><span>cost</span></div>'
                     f'<div><b>{result["hops"]}</b><span>stops</span></div>'
                     f'<div><b>{result["nodes_expanded"]}</b><span>expanded</span></div></div>'
                     f'<p class="small muted">{" → ".join(nice(node) for node in result["path"])}</p></div>')
    return '<div class="row">' + "".join(cards) + "</div>"


def compare_bars(results, key, accent=False):
    """
    Function: compare_bars
    Purpose : Bars comparing one measure (cost, stops, expanded) per algorithm.
    """
    largest = max([result[key] for result in results] + [1])
    return "".join(f'<div class="cmp"><span>{esc(result["algorithm"])}</span><span class="t"><i class="{"c" if accent else ""}" '
                   f'style="width:{100 * result[key] / largest:.1f}%;animation-delay:{index * 0.08:.2f}s"></i></span>'
                   f'<b style="text-align:right">{esc(result[key])}</b></div>' for index, result in enumerate(results))


# -------------------------------------------------------------------
# CSP
# -------------------------------------------------------------------
def stats_row(items):
    """
    Function: stats_row
    Purpose : Row of solver statistics.
    Input   : list of (number, label)
    """
    return '<div class="tiles">' + "".join(
        f'<div class="stat" style="animation-delay:{index * 0.06:.2f}s"><b>{esc(number)}</b><small>{esc(label)}</small></div>'
        for index, (number, label) in enumerate(items)) + "</div>"


def constraint_graph(solution, groups, edges, precedence, positions):
    """
    Function: constraint_graph
    Purpose : The G1-G6 exam constraint graph in the layout and colours of
              the project's CSP diagram (red != shared students, purple
              examiner clash, orange supervisor clash, blue precedence
              arrow). Each group's box is tinted by its assigned time slot.
    """
    width, height = 720, 380
    all_x = [point[0] for point in positions.values()]
    all_y = [point[1] for point in positions.values()]

    def to_pixels(group):
        """
        Function: to_pixels
        Purpose : Diagram position of a group (inches) to SVG pixels.
        """
        x_value, y_value = positions[group]
        return (90 + (x_value - min(all_x)) / ((max(all_x) - min(all_x)) or 1) * (width - 180),
                height - 60 - (y_value - min(all_y)) / ((max(all_y) - min(all_y)) or 1) * (height - 120))

    def edge_label(x1, y1, x2, y2, text, colour_var):
        """
        Function: edge_label
        Purpose : Constraint label at the middle of an edge.
        """
        middle_x, middle_y = (x1 + x2) / 2, (y1 + y2) / 2
        box_width = 7 * len(text) + 10
        return (f'<rect class="elabelbg" x="{middle_x - box_width / 2:.1f}" y="{middle_y - 9:.1f}" '
                f'width="{box_width}" height="16" rx="5"/><text class="elabel" x="{middle_x:.1f}" '
                f'y="{middle_y + 3.5:.1f}" text-anchor="middle" style="fill:var({colour_var})">{esc(text)}</text>')

    lines, labels = [], []
    for group_a, group_b, constraint in edges:
        if "examiner" in constraint:
            kind, colour_var, short_text = "examiner", "--c-exam", "!= examiner clash"
        elif "supervisor" in constraint:
            kind, colour_var, short_text = "supervisor", "--c-sup", "!= supervisor clash"
        else:
            kind, colour_var, short_text = "students", "--c-ne", "!="
        (x1, y1), (x2, y2) = to_pixels(group_a), to_pixels(group_b)
        lines.append(f'<line class="ge {kind}" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}">'
                     f'<title>{esc(group_a)}–{esc(group_b)}: {esc(constraint)}</title></line>')
        labels.append(edge_label(x1, y1, x2, y2, short_text, colour_var))
    for group_a, group_b, constraint in precedence:
        (x1, y1), (x2, y2) = to_pixels(group_a), to_pixels(group_b)
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy) or 1
        arrow_x, arrow_y = x2 - dx / length * 66, y2 - dy / length * 40      # stop at the box edge
        lines.append(f'<line class="ge before" x1="{x1:.1f}" y1="{y1:.1f}" x2="{arrow_x:.1f}" y2="{arrow_y:.1f}" '
                     f'marker-end="url(#arrowhead)"><title>{esc(group_a)} before {esc(group_b)}</title></line>')
        labels.append(edge_label(x1, y1, x2, y2, constraint, "--c-prec"))
    nodes = []
    for index, (group, (slot, room)) in enumerate(sorted(solution.items())):
        x_value, y_value = to_pixels(group)
        slot_index = config.TIME_SLOTS.index(slot)
        nodes.append(f'<g class="node" style="animation-delay:{index * 0.08:.2f}s">'
                     f'<rect class="gn" x="{x_value - 62:.1f}" y="{y_value - 38:.1f}" width="124" height="76" rx="16" '
                     f'style="stroke:var(--s{slot_index});fill:color-mix(in srgb,var(--s{slot_index}) 14%,var(--panel))"/>'
                     f'<text x="{x_value:.1f}" y="{y_value - 14:.1f}" text-anchor="middle" style="font-weight:700;font-size:15px">'
                     f'{esc(group)}</text><text class="sub" x="{x_value:.1f}" y="{y_value + 3:.1f}" text-anchor="middle">'
                     f'{groups[group]} students · {esc(slot[:5])}</text>'
                     f'<text class="sub" x="{x_value:.1f}" y="{y_value + 19:.1f}" text-anchor="middle">{nice(room)}</text></g>')
    definitions = ('<defs><marker id="arrowhead" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" '
                   'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--c-prec)"/></marker></defs>')
    legend = ('<div class="legend" style="gap:8px 18px"><span class="lg">!= shared students</span>'
              '<span class="lg examiner">!= examiner clash</span><span class="lg supervisor">!= supervisor clash</span>'
              '<span class="lg before">&lt; precedence</span>'
              + "".join(f'<span class="lg slot" style="--sc:var(--s{index})">{esc(slot)}</span>'
                        for index, slot in enumerate(config.TIME_SLOTS)) + "</div>")
    return (f'<svg class="cgraph" viewBox="0 0 {width} {height}" role="img" aria-label="Exam constraint graph">{definitions}'
            f'{"".join(lines)}{"".join(labels)}{"".join(nodes)}</svg>{legend}')


def booking_grid(booking_table):
    """
    Function: booking_grid
    Purpose : Rooms x time slots grid showing every booking (exams, fixed
              weekly bookings, bookings made by requests, free slots).
    """
    cells = ['<span class="hd">Room</span>'] + [f'<span class="hd">{esc(slot)}</span>' for slot in config.TIME_SLOTS]
    occupied = {(row.Slot, row.Room): row.Booked_By for row in booking_table.itertuples()}
    cell_number = 0
    for room, room_info in config.ROOMS.items():
        cells.append(f'<span class="rm">{nice(room)}<small>{esc(room_info["type"])} · '
                     f'{room_info["capacity"]} seats</small></span>')
        for slot in config.TIME_SLOTS:
            owner = occupied.get((slot, room))
            if owner is None:
                css_class, text = "free", "free"
            elif owner.startswith("Exam"):
                css_class, text = "exam", owner
            elif owner in FIXED_BOOKINGS:
                css_class, text = "fixed", owner
            else:
                css_class, text = "session", owner
            cells.append(f'<span class="bcell {css_class}" style="animation-delay:{cell_number * 0.012:.3f}s">'
                         f'{esc(text)}</span>')
            cell_number += 1
    legend = ('<div class="legend"><span style="color:color-mix(in srgb,var(--l-elig) 30%,var(--panel))"><i></i>'
              '<span style="color:var(--ink)">Exam</span></span><span style="color:var(--panel-2)">'
              '<i style="outline:1px solid var(--rule)"></i><span style="color:var(--ink)">Fixed weekly</span></span>'
              '<span style="color:var(--brand)"><i></i><span style="color:var(--ink)">Booked by a request</span></span>'
              '<span style="color:var(--rule)"><i></i><span style="color:var(--ink)">Free</span></span></div>')
    return f'<div class="bgridbox"><div class="bgrid">{"".join(cells)}</div></div>{legend}'
