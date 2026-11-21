import gradio as gr
from ultralytics import YOLO

model = YOLO("best.pt")  # classes: paper, rock, scissors

BEATS = {"rock": "scissors", "paper": "rock", "scissors": "paper"}
COLORS = {"rock": "#F9A8D4", "paper": "#A5B4FC", "scissors": "#86EFAC"}
GAME_MODE, ADVANCED = "Game Mode", "Advanced Settings"
GAME_CONF = 0.5  # fixed confidence in Game Mode


# ---------- icons (solid fills, so they also render when a page is hidden) ----------
def icon(kind, size=72):
    body = {
        "rock": '<rect x="22" y="40" width="56" height="42" rx="18"/><rect x="24" y="30" width="13" height="28" rx="6.5"/>'
                '<rect x="38" y="26" width="13" height="30" rx="6.5"/><rect x="52" y="27" width="13" height="30" rx="6.5"/>'
                '<rect x="66" y="31" width="12" height="26" rx="6"/><rect x="16" y="56" width="30" height="14" rx="7" transform="rotate(-20 31 63)"/>',
        "paper": '<rect x="27" y="22" width="11" height="44" rx="5.5"/><rect x="39" y="12" width="11" height="52" rx="5.5"/>'
                 '<rect x="51" y="14" width="11" height="50" rx="5.5"/><rect x="63" y="24" width="11" height="42" rx="5.5"/>'
                 '<rect x="12" y="56" width="30" height="12" rx="6" transform="rotate(-35 27 62)"/><rect x="26" y="50" width="48" height="40" rx="18"/>',
        "scissors": '<rect x="36" y="10" width="11" height="54" rx="5.5" transform="rotate(-14 41 60)"/>'
                    '<rect x="55" y="10" width="11" height="54" rx="5.5" transform="rotate(14 60 60)"/>'
                    '<rect x="26" y="54" width="48" height="36" rx="16"/><rect x="16" y="62" width="30" height="12" rx="6" transform="rotate(-15 31 68)"/>',
    }[kind]
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 100 100" fill="#E9D5FF" stroke="#7C3AED" '
            f'stroke-width="1.4" aria-hidden="true">{body}</svg>')


def line_icon(path):
    return ('<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#C4B5FD" stroke-width="1.8" '
            f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{path}</svg>')


LOGO = ('<svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#C4B5FD" stroke-width="1.8" aria-hidden="true">'
        '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3"/></svg>')

# ---------- static page content ----------
NAV_BRAND = f'<div class="brand">{LOGO}<span>RPS Vision</span></div>'

HERO = f"""
<div class="hero">
  <div class="deco l"></div><div class="deco r"></div><div class="ring"></div>
  <span class="badge">AI-Powered</span>
  <h1>Rock Paper <span class="grad">Scissors</span></h1>
  <p class="lead">Play the classic game with Computer Vision!</p>
  <p class="sub">Show your hand, let YOLOv8 detect, and see who wins!</p>
  <div class="stage">
    <i class="c tl"></i><i class="c tr"></i><i class="c bl"></i><i class="c br"></i>
    <div class="cards">
      <div class="gcard tilt-l">{icon("rock", 96)}<b>Rock</b></div>
      <div class="gcard mid">{icon("paper", 120)}<b>Paper</b></div>
      <div class="gcard tilt-r">{icon("scissors", 96)}<b>Scissors</b></div>
    </div>
  </div>
</div>
"""

FEATURES = f"""
<div class="features">
  <div class="feat"><div class="tile">{line_icon('<path d="M4 8h3l2-3h6l2 3h3v11H4z"/><circle cx="12" cy="13" r="3.5"/>')}</div>
    <div><b>Real-time Detection</b><span>Detects hand gestures using YOLOv8</span></div></div>
  <div class="feat"><div class="tile">{line_icon('<path d="M13 3L5 14h6l-1 7 8-11h-6z"/>')}</div>
    <div><b>Two Players</b><span>Play with a friend using one camera</span></div></div>
  <div class="feat"><div class="tile">{line_icon('<path d="M5 20V12M12 20V5M19 20V9"/>')}</div>
    <div><b>Track Score</b><span>See the winner and game results</span></div></div>
</div>
"""

HOW = """
<div class="hero">
  <h1 style="font-size:44px">How It <span class="grad">Works</span></h1>
  <p class="sub">Three simple steps</p>
  <div class="steps">
    <div class="card"><div class="num">1</div><b>Show your hands</b><span>Take a photo or upload one with two hands: Player 1 on the left, Player 2 on the right.</span></div>
    <div class="card"><div class="num">2</div><b>AI detects</b><span>YOLOv8 finds each hand and tells rock, paper or scissors.</span></div>
    <div class="card"><div class="num">3</div><b>See who wins</b><span>Rock beats scissors, scissors beats paper, paper beats rock.</span></div>
  </div>
</div>
"""

ABOUT = """
<div class="hero">
  <h1 style="font-size:44px">About <span class="grad">RPS Vision</span></h1>
  <div class="card" style="max-width:720px;margin:24px auto 0;text-align:left;line-height:1.7">
    RPS Vision is a computer vision project. A YOLOv8 model, trained on a custom dataset, detects three hand gestures
    (rock, paper, scissors) and this app turns it into a two-player game built with Gradio.
  </div>
</div>
"""

CSS = """
.gradio-container {
  --body-background-fill:#0B0820; --background-fill-primary:#161033; --background-fill-secondary:#1B1442;
  --block-background-fill:#161033; --block-border-color:#3B2F7A; --border-color-primary:#3B2F7A;
  --body-text-color:#EDE9FE; --body-text-color-subdued:#B4AEDD; --block-label-text-color:#C4B5FD;
  --block-label-background-fill:#1B1442; --block-title-text-color:#C4B5FD; --input-background-fill:#1B1442;
  --button-secondary-background-fill:#241A55; --button-secondary-text-color:#EDE9FE; --slider-color:#8B5CF6;
  background:radial-gradient(1200px 600px at 50% -10%, #3B1D8F55, transparent), #0B0820 !important;
  color:#EDE9FE !important; font-family:Poppins,'Segoe UI',sans-serif;
}
/* nav */
#nav {align-items:center; padding:6px 8px; gap:4px;}
.brand {display:flex; align-items:center; gap:10px; font-weight:800; font-size:20px; color:#fff;}
.navlink {background:transparent !important; border:0 !important; box-shadow:none !important; color:#C4B5FD !important; font-weight:600 !important;}
.navlink:hover {color:#fff !important;}
/* buttons */
.cta {background:linear-gradient(90deg,#7C3AED,#A855F7) !important; color:#fff !important; border:0 !important; height:60px; font-size:18px !important;
  font-weight:700 !important; border-radius:999px !important; box-shadow:0 8px 30px rgba(124,58,237,.5);}
.ghost {background:transparent !important; color:#EDE9FE !important; border:2px solid #5B45B8 !important; height:60px; font-size:18px !important;
  font-weight:700 !important; border-radius:999px !important;}
button.primary {background:linear-gradient(90deg,#7C3AED,#A855F7) !important; color:#fff !important; border:0 !important; font-weight:700 !important;}
#cta-row {justify-content:center; gap:16px;}
/* hero */
.hero {position:relative; text-align:center; padding:24px 8px 8px; overflow:hidden;}
.badge {display:inline-block; padding:6px 18px; border-radius:999px; background:#1B1442; border:1px solid #4C3AA5; color:#C4B5FD; font-size:14px; font-weight:600;}
.hero h1 {font-size:60px; font-weight:800; color:#fff; margin:14px 0 6px; line-height:1.1;}
.grad {background:linear-gradient(90deg,#A78BFA,#C084FC); -webkit-background-clip:text; background-clip:text; color:transparent; -webkit-text-fill-color:transparent;}
.lead {font-size:21px; color:#EDE9FE; margin:6px 0 0;}
.sub {font-size:15px; color:#B4AEDD; margin:4px 0 0;}
.deco {position:absolute; top:90px; width:150px; height:210px; border-radius:36px; background:linear-gradient(160deg,#6D3FE0AA,#2A1A6644); border:1px solid #6D4FD855;}
.deco.l {left:-70px; transform:rotate(-14deg);} .deco.r {right:-70px; transform:rotate(14deg);}
.ring {position:absolute; left:50%; top:60px; width:760px; height:760px; margin-left:-380px; border-radius:50%; border:1px solid #6D4FD833; pointer-events:none;}
.stage {position:relative; display:inline-block; padding:36px 56px; margin:20px 0 8px;}
.c {position:absolute; width:38px; height:38px; border:3px solid #8B5CF6;}
.c.tl {top:0; left:0; border-right:0; border-bottom:0; border-top-left-radius:10px;}
.c.tr {top:0; right:0; border-left:0; border-bottom:0; border-top-right-radius:10px;}
.c.bl {bottom:0; left:0; border-right:0; border-top:0; border-bottom-left-radius:10px;}
.c.br {bottom:0; right:0; border-left:0; border-top:0; border-bottom-right-radius:10px;}
.cards {display:flex; align-items:center; justify-content:center; gap:14px; flex-wrap:wrap;}
.gcard {width:180px; height:200px; border-radius:26px; background:linear-gradient(160deg,#5B34C855,#1A1048DD); border:1px solid #7C5FE0AA;
  display:flex; flex-direction:column; align-items:center; justify-content:center; gap:10px; color:#fff; font-size:22px;}
.gcard.tilt-l {transform:rotate(-6deg);} .gcard.tilt-r {transform:rotate(6deg);}
.gcard.mid {width:210px; height:236px; border:2px solid #C4B5FD; box-shadow:0 0 50px rgba(139,92,246,.6); background:linear-gradient(160deg,#7C3AEDAA,#2A1A78EE);}
.features {display:flex; justify-content:space-around; gap:20px; flex-wrap:wrap; margin:20px 0; text-align:left;}
.feat {display:flex; align-items:center; gap:14px;}
.feat b {display:block; color:#fff; font-size:16px;} .feat span {color:#B4AEDD; font-size:13px;}
.tile {width:56px; height:56px; border-radius:16px; background:#2A1B66; display:flex; align-items:center; justify-content:center;}
/* cards */
.card {background:rgba(30,20,80,.55); border:1px solid rgba(167,139,250,.3); border-radius:18px; padding:16px; text-align:center;}
.pname {font-size:13px; color:#EDE9FE;} .gesture {font-size:24px; font-weight:800; margin-top:2px;}
.result {font-size:24px; font-weight:800; color:#C4B5FD;} .small {font-size:13px; color:#B4AEDD;}
.score {display:flex; gap:12px; justify-content:center; font-size:32px; font-weight:800;}
.steps {display:flex; gap:16px; margin-top:24px; flex-wrap:wrap;}
.steps .card {flex:1; min-width:200px; display:flex; flex-direction:column; gap:8px;} .steps .card span {color:#B4AEDD; font-size:14px;}
.num {width:38px; height:38px; border-radius:50%; background:linear-gradient(90deg,#7C3AED,#A855F7); color:#fff; font-weight:800; display:flex; align-items:center; justify-content:center; margin:0 auto;}
#title {text-align:center; font-size:38px; font-weight:800; color:#fff; margin:4px 0 0;}
#sub {text-align:center; color:#B4AEDD; margin-bottom:12px;}
/* settings drawer */
#drawer {position:fixed !important; top:0; right:0; height:100vh; width:min(340px,90vw); z-index:1000; background:#120C2E; border-left:1px solid #3B2F7A;
  padding:24px; box-shadow:-20px 0 60px rgba(0,0,0,.55); overflow:auto; gap:16px;}
"""


# ---------- game logic ----------
def detect_players(img, conf):
    """Return (annotated_image, [(gesture, confidence), (gesture, confidence)]) left-to-right."""
    bgr = img[:, :, ::-1]  # Gradio gives RGB, ultralytics expects BGR for numpy arrays
    r = model.predict(bgr, conf=conf, agnostic_nms=True, verbose=False)[0]
    annotated = r.plot()[:, :, ::-1]  # plot() returns BGR -> back to RGB
    hands = []
    for b in r.boxes:
        x1, _, x2, _ = b.xyxy[0].tolist()
        hands.append(((x1 + x2) / 2, model.names[int(b.cls)], float(b.conf)))
    hands = sorted(hands, key=lambda h: -h[2])[:2]   # keep the 2 most confident
    hands = sorted(hands, key=lambda h: h[0])        # left -> right
    return annotated, [(g, c) for _, g, c in hands]


def judge(g1, g2):
    if g1 == g2:
        return 0, "It's a tie!", f"{g1.capitalize()} vs {g2.capitalize()}"
    if BEATS[g1] == g2:
        return 1, "Player 1 Wins!", f"{g1.capitalize()} beats {g2.capitalize()}"
    return 2, "Player 2 Wins!", f"{g2.capitalize()} beats {g1.capitalize()}"


def card(label, hand):
    if hand is None:
        return f'<div class="card"><div class="pname">{label}</div><div class="gesture">-</div></div>'
    g, c = hand
    pic = icon(g, 56) if g in BEATS else ""
    return (f'<div class="card"><div class="pname">{label}</div>{pic}'
            f'<div class="gesture" style="color:{COLORS.get(g, "#EDE9FE")}">{g.capitalize()}</div>'
            f'<div class="small">{c:.0%}</div></div>')


def score_html(score):
    return (f'<div class="card"><div class="pname">Score</div><div class="score">'
            f'<span style="color:#F9A8D4">{score[0]}</span><span>-</span>'
            f'<span style="color:#A5B4FC">{score[1]}</span></div></div>')


def result_html(title, sub=""):
    return f'<div class="card"><div class="result">{title}</div><div class="small">{sub}</div></div>'


def play(img, mode, conf, score):
    conf = conf if mode == ADVANCED else GAME_CONF  # slider only counts in Advanced Settings
    if img is None:
        return None, card("Player 1", None), card("Player 2", None), result_html("Add an image first"), score, score_html(score)
    annotated, hands = detect_players(img, conf)
    if len(hands) < 2:
        msg = "Show both hands in the picture" if hands else "No hands detected"
        p1 = card("Player 1", hands[0] if hands else None)
        return annotated, p1, card("Player 2", None), result_html(msg, "Try better lighting, or lower the confidence in Settings."), score, score_html(score)
    winner, title, sub = judge(hands[0][0], hands[1][0])
    score = list(score)
    if winner:
        score[winner - 1] += 1
    return annotated, card("Player 1", hands[0]), card("Player 2", hands[1]), result_html(title, sub), score, score_html(score)


def reset():
    return None, card("Player 1", None), card("Player 2", None), result_html("Ready!"), [0, 0], score_html([0, 0])


def show(page):
    return (gr.update(visible=page == "home"), gr.update(visible=page == "how"),
            gr.update(visible=page == "about"), gr.update(visible=page == "game"), gr.update(visible=False))


def toggle_slider(mode):
    return gr.update(visible=mode == ADVANCED)


# ---------- UI ----------
with gr.Blocks(css=CSS, title="RPS Vision - Rock Paper Scissors") as demo:
    # Top navigation (all pages)
    with gr.Row(elem_id="nav"):
        gr.HTML(NAV_BRAND)
        nav_home = gr.Button("Home", elem_classes="navlink", scale=0, min_width=100)
        nav_how = gr.Button("How It Works", elem_classes="navlink", scale=0, min_width=130)
        nav_about = gr.Button("About", elem_classes="navlink", scale=0, min_width=100)

    # Home
    with gr.Column(visible=True) as home:
        gr.HTML(HERO)
        with gr.Row(elem_id="cta-row"):
            play_btn = gr.Button("Play Game", elem_classes="cta", scale=0, min_width=280)
            how_btn = gr.Button("How It Works", elem_classes="ghost", scale=0, min_width=240)
        gr.HTML(FEATURES)

    # How it works
    with gr.Column(visible=False) as how:
        gr.HTML(HOW)
        with gr.Row(elem_id="cta-row"):
            play_btn2 = gr.Button("Play Game", elem_classes="cta", scale=0, min_width=280)

    # About
    with gr.Column(visible=False) as about:
        gr.HTML(ABOUT)

    # Game
    with gr.Column(visible=False) as game:
        with gr.Row():
            gr.HTML('<div id="title">Rock Paper <span class="grad">Scissors</span></div>'
                    '<div id="sub">Show your hand · Let AI detect · See who wins!</div>')
            settings_btn = gr.Button("Settings", elem_classes="ghost", scale=0, min_width=140)
        score = gr.State([0, 0])
        with gr.Row():
            with gr.Column(scale=3):
                inp = gr.Image(sources=["webcam", "upload"], type="numpy", label="Camera / Upload (Player 1 = left, Player 2 = right)")
                with gr.Row():
                    go = gr.Button("Play", variant="primary")
                    again = gr.Button("Reset Game")
            with gr.Column(scale=3):
                out_img = gr.Image(label="Detection", interactive=False)
                with gr.Row():
                    p1 = gr.HTML(card("Player 1", None))
                    p2 = gr.HTML(card("Player 2", None))
            with gr.Column(scale=2):
                score_box = gr.HTML(score_html([0, 0]))
                res = gr.HTML(result_html("Ready!"))

    # Settings drawer (hidden until the Settings button is pressed)
    with gr.Column(visible=False, elem_id="drawer") as drawer:
        gr.HTML('<div style="font-size:20px;font-weight:800;color:#fff">Settings</div>')
        mode = gr.Radio([GAME_MODE, ADVANCED], value=GAME_MODE, label="Mode",
                        info="Game Mode: confidence is fixed at 0.5. Advanced Settings: choose it yourself.")
        conf = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Confidence threshold", visible=False)
        close_btn = gr.Button("Close", elem_classes="ghost")

    pages = [home, how, about, game, drawer]
    for b in (nav_home,):
        b.click(lambda: show("home"), None, pages)
    for b in (nav_how, how_btn):
        b.click(lambda: show("how"), None, pages)
    nav_about.click(lambda: show("about"), None, pages)
    for b in (play_btn, play_btn2):
        b.click(lambda: show("game"), None, pages)

    settings_btn.click(lambda: gr.update(visible=True), None, drawer)
    close_btn.click(lambda: gr.update(visible=False), None, drawer)
    mode.change(toggle_slider, mode, conf)

    go.click(play, [inp, mode, conf, score], [out_img, p1, p2, res, score, score_box])
    again.click(reset, None, [out_img, p1, p2, res, score, score_box])

if __name__ == "__main__":
    demo.launch() ايش يسوي الموقع؟
