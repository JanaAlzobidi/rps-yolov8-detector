"""Rock Paper Scissors game with a YOLOv8 detector, served with Gradio.

Two players show their hands to the camera. The model finds both hands,
reads each gesture, and announces the winner.

    Player 1 = the hand on the LEFT of the image
    Player 2 = the hand on the RIGHT of the image

Tabs:
    Play   - upload a photo or take one with the webcam
    Video  - upload a short video; every frame is annotated (bonus)

Run locally:
    python app.py                      # uses weights/best.pt
    python app.py --weights path/to/best.pt --share

In Google Colab:
    !python app.py --share             # prints a public gradio.live link
"""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import gradio as gr

from rps import (
    DEFAULT_WEIGHTS,
    add_verdict_banner,
    bgr_to_rgb,
    load_model,
    model_imgsz,
    play,
)

EXAMPLES_DIR = Path(__file__).parent / "examples"
MAX_VIDEO_SECONDS = 30  # longer videos are cut, to keep processing time reasonable


def parse_args():
    p = argparse.ArgumentParser(description="Rock Paper Scissors YOLOv8 web app")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="path to the trained best.pt")
    p.add_argument("--conf", type=float, default=0.5, help="default confidence threshold")
    p.add_argument("--share", action="store_true", help="create a public gradio.live link")
    p.add_argument("--port", type=int, default=7860)
    return p.parse_args()


def ffmpeg_path():
    """Find an ffmpeg binary: the system one, or the one bundled with imageio-ffmpeg."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def build_app(model, default_conf):
    imgsz = model_imgsz(model)

    # ---------- Play tab ----------
    def run_image(image, conf):
        if image is None:
            return None, "Upload a photo or take one with the webcam.", []
        annotated, rnd = play(model, image, conf=conf, imgsz=imgsz)
        rows = []
        for hand in rnd.hands:
            role = "Player 1" if hand is rnd.player1 else "Player 2" if hand is rnd.player2 else "-"
            rows.append([role, hand.gesture, round(hand.confidence, 2)])
        return bgr_to_rgb(annotated), rnd.verdict, rows

    # ---------- Video tab ----------
    def run_video(video_path, conf, progress=gr.Progress()):
        if video_path is None:
            raise gr.Error("Upload a video first.")
        ffmpeg = ffmpeg_path()
        if ffmpeg is None:
            raise gr.Error("ffmpeg not found. Install it, or run: pip install imageio-ffmpeg")

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        max_frames = min(total, int(fps * MAX_VIDEO_SECONDS))

        tmp = Path(tempfile.mkdtemp())
        raw_path, out_path = tmp / "raw.mp4", tmp / "result.mp4"
        writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        frames = 0
        while frames < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            annotated, rnd = play(model, frame, conf=conf, imgsz=imgsz)
            writer.write(add_verdict_banner(annotated, rnd.verdict))
            frames += 1
            progress(frames / max_frames, desc=f"Frame {frames}/{max_frames}")
        cap.release()
        writer.release()

        if frames == 0:
            raise gr.Error("Could not read any frames from this video.")

        # Browsers cannot play OpenCV's mp4v output, so re-encode to H.264.
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", str(raw_path),
             "-vcodec", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(out_path)],
            check=True,
        )
        if total > max_frames:
            gr.Info(f"Only the first {MAX_VIDEO_SECONDS} seconds were processed.")
        return str(out_path)

    # ---------- Layout ----------
    with gr.Blocks(title="Rock Paper Scissors") as demo:
        gr.Markdown(
            "# Rock Paper Scissors\n"
            "Show two hands to the camera. **Left hand = Player 1**, **right hand = Player 2**.\n\n"
            "Raise the confidence threshold if the model draws boxes on things that are not hands; "
            "lower it if it misses a hand."
        )
        conf = gr.Slider(0.1, 0.9, value=default_conf, step=0.05, label="Confidence threshold")

        with gr.Tab("Play"):
            with gr.Row():
                img_in = gr.Image(type="pil", sources=["upload", "webcam"], label="Photo")
                img_out = gr.Image(label="Detections")
            verdict = gr.Textbox(label="Result")
            hands_table = gr.Dataframe(
                headers=["Player", "Gesture", "Confidence"],
                label="Detected hands (left to right)",
                interactive=False,
            )
            play_btn = gr.Button("Play", variant="primary")
            play_btn.click(run_image, [img_in, conf], [img_out, verdict, hands_table])

            examples = sorted(p for p in EXAMPLES_DIR.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
            if examples:
                gr.Examples([[str(p)] for p in examples], inputs=img_in, label="Example photos")

        with gr.Tab("Video (bonus)"):
            gr.Markdown(f"Keep videos short. Only the first {MAX_VIDEO_SECONDS} seconds are processed.")
            with gr.Row():
                vid_in = gr.Video(label="Video")
                vid_out = gr.Video(label="Detections")
            gr.Button("Detect", variant="primary").click(run_video, [vid_in, conf], vid_out)

    return demo


def main():
    args = parse_args()
    model = load_model(args.weights)
    print(f"Loaded {args.weights} | classes: {list(model.names.values())} | imgsz: {model_imgsz(model)}")
    build_app(model, args.conf).launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
