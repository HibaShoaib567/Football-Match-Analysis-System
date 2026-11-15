
import os
import uuid
import subprocess
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt

from analytics import process_video

PROJECT_DIR = "/content/drive/MyDrive/Football_Project"
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
MODEL_PATH = os.path.join(PROJECT_DIR, "best.pt")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def football_analysis(video, confidence, progress=gr.Progress()):

    if video is None:
        raise gr.Error("Please upload a football video.")

    if not os.path.exists(MODEL_PATH):
        raise gr.Error("best.pt model file not found!")

    video_path = video

    if not os.path.exists(video_path):
        raise gr.Error("Uploaded video not found!")

    unique_id = uuid.uuid4().hex[:8]

    output_path = os.path.join(
        OUTPUT_DIR,
        f"processed_{unique_id}.mp4"
    )

    try:
        result = process_video(
            video_path=video_path,
            model_path=MODEL_PATH,
            output_path=output_path,
            progress_callback=lambda x: progress(
                x, desc="Analyzing football match..."
            ),
            confidence=confidence
        )

        players_df = result["players"]
        teams_df = result["teams"]
        tracking_df = result["tracking"]

        # Save CSV reports
        player_csv = os.path.join(
            OUTPUT_DIR, f"player_statistics_{unique_id}.csv"
        )
        team_csv = os.path.join(
            OUTPUT_DIR, f"team_statistics_{unique_id}.csv"
        )
        tracking_csv = os.path.join(
            OUTPUT_DIR, f"player_tracking_{unique_id}.csv"
        )

        players_df.to_csv(player_csv, index=False)
        teams_df.to_csv(team_csv, index=False)
        tracking_df.to_csv(tracking_csv, index=False)

        # Convert processed video for browser compatibility
        browser_video = os.path.join(
            OUTPUT_DIR, f"browser_{unique_id}.mp4"
        )

        try:
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", output_path,
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    "-an",
                    browser_video
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            if os.path.exists(browser_video):
                output_video = browser_video
            else:
                output_video = output_path

        except Exception:
            output_video = output_path

        # Match overview
        unique_players = (
            players_df["Player_ID"].nunique()
            if not players_df.empty else 0
        )

        summary = f"""
FOOTBALL MATCH ANALYSIS COMPLETED

Total Frames: {result['total_frames']}

Unique Players: {unique_players}

Ball Detections: {result['ball_detections']}

Referee Detections: {result['referee_detections']}

Team Classification: Completed

Processed Video: Ready
"""

        # Player movement chart
        if not players_df.empty:

            chart_df = players_df.nlargest(
                15, "Movement_Pixels"
            ).sort_values("Movement_Pixels")

            fig1, ax1 = plt.subplots(figsize=(9, 5))

            ax1.barh(
                chart_df["Player_ID"].astype(str),
                chart_df["Movement_Pixels"]
            )

            ax1.set_title("Top 15 Players - Movement")
            ax1.set_xlabel("Movement (Pixels)")
            ax1.set_ylabel("Player ID")

            fig1.tight_layout()

        else:
            fig1 = None

        # Team possession chart
        if not teams_df.empty:

            fig2, ax2 = plt.subplots(figsize=(7, 4))

            ax2.bar(
                teams_df["Team"],
                teams_df["Possession_Percent"]
            )

            ax2.set_title("Team Possession Analysis")
            ax2.set_ylabel("Possession (%)")
            ax2.set_ylim(0, 100)

            fig2.tight_layout()

        else:
            fig2 = None

        return (
            output_video,
            summary,
            players_df,
            teams_df,
            fig1,
            fig2,
            player_csv,
            team_csv,
            tracking_csv
        )

    except Exception as e:
        raise gr.Error(f"Analysis failed: {str(e)}")


with gr.Blocks(
    theme=gr.themes.Soft(),
    title="Football Analytics Dashboard"
) as demo:

    gr.Markdown(
        """
        # ⚽ Football Analytics Dashboard

        ### AI-Powered Football Match Analysis

        Player Detection | Team Classification |
        Ball Detection | Performance Analytics
        """
    )

    with gr.Row():

        with gr.Column():

            video_input = gr.Video(
                label="Upload Football Match Video"
            )

            confidence = gr.Slider(
                minimum=0.10,
                maximum=0.90,
                value=0.25,
                step=0.05,
                label="Detection Confidence"
            )

            analyze_btn = gr.Button(
                "Start Football Analysis",
                variant="primary"
            )

        with gr.Column():

            output_video = gr.Video(
                label="Processed Match Video"
            )

            summary = gr.Textbox(
                label="Match Overview",
                lines=9,
                interactive=False
            )

    gr.Markdown("## 📊 Player Performance Statistics")

    player_table = gr.Dataframe(
        label="Player Statistics",
        interactive=False
    )

    with gr.Row():

        player_chart = gr.Plot(
            label="Player Movement Analysis"
        )

        team_chart = gr.Plot(
            label="Team Possession Analysis"
        )

    gr.Markdown("## 🏆 Team Statistics")

    team_table = gr.Dataframe(
        label="Team Classification and Possession",
        interactive=False
    )

    gr.Markdown("## 📥 Download Match Reports")

    with gr.Row():

        player_download = gr.File(
            label="Player Statistics CSV"
        )

        team_download = gr.File(
            label="Team Statistics CSV"
        )

        tracking_download = gr.File(
            label="Player Tracking CSV"
        )

    analyze_btn.click(
        fn=football_analysis,
        inputs=[
            video_input,
            confidence
        ],
        outputs=[
            output_video,
            summary,
            player_table,
            team_table,
            player_chart,
            team_chart,
            player_download,
            team_download,
            tracking_download
        ]
    )


demo.launch(
    share=True,
    debug=False,
    server_name="0.0.0.0"
)
