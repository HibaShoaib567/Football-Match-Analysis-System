
import os
import uuid
import streamlit as st
import pandas as pd

from analytics import process_video


st.set_page_config(
    page_title="Football Analytics Dashboard",
    page_icon="⚽",
    layout="wide"
)


# ==========================================
# HEADER
# ==========================================

st.title("⚽ Football Analytics Dashboard")

st.markdown(
    "### AI-Powered Football Match Analysis"
)

st.caption(
    "YOLOv8 | ByteTrack | K-Means | Player Performance"
)

st.divider()


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("Match Settings")

model_path = st.sidebar.text_input(
    "YOLO Model Path",
    value="best.pt"
)

confidence = st.sidebar.slider(
    "Detection Confidence",
    min_value=0.10,
    max_value=0.90,
    value=0.25,
    step=0.05
)


# ==========================================
# VIDEO UPLOAD
# ==========================================

st.subheader("Upload Football Match")

uploaded_video = st.file_uploader(
    "Choose your football video",
    type=["mp4", "avi", "mov", "mkv"]
)


if uploaded_video:

    st.video(uploaded_video)

    st.info("Video uploaded successfully!")

    if st.button(
        "Start Match Analysis",
        type="primary",
        use_container_width=True
    ):

        if not os.path.exists(model_path):

            st.error("YOLO model file not found!")

        else:

            os.makedirs("outputs", exist_ok=True)

            video_name = (
                "input_" + uuid.uuid4().hex[:8] + ".avi"
            )

            output_name = (
                "processed_" + uuid.uuid4().hex[:8] + ".mp4"
            )

            video_path = os.path.join(
                "outputs", video_name
            )

            output_path = os.path.join(
                "outputs", output_name
            )

            with open(video_path, "wb") as f:
                f.write(uploaded_video.getbuffer())

            progress = st.progress(0)

            status = st.empty()

            try:

                status.info(
                    "Analyzing video and collecting jersey colors..."
                )

                result = process_video(
                    video_path=video_path,
                    model_path=model_path,
                    output_path=output_path,
                    confidence=confidence,
                    progress_callback=lambda x: (
                        progress.progress(
                            min(int(x * 100), 100)
                        )
                    )
                )

                progress.progress(100)

                status.success(
                    "Football analysis completed!"
                )

                st.session_state["football_result"] = result

                st.session_state["output_path"] = output_path

            except Exception as e:

                status.error("Analysis failed!")

                st.exception(e)


# ==========================================
# DISPLAY ANALYTICS
# ==========================================

if "football_result" in st.session_state:

    result = st.session_state["football_result"]

    players = result["players"]
    teams = result["teams"]

    st.divider()

    st.header("📊 Match Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Unique Players",
        players["Player_ID"].nunique()
    )

    c2.metric(
        "Total Frames",
        result["total_frames"]
    )

    c3.metric(
        "Ball Detections",
        result["ball_detections"]
    )

    c4.metric(
        "Referee Detections",
        result["referee_detections"]
    )

    # ======================================
    # TEAM STATISTICS
    # ======================================

    st.divider()

    st.header("🏆 Team Statistics")

    st.dataframe(
        teams,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Estimated Team Possession")

    st.bar_chart(
        teams.set_index("Team")["Possession_Percent"]
    )

    # ======================================
    # PLAYER PERFORMANCE
    # ======================================

    st.divider()

    st.header("👤 Player Performance")

    if not players.empty:

        st.dataframe(
            players,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Player Movement Comparison")

        st.bar_chart(
            players.set_index("Player_ID")[
                "Movement_Pixels"
            ]
        )

    else:

        st.warning("No tracked players found.")

    # ======================================
    # PROCESSED VIDEO
    # ======================================

    st.divider()

    st.header("🎥 Processed Match Video")

    output_path = st.session_state["output_path"]

    if os.path.exists(output_path):

        st.video(output_path)

        with open(output_path, "rb") as video_file:

            st.download_button(
                "Download Processed Video",
                data=video_file.read(),
                file_name="football_analysis.mp4",
                mime="video/mp4"
            )

    # ======================================
    # CSV DOWNLOADS
    # ======================================

    st.divider()

    st.header("📥 Download Match Reports")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.download_button(
            "Player Statistics CSV",
            players.to_csv(index=False),
            file_name="player_statistics.csv",
            mime="text/csv"
        )

    with col2:

        st.download_button(
            "Team Statistics CSV",
            teams.to_csv(index=False),
            file_name="team_statistics.csv",
            mime="text/csv"
        )

    with col3:

        tracking = result["tracking"]

        st.download_button(
            "Player Tracking CSV",
            tracking.to_csv(index=False),
            file_name="player_tracking.csv",
            mime="text/csv"
        )
