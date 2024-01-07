import streamlit as st
import subprocess
from streamlit_option_menu import option_menu
import os

def custom_title(title_text, font_color='#700107'):
    title_html = f'<h1 style="color: {font_color}; margin-bottom: 5px;">{title_text}</h1>'
    st.markdown(title_html, unsafe_allow_html=True)

def custom_header(header_text, font_color='#002048', font_size='24px', line_height='0.05'):
    header_html = f'<h2 style="color: {font_color}; font-size: {font_size}; line-height: {line_height}; margin: 0;">{header_text}</h2>'
    st.markdown(header_html, unsafe_allow_html=True)

def custom_text(text, font_color='#002048', font_size='18px', font_weight='bold'):
    text_html = f'<p style="color: {font_color}; font-size: {font_size}; font-weight: {font_weight}; margin: 0;">{text}</p>'
    st.markdown(text_html, unsafe_allow_html=True)

with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "About", "Team"]
    )

# Add your logo here
logo = st.image("https://media.licdn.com/dms/image/C4D16AQHK9OSLGSDZvQ/profile-displaybackgroundimage-shrink_200_800/0/1612424173414?e=2147483647&v=beta&t=aFHYrrBW0rhbpQBLNEiKYG9oDUbMndtzxncOqOhg4L8", width=150)

if selected == "Home":
    # Get all attendee names from the 'faces' folder
    path = 'AIML PHOTOS'
    all_attendees = {os.path.splitext(filename)[0] for filename in os.listdir(path)}

    custom_title("Welcome to AU Attendance App")
    custom_header("An Automated Facial Recognition Attendance System", font_size='28px')

    signal_file_path = 'run_signal.txt'

    def is_running():
        if os.path.exists(signal_file_path):
            with open(signal_file_path, 'r') as file:
                return file.read().strip() == 'run'
        return False

    if not is_running() and st.button("Start Marking Attendance"):
        with open(signal_file_path, 'w') as file:
            file.write('run')
        subprocess.Popen(["python", "main.py"])

    if is_running() and st.button("Stop Marking Attendance"):
        with open(signal_file_path, 'w') as file:
            file.write('stop')

    if os.path.exists("current_attendance.txt"):
        with open("current_attendance.txt", "r") as file:
            present_attendees = sorted([line.strip() for line in file])
            absent_attendees = sorted(list(all_attendees - set(present_attendees)))

        st.subheader("Present Attendees")
        st.write(", ".join(present_attendees))

        st.subheader("Absent Attendees")
        st.write(", ".join(absent_attendees))

    st.markdown("©2020 Anurag University.")

if selected == "About":
    custom_title("About FaceSense")
    custom_text("FaceSense redefines attendance management in educational institutions by leveraging the power of deep learning and the FaceNet model. This innovative app automates attendance tracking through precise facial recognition, eliminating errors and enabling real-time monitoring. Modernizing traditional methods, FaceSense epitomizes the fusion of technology and education, ushering in an era of efficient and accurate attendance management.")

if selected == "Team":
    custom_title("Batch No-15 : AIML")
    custom_text("20EG107126 - HEMANTH SAI<br>"
                "20EG107127 - K VENKATA KAVYA<br>"
                "20EG107147 - S NAMITH")

page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
opacity: 1.0;
}

[data-testid="stSidebar"] {
background-image: url("https://img.freepik.com/free-vector/confirmed-attendance-concept-illustration_114360-7495.jpg?w=740&t=st=1697376996~exp=1697377596~hmac=7f8292a6a5769306d55726206dff203807331a792c3509b7ba6dfa1d0631bc7f");
background-position: center;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)