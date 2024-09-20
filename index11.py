import requests
from io import BytesIO
import streamlit as st
import base64
from streamlit_option_menu import option_menu
import cv2
import numpy as np
from PIL import Image
import pickle
import face_recognition
import cvzone
import os
import mediapipe as mp  # Import MediaPipe for face landmarks
import pandas as pd
from datetime import datetime

from joblib import load






st.set_page_config(
    page_icon="Logo4.png",
    page_title="DFAS",
    layout="wide"
)



    # Load data
data = pd.read_csv("database.csv")
data.drop("Unnamed: 0", axis=1, inplace=True)
data["Enrollment_id"] = data["Enrollment_id"].astype(str)


# Function to convert an image file to base64 encoding
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# # Convert the image "13.jpg" to base64 encoding
# img = get_img_as_base64("13.jpg")

# Convert the local image "HERO.jpg" to base64 encoding
# Background_img = get_img_as_base64("bbb.jpg")  # Replace with your local image path



# Convert the local images to base64 encoding
background_img1 = get_img_as_base64("bg4.jpg")  # Replace with your local image path


# # CSS styling for the Streamlit app with multiple background images
# page_bg_img = f"""
# <style>
# [data-testid="stAppViewContainer"] > .main {{
#     background-image: 
#         url("data:image/jpeg;base64,{background_img1}"),
#         url("data:image/jpeg;base64,{background_img2}");
#     background-size: cover, cover;
#     background-position: center, center;
#     background-repeat: no-repeat, no-repeat;
#     background-attachment: local, local;
# }}

# CSS styling for the Streamlit app
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
    # background-image: url("data:image/jpeg;base64,{"Background_img"}");
        background-image: 
        url("data:image/jpeg;base64,{background_img1}");

    background-size: 100%;
    background-position: top left;
    # background-repeat: no-repeat;
    # background-attachment: local;
    # opacity: 0.3;
    # transition: opacity 2s ease-in-out; /* 2 seconds transition */
}}
[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}
[data-testid="stSidebar"] > div:first-child {{
    background-repeat: no-repeat;
    background-attachment: fixed;
    # background: rgb(18 18 18 / 0%);
    background: #0d425d;
}}


.st-emotion-cache-1gv3huu {{
    position: relative;
    top: 2px;
    background-color: #000;
    z-index: 999991;
    min-width: 244px;
    max-width: 550px;
    transform: none;
    transition: transform 300ms, min-width 300ms, max-width 300ms;
}}

.st-emotion-cache-1jicfl2 {{
    width: 100%;
    padding: 4rem 1rem 4rem;
    min-width: auto;
    max-width: initial;

}}


.st-emotion-cache-4uzi61 {{
    border: 1px solid rgba(49, 51, 63, 0.2);
    border-radius: 0.5rem;
    padding: calc(-1px + 1rem);
    background: rgb(240 242 246);
    box-shadow: 0 5px 8px #6c757d;
}}

.st-emotion-cache-1vt4y43 {{
    display: inline-flex;
    -webkit-box-align: center;
    align-items: center;
    -webkit-box-pack: center;
    justify-content: center;
    font-weight: 400;
    padding: 0.25rem 0.75rem;
    border-radius: 0.5rem;
    min-height: 2.5rem;
    margin: 0px;
    line-height: 1.6;
    color: inherit;
    width: auto;
    COLOR: WHITE;
    user-select: none;
    background-color: #0461f1;
    border: 1px solid rgba(49, 51, 63, 0.2);
}}

.st-emotion-cache-qcpnpn {{
    border: 1px solid rgb(163, 168, 184);
    border-radius: 0.5rem;
    # padding: calc(-1px + 1rem);
    padding: calc(40px + 0rem);
    background-color: rgb(38, 39, 48);
    MARGIN-TOP: 9PX;
    box-shadow: 0 5px 8px #6c757d;


}}




.st-emotion-cache-15hul6a {{
    user-select: none;
    background-color: #ffc107;
    border: 1px solid rgba(250, 250, 250, 0.2);
    
}}

.st-emotion-cache-1hskohh {{
    margin: 0px;
    padding-right: 2.75rem;
    color: rgb(250, 250, 250);
    border-radius: 0.5rem;
    background: #000;
}}

.st-emotion-cache-12pd2es {{
    margin: 0px;
    padding-right: 2.75rem;
    color: #f0f2f6;
    border-radius: 0.5rem;
    background: #000;
}}

.st-emotion-cache-1r6slb0 {{
    width: calc(33.3333% - 1rem);
    flex: 1 1 calc(33.3333% - 1rem);
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
}}
.st-emotion-cache-12w0qpk {{
    width: calc(25% - 1rem);
    flex: 1 1 calc(25% - 1rem);
    display: flex;
    flex-direction: row;
    justify-content: CENTER;
    ALIGN-ITEMS: CENTER;
}}



.st-emotion-cache-1kyxreq {{
    display: flex;
    flex-flow: wrap;
    row-gap: 1rem;
    align-items: center;
    justify-content: center;
}}

img {{
    vertical-align: middle;
    border-radius: 10px;
 
}}


    h5 {{
    font-family: "Source Sans Pro", sans-serif;
    font-weight: 600;
    color: rgb(14 14 14);
    padding: 0px 0px 1rem;
    margin: 0px;
    line-height: 1.2;
}}



div[data-baseweb="tab-list"] {{
    background-color: #00BCD4;
    padding: 5px;
    border-radius: 3px;
}}


div[data-baseweb="tab-list"] button[aria-selected="true"] {{
    background-color: #0008ff;
    color: white;
    border-radius: 20px;
    padding: 22px;
    border: none;
}}

div[data-baseweb="tab-list"] button:hover {{
    background-color: #ffcc00;
    color: white;
    border-radius: 20px;
    padding: 22px;
}}



.st-d4 {{
 background-color: #00BCD4; 
}}

# .st-dh {{
#     padding-top: 2rem;
#     padding: 40px;
#     # background: #1191bd;
#     background: white;
#     border-radius: 10px;
#     margin-top: 16px;
#     # color:white;
# }}

.st-dh {{
    padding-top: 1rem;
    margin-top: 10px;
    padding: 25px;
    background: white;
    border-radius: 10px;
    color:black;

}}

h1 {{
    font-family: "Source Sans Pro", sans-serif;
    font-weight: 700;
    color: #2edcf2;
    padding: 1.25rem 0px 1rem;
    margin: 0px;
    line-height: 1.2;
}}


.st-emotion-cache-13n2bn5 {{
    position: relative;
    overflow: hidden;
    width: 80%;
    margin-left: 74px;

    object-fit: contain;
}}

.st-emotion-cache-6nrhu6 {{
    position: relative;
    display: inline-block;
    # margin-left: 90px;
     box-shadow: 0 5px 8px #6c757d;
}}


.st-emotion-cache-bcargt {{
    position: relative;
    display: inline-flex;
    flex-direction: column;
    -webkit-box-align: center;
    align-items: center;
    -webkit-box-pack: center;
    justify-content: center;
    background-color: #0281f0;
    border: 1px solid rgba(250, 250, 250, 0.2);
    border-radius: 0px 0px 0.5rem 0.5rem;
    font-weight: 400;
    padding: 0.375rem 0.75rem;
    margin: 0px;
    line-height: 1.6;
    color: inherit;
    width: 100%;
    user-select: none;
}}


.st-emotion-cache-ltfnpr {{
    font-family: "Source Sans Pro", sans-serif;
    font-size: 17px;
    color: #f1ba05;
    text-align: center;
    margin-top: 0.375rem;
    overflow-wrap: break-word;
    padding: 0.125rem;
}}




</style>
"""

# Apply CSS styling to the Streamlit app
st.markdown(page_bg_img, unsafe_allow_html=True)









# Sidebar configuration
with st.sidebar:
    # Display logo image
    st.image("Logo5.png", use_column_width=True)

    # Adding a custom style with HTML and CSS for sidebar
    st.markdown("""
        <style>
            .custom-text {
                font-size: 20px;
                font-weight: bold;
                text-align: center;
                color:#ffc107
            }
            .custom-text span {
                color: #04ECF0; /* Color for the word 'Recommendation' */
            }
        </style>
    """, unsafe_allow_html=True)
  


  
    # Displaying the subheader with custom styling
    st.markdown('<p class="custom-text"> Digital Face <span>Attendance </span> System</p>', unsafe_allow_html=True)

    # HTML and CSS for the GitHub button
    github_button_html = """
    <div style="text-align: center; margin-top: 50px;">
        <a class="button" href="https://github.com/Salman7292" target="_blank" rel="noopener noreferrer">Visit my GitHub</a>
    </div>

    <style>
        /* Button styles */
        .button {
            display: inline-block;
            padding: 10px 20px;
            background-color: #ffc107;
            color: black;
            text-decoration: none;
            border-radius: 5px;
            text-align: center;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        .button:hover {
            background-color: #000345;
            color: white;
            text-decoration: none; /* Remove underline on hover */
        }
    </style>
    """

    # Display the GitHub button in the sidebar
    st.markdown(github_button_html, unsafe_allow_html=True)
    
    # Footer HTML and CSS
    footer_html = """
    <div style="padding:10px; text-align:center;margin-top: 10px;">
        <p style="font-size:20px; color:#ffffff;">Made with ❤️ by Salman Malik</p>
    </div>
    """

    # Display footer in the sidebar
    st.markdown(footer_html, unsafe_allow_html=True)


# Define the option menu for navigation
selections = option_menu(
    menu_title=None,
options = ['Home', 'Scan Your Face', 'Attendance Sheet', 'Enroll Student'],
icons = ['house-fill', 'camera-fill', 'file-earmark-spreadsheet-fill', 'person-plus-fill']  ,# Added icon for 'Enroll Student'



    menu_icon="cast",
    default_index=0,
    orientation='horizontal',
    styles={
        "container": {
            "padding": "5px 23px",
            "background-color": "#001c99",
            "border-radius": "8px",
            "box-shadow": "0px 4px 10px rgba(0, 0, 0, 0.25)"
        },
        "icon": {"color": "#f9fafb", "font-size": "18px"},
        "hr": {"color": "#0d6dfdbe"},
        "nav-link": {
            "color": "#f9fafb",
            "font-size": "15px",
            "text-align": "center",
            "margin": "0 10px",
            "--hover-color": "#0761e97e",
            "padding": "10px 10px",
            "border-radius": "16px"
        },
        "nav-link-selected": {"background-color": "#ffc107", "font-size": "12px"},
    }
)

if selections == "Home":
# Define HTML and CSS for the hero section
    code = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<style>
#  .hero-section {
#     padding: 40px 20px;
#     font-family: Arial, sans-serif;
#     BACKGROUND: WHITE;
#     BORDER-RADIUS: 10PX;
# }

.hero-section {
    padding: 40px 20px;
    font-family: Arial, sans-serif;
    background: rgba(0, 0, 0, 0.7); /* Black with 70% opacity for transparency */
    border-radius: 10px;
}


    .hero-heading {
        font-size: 2.5rem;
        margin-bottom: 20px;
        # color: #343a40;
            color: #feffffbd;
        font-family: 'Roboto', sans-serif;
        font-weight: 700;
    }
    .hero-text {
        font-size: 1.2rem;
        # line-height: 1.6;
        color: #6c757d;
        # max-width: 800px;
        # margin: 0 auto;
    }

    

    ul{
    margin: 0px 0px 1rem;
    padding: 0px;
    font-size: 1rem;
    font-weight: 400;
    # COLOR: black;
     color: #feffffbd;

    }
</style>
</head>
<body>
<section class="hero-section">
<div class="container">
<h1 class="hero-heading">Digital Face Attendance System</h1>
<p >
    Welcome to our Digital Face Attendance System. Automatically capture student attendance with advanced facial recognition technology. Simply scan the students' faces, and the system will mark their attendance seamlessly. Our system offers:
</p>
<ul class="features-list">
    <li><strong>Real-Time Face Detection:</strong> Instantly recognize and verify student identities through accurate face scanning.</li>
    <li><strong>Automated Attendance Marking:</strong> Eliminate manual processes by automatically recording attendance in the system.</li>
    <li><strong>Secure and Reliable:</strong> Ensure data privacy and reliability with state-of-the-art biometric technology.</li>
    <li><strong>Comprehensive Reporting:</strong> Generate detailed attendance reports for students, classes, and events.</li>
</ul>
<p >
    Empower your institution with our reliable and precise attendance tracking system, designed for efficiency, security, and accuracy.
</p>

</div>

</section>
</body>
</html>
"""





# Use Streamlit to display the HTML content
    st.markdown(code, unsafe_allow_html=True)

elif selections == "Scan Your Face":
    enroll_flag=0
    st.markdown(
        """
        <h1 style='text-align: center;'>Scan Your Face Here</h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <hr style="border: none; height: 2px;width: 50%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
        """,
        unsafe_allow_html=True
    )




  

    # Path to the folder containing images
    folder_path = "student"
    images = os.listdir(folder_path)
    img_path = [os.path.join(folder_path, i) for i in images]

    # st.write(f"Image paths: {img_path}")

    # # Load face encodings and metadata
    # with open("Encoding_file5.pkl", "rb") as file:
    #     encoding_known_list = pickle.load(file)
    #     encoding_known_list1, name_list, enrollment_id = encoding_known_list


   

    # Load face encodings and metadata
    encoding_known_list = load("Encoding_file5.pkl")
    encoding_known_list1, name_list, enrollment_id = encoding_known_list


    # st.write(f"Enrollment IDs: {enrollment_id}")

    # Initialize MediaPipe face mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=2, refine_landmarks=True)
    drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))



    
    # Use the camera_input widget to access the webcam
    image = st.camera_input(" ")

    if image:
        # Convert the uploaded image to an OpenCV format
        img = Image.open(image)
        img = np.array(img)

        # # Display the image
        # st.image(img, caption="Captured Image", use_column_width=True)

        # Resize and convert the image for face recognition and landmarks
        imgs = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)

        # Face recognition processing
        face_current_frame = face_recognition.face_locations(imgs)
        encoder_frame = face_recognition.face_encodings(imgs, face_current_frame)

        # MediaPipe face landmarks processing
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)

        # Draw landmarks if detected
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    image=img, 
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec
                )

        # Face recognition matching and attendance marking
        for encode_face, faceloc in zip(encoder_frame, face_current_frame):
            matches = face_recognition.compare_faces(encoding_known_list1, encode_face)
            facedis = face_recognition.face_distance(encoding_known_list1, encode_face)

            if len(facedis) > 0:
                min_index = np.argmin(facedis)

                if min_index < len(enrollment_id):  # Check if min_index is valid
                    top, right, bottom, left = faceloc
                    top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
                    corner_length = 30
                    corner_thickness = 5
                    cvzone.cornerRect(img, (left, top, right - left, bottom - top), l=corner_length, t=corner_thickness)

                    if matches[min_index]:
                        name = name_list[min_index]
                        enrollment = enrollment_id[min_index]
                        selected_image = img_path[min_index]
                        selected_image = cv2.imread(selected_image)


                        cv2.putText(img, f'{name}', (left-20, top+10 - 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

                        enrollment_id_str = str(enrollment)
                        current_date = datetime.now().strftime("%Y-%m-%d")
                        current_day = datetime.now().strftime("%A")
                        

                        filtered_data = data[(data['Enrollment_id'] == enrollment_id_str) & (data['Date'] == current_date)]

                        if not filtered_data.empty:
                            # st.warning(f"Student {enrollment_id_str} has already enrolled today.")
                            pass
                        else:
                            data.loc[len(data)] = [enrollment_id_str, name, "Present","Softwere Engneering",current_day,current_date]
                            enroll_flag=1
                            # st.success(f"Enrollment successful for {name} on {current_date}.")
                            data.to_csv("database.csv")

        


        col1,col2= st.columns(2)
        with col1:
            st.subheader("Scan Image")
            landmarks_image=cv2.resize(img,(2123, 1850))

            # Display the final image with landmarks and attendance info
            st.image(landmarks_image, caption="Processed Image with Landmarks", use_column_width=True)


        with col2:
            st.subheader("Detected Image")
            selected_image=cv2.resize(selected_image,(2123, 1850))
            selected_image = cv2.cvtColor(selected_image, cv2.COLOR_BGR2RGB)
            # st.write(selected_image.shape)

            # Display the final image with landmarks and attendance info
            st.image(selected_image, caption="Detected Student", use_column_width=True)

        if enroll_flag==1:
            st.success(f"Successful Attendance Marked for {name} on {current_date}.")
        else:
            st.warning(f"Student with {enrollment_id_str} Enrollment Id has already Marked Their Attendance today.")


        # Save the updated attendance data
        data["Enrollment_id"] = data["Enrollment_id"].astype(str)
        # data.to_csv("database.csv")
       

    else:
        st.error("Please capture an image to proceed.")



elif selections=="Attendance Sheet":
    st.markdown(
        """
        <h1 >Attendance Sheet</h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <hr style="border: none; height: 2px;width: 50%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%);" />
        """,
        unsafe_allow_html=True
    )

    
    st.dataframe(data.head(15))




elif selections=="Enroll Student":
    def display_student(image_path,Name):
        student_image = cv2.imread(image_path)
        student_image=cv2.resize(student_image,(200, 200))
        student_image = cv2.cvtColor(student_image, cv2.COLOR_BGR2RGB)
        # Display the final image with landmarks and attendance info, adjust size here
        st.image(student_image, caption=Name, use_column_width=False,width=200)  # Adjusted to fill column width
    st.markdown(
        """
        <h1 style='text-align: center;'>These Student Are Enroll With Us</h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <hr style="border: none; height: 2px;width: 50%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
        """,
        unsafe_allow_html=True
    )



    # Row 1
    student1, student2, student3, student4 = st.columns(4)



    with student1:
        image_path1 = "student/Abdal-860-1.JPG"
        display_student(image_path1,"Abdal")

    with student2:
        image_path2 = "student/Ahmad Suliman-885-1.jpg"
        display_student(image_path2,"Ahmad Suliman")

    with student3:
        image_path3 = "student/Ali-886-2.jpg"
        display_student(image_path3,"Muhammad Ali")

    with student4:
        image_path4 = "student/Amir Zib-890-1.JPG"
        display_student(image_path4,"Amir Zib")

    
   


    # Row 2
    student5, student6, student7, student8 = st.columns(4)



    with student5:
        image_path1 = 'student\Ansar-875-1.jpg'
        display_student(image_path1,"Muhammad Ansar")

    with student6:
        image_path2 = "student\Asad-894-1.jpg"
        display_student(image_path2,"Asad")

    with student7:
        image_path3 = "student\Salman-862-1.jpg"
        display_student(image_path3,"Muhammad Salman")

    with student8:
        image_path4 = "student\Munsif-868-1.jpg"
        display_student(image_path4,"Munsif Khan")






    # Row 3
    student9, student10, student11, student12 = st.columns(4)



    with student9:
        image_path1 = 'student\Hashim-863-1.jpg'
        display_student(image_path1,"Muhammad Hashim")

    with student10:
        image_path2 = r"student\Nasar-889-2.JPG"
        display_student(image_path2,"Umar Nasar")

    with student11:
        image_path3 = "student\Masood-884-1.jpg"
        display_student(image_path3,"Masood Khan")

    with student12:
        image_path4 = "student\Hilal-874-2.JPG"
        display_student(image_path4,"Muhammad Hilal")



    # Row 3
    student13, student14, student15, student16 = st.columns(4)



    with student13:
        image_path1 = r'student\awal Said-877-1.JPG'
        display_student(image_path1,"Awal Said")

    with student14:
        image_path2 = r"student\Said Wali-895-1.jpg"
        display_student(image_path2,"Said Wali Khan")

    with student15:
        image_path3 = r"student\Zakria-881-1.jpg"
        display_student(image_path3,"Muhammad Zakria")

    with student16:
        image_path4 = r"student\Raheel-865-1.JPG"
        display_student(image_path4,"Raheel Jamal")




    # Row 3
    student17, student18, student19, student20 = st.columns(4)



    with student13:
        image_path1 = r'student\Inam-876-2.JPG'
        display_student(image_path1,"Inam Ullah")

    with student14:
        image_path2 = r"student\Sami Ullah-869-1.JPG"
        display_student(image_path2,"Sami Ullah")

    with student15:
        image_path3 = r"student\Niaz-861-1.jpg"
        display_student(image_path3,"Niaz Ali")

    with student16:
        image_path4 = r"student\Mudsir-892-1.JPG"
        display_student(image_path4,"Mudsir Hussin Sha")



    # Row 3
    student21, student22, student23, student24 = st.columns(4)



    with student21:
        image_path1 = r'student\Muzmmil-878-1.JPG'
        display_student(image_path1,"Muzmmil")

    with student22:
        image_path2 = r"student\Shawir Khan-887-1.JPG"
        display_student(image_path2,"Shawir Khan")

    with student23:
        image_path3 = r"student\Hasan Ahamad-873-1.JPG"
        display_student(image_path3,"Hasan Ahamad")

    with student24:
        image_path4 = r"student\Suliman-867-1.jpg"
        display_student(image_path4,"Suliman(cr)")






    # Row 3
    student25, student26, student27, student28 = st.columns(4)



    with student25:
        image_path1 = r'student\Mohib-841-4.jpg'
        display_student(image_path1,"Mohib Wadood (Nazim)")

    with student26:
        image_path2 = r"student\Haroon-914-1.jpg"
        display_student(image_path2,"Haroon Ahmad (Jonuir)")

    with student27:
        image_path3 = r"student\Fayaz-924-1.jpg"
        display_student(image_path3,"Fayaz (Jonuir)")

    with student28:
        image_path4 = r"student\Rizwan-662-1.jpg"
        display_student(image_path4,"Rizwan(cr)")


