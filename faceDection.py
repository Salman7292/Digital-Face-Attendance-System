import cv2
import os
import face_recognition
from tqdm import tqdm
import pickle

# Clear the terminal screen (optional, for Windows)
os.system("cls")






def encoding(images_list):
    Encoded_images = []  # List to store converted RGB images

    for img in tqdm(images_list):
        # Convert BGR image to RGB
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encod= face_recognition.face_encodings(rgb_img)[0]

        Encoded_images.append(encod)
       
    return Encoded_images
        

    

       







# Path to the folder containing images
folder_path = "student"

# Check if the folder exists
if not os.path.exists(folder_path):
    print(f"Error: The folder '{folder_path}' does not exist.")
    exit()

# List all image files in the folder
images = os.listdir(folder_path)


name_list=[]
enrollment_id=[]
print(images)


for img in images:
    image_component = img.split('-')
    name_list.append(image_component[0])
    enrollment_id.append(image_component[1])



print(name_list)
print(enrollment_id)

# List to store the loaded images
image_list = []

# Load each image and append it to the image list
for image in images:
    img_path = os.path.join(folder_path, image)
    img = cv2.imread(img_path)
    if img is not None:
        image_list.append(img)
    else:
        print(f"Error: Unable to load image '{img_path}'.")




encoding_known=encoding(image_list)
print(encoding_known)


encoding_known_list=[encoding_known,name_list,enrollment_id]


print(encoding_known_list)

file=open("Encoding_file5.pkl","wb")
pickle.dump(encoding_known_list,file)




# # Check if the list has at least two images and display the second one
# if len(image_list) > 1:
#     # Resize the second image to (50x50)
#     resized_image = cv2.resize(image_list[0], (400, 400))
    
#     # Display the resized image
#     cv2.imshow("Second Image", resized_image)
    
#     # Wait until a key is pressed to close the window
#     cv2.waitKey(0)
    
#     # Close the window after the key is pressed
#     cv2.destroyAllWindows()
# else:
#     print("Not enough images in the list.")
