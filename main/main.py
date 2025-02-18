import os
import instaloader
import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
import pickle

app = FastAPI()
def extract_profile_data(username):
    # Create an instance of Instaloader
    loader = instaloader.Instaloader()

    try:
        profile = instaloader.Profile.from_username(loader.context, username)

        print(f"Username: {profile.username}")
        print(f"Full Name: {profile.full_name}")
        print(f"Bio: {profile.biography}")
        print(f"Followers: {profile.followers}")
        print(f"Following: {profile.followees}")
        print(f"Number of Posts: {profile.mediacount}")
        print(f"Is Private: {profile.is_private}")
        data = {
        "userFollowerCount": profile.followers,
        "userFollowingCount": profile.followees,
        "userBiographyLength": len(profile.biography) if profile.biography else 0,
        "userMediaCount": profile.mediacount,
        "userHasProfilePic": False,  
        "userIsPrivate": int(profile.is_private),
        "usernameDigitCount": sum(c.isdigit() for c in profile.username),
        "usernameLength": len(profile.username),
        }
        loader.download_profile(username, profile_pic_only=True)

        user_dir = os.path.join(os.getcwd(), username)

        profile_pic_filename = None
        for file in os.listdir(user_dir):
            if file.endswith("_profile_pic.jpg"):
                profile_pic_filename = os.path.join(user_dir, file)
                break

        if profile_pic_filename:
            user_profile_pic = cv2.imread(profile_pic_filename)

            default_profile_pic = cv2.imread("default_pic.jpg")  
            default_profile_pic2 = cv2.imread("default_pic2.jpg")  
           
            if user_profile_pic is None:
                print(f"Error: Unable to load user's profile picture from {profile_pic_filename}.")
                return

            if default_profile_pic is None:
                print("Error: Unable to load default profile picture. Ensure 'default_pic.jpg' is in the script directory.")
                return

            if default_profile_pic2 is None:
                print("Error: Unable to load default profile picture. Ensure 'default_pic.jpg' is in the script directory.")
                return
            if user_profile_pic.shape != default_profile_pic.shape:
                default_profile_pic = cv2.resize(default_profile_pic, (user_profile_pic.shape[1], user_profile_pic.shape[0]))
            if user_profile_pic.shape != default_profile_pic2.shape:
                default_profile_pic2 = cv2.resize(default_profile_pic2, (user_profile_pic.shape[1], user_profile_pic.shape[0]))

            difference = cv2.absdiff(user_profile_pic, default_profile_pic)
            difference2 = cv2.absdiff(user_profile_pic, default_profile_pic2)
            b, g, r = cv2.split(difference)
            b2, g2, r2 = cv2.split(difference2)
            hasProfilePic = False
            if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
                data["userHasProfilePic"] = False
                print(f"The user '{username}' is using the default profile picture.")

            elif cv2.countNonZero(b2) == 0 and cv2.countNonZero(g2) == 0 and cv2.countNonZero(r2) == 0:
                data["userHasProfilePic"] = False
                print(f"The user '{username}' is using the default profile picture.")

            else:
                data["userHasProfilePic"] = True
                print(f"The user '{username}' has set a custom profile picture.")
        else:
            print("Profile picture not found.")

        for post in profile.get_posts():
            loader.download_post(post, target=f"{profile.username}_posts")

        download_highlights(profile, loader)

        save_profile_data(profile)
        return profile

    except instaloader.exceptions.ProfileNotExistsException:
        print(f"Profile with username '{username}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

def download_highlights(profile, loader):
    try:
        highlight_dir = f"{profile.username}_highlights"
        if not os.path.exists(highlight_dir):
            os.makedirs(highlight_dir)

        for highlight in profile.get_highlights():
            for item in highlight.get_items():
                loader.download_storyitem(item, target=highlight_dir)

    except Exception as e:
        print(f"Error while downloading highlights: {e}")

def save_profile_data(profile):
    import json
    try:
        with open(f"{profile.username}_profile_data.json", "w", encoding="utf-8") as file:
            data = {
                "Username": profile.username,
                "Full Name": profile.full_name,
                "Bio": profile.biography,
                "Followers": profile.followers,
                "Following": profile.followees,
                "Number of Posts": profile.mediacount
            }
            json.dump(data, file, indent=4, ensure_ascii=False)

    except Exception as e:
        print(f"Error while saving profile data: {e}")




    



@app.get("/")
def read_root():
    return {"message": "Welcome to the Instagram Profile Data Extractor API"}
# Load the pre-trained model
model_path = os.path.join(os.path.dirname(__file__),"..\\" "models", "instaModel.h5")
model = tf.keras.models.load_model(model_path)

# Load the scaler
scaler_path = os.path.join(os.path.dirname(__file__),"..\\", "models", "scaler.pkl")
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

# Path to default profile pictures
default_pic_path = os.path.join(os.path.dirname(__file__), "default_pics", "default_pic.jpg")
default_pic2_path = os.path.join(os.path.dirname(__file__), "default_pics", "default_pic2.jpg")

class ProfileData(BaseModel):
    username: str
    followers: int
    followees: int
    biography_length: int
    mediacount: int
    has_profile_pic: bool
    is_private: bool
    username_digit_count: int
    username_length: int

def download_profile_pic(username: str) -> str:
    loader = instaloader.Instaloader()
    profile = instaloader.Profile.from_username(loader.context, username)
    profile_pic_filename = f"{username}_profile_pic.jpg"
    loader.download_profilepic(profile, filename=profile_pic_filename)
    return profile_pic_filename

def compare_images(img1_path: str, img2_path: str) -> bool:
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        return False

    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    difference = cv2.absdiff(img1, img2)
    b, g, r = cv2.split(difference)
    return cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0

def extract_profile_data(username: str) -> ProfileData:
    loader = instaloader.Instaloader()
    try:
        profile = instaloader.Profile.from_username(loader.context, username)
    except instaloader.exceptions.ProfileNotExistsException:
        raise HTTPException(status_code=404, detail=f"Profile '{username}' does not exist.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    profile_pic_path = download_profile_pic(username)
    has_profile_pic = not (
        compare_images(profile_pic_path, default_pic_path) or
        compare_images(profile_pic_path, default_pic2_path)
    )
    os.remove(profile_pic_path)

    return ProfileData(
        username=username,
        followers=profile.followers,
        followees=profile.followees,
        biography_length=len(profile.biography) if profile.biography else 0,
        mediacount=profile.mediacount,
        has_profile_pic=has_profile_pic,
        is_private=profile.is_private,
        username_digit_count=sum(c.isdigit() for c in username),
        username_length=len(username)
    )

@app.get("/check_profile/{username}")
async def check_profile(username: str):
    profile_data = extract_profile_data(username)
    print(profile_data)
    feature_array = np.array([[
        profile_data.followers,
        profile_data.followees,
        profile_data.biography_length,
        profile_data.mediacount,
        int(profile_data.has_profile_pic),
        int(profile_data.is_private),
        profile_data.username_digit_count,
        profile_data.username_length
    ]])

    scaled_features = scaler.transform(feature_array)
    scaled_features = scaled_features.reshape(1, 1, -1)

    fake_probability = model.predict(scaled_features)[0][0]
    return {
        "profile_data": profile_data.dict(),
         "fake_probability": round(fake_probability * 100, 2)
    }
