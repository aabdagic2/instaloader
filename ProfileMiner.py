import instaloader
import os
import cv2
import numpy as np

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
                hasProfilePic = True
                print(f"The user '{username}' is using the default profile picture.")

            elif cv2.countNonZero(b2) == 0 and cv2.countNonZero(g2) == 0 and cv2.countNonZero(r2) == 0:
                hasProfilePic = True
                print(f"The user '{username}' is using the default profile picture.")

            else:
                hasProfilePic = False
                print(f"The user '{username}' has set a custom profile picture.")
        else:
            print("Profile picture not found.")

        for post in profile.get_posts():
            loader.download_post(post, target=f"{profile.username}_posts")

        download_highlights(profile, loader)

        save_profile_data(profile)

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

if __name__ == "__main__":
    username = input("Enter the Instagram username: ")
    extract_profile_data(username)


    
