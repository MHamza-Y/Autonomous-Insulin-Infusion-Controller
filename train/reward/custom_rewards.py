def custom_reward(BG_last_hour):
    if BG_last_hour[-1] > 260:
        return -10
    elif BG_last_hour[-1] > 180:
        return -2
    elif BG_last_hour[-1] > 140:
        return -1
    elif BG_last_hour[-1] < 100:
        return -1
    elif BG_last_hour[-1] < 90:
        return -2
    elif BG_last_hour[-1] < 70:
        return -10
    else:
        return 2
