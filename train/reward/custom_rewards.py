from simglucose.analysis.risk import risk_index

REF_BG_POINT = 131
LOW_BG_LIMIT = 70
HIGH_BG_LIMIT = 330
MAX_HIGH_TO_REF_DIFF = HIGH_BG_LIMIT - REF_BG_POINT
MAX_LOW_TO_REF_DIFF = REF_BG_POINT - LOW_BG_LIMIT


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


def shaped_reward_around_normal_bg(BG_last_hour):
    current_bg = BG_last_hour[-1]

    if current_bg > REF_BG_POINT:
        diff = current_bg - REF_BG_POINT
        reward = 1 - ((diff / MAX_HIGH_TO_REF_DIFF) ** 0.4)
    else:
        diff = REF_BG_POINT - current_bg
        reward = 1 - ((diff / MAX_LOW_TO_REF_DIFF) ** 0.4)

    if current_bg <= LOW_BG_LIMIT or current_bg >= HIGH_BG_LIMIT:
        reward = reward - 10

    return reward

