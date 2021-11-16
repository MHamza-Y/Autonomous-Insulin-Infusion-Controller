from simglucose.analysis.risk import risk_index

REF_BG_POINT = 131
LOW_BG_LIMIT = 70
HIGH_BG_LIMIT = 330
HIGH_BG_SAFE_LIMIT = 170
LOW_BG_SAFE_LIMIT = 110
MAX_HIGH_SAFE_INTERVAL = HIGH_BG_SAFE_LIMIT - REF_BG_POINT
MAX_LOW_SAFE_INTERVAL = REF_BG_POINT - LOW_BG_SAFE_LIMIT
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
        reward = 1 - ((diff / MAX_HIGH_TO_REF_DIFF) ** 0.06)
    else:
        diff = REF_BG_POINT - current_bg
        reward = 1 - ((diff / MAX_LOW_TO_REF_DIFF) ** 0.2)

    if current_bg <= LOW_BG_LIMIT:
        reward += -10
    elif current_bg >= HIGH_BG_LIMIT:
        reward += -20

    return reward


def shaped_negative_reward_around_normal_bg(BG_last_hour):
    current_bg = BG_last_hour[-1]

    if REF_BG_POINT < current_bg <= HIGH_BG_SAFE_LIMIT:
        diff = current_bg - REF_BG_POINT
        reward = 3 * (1 - ((diff / MAX_HIGH_SAFE_INTERVAL) ** 0.2))
    elif REF_BG_POINT >= current_bg > LOW_BG_SAFE_LIMIT:
        diff = REF_BG_POINT - current_bg
        reward = 3 * (1 - ((diff / MAX_LOW_SAFE_INTERVAL) ** 0.2))
    elif current_bg > HIGH_BG_SAFE_LIMIT:
        reward = -2
    else:
        reward = -1

    if current_bg <= LOW_BG_LIMIT:
        reward += -100
    elif current_bg >= HIGH_BG_LIMIT:
        reward += -300

    return reward
