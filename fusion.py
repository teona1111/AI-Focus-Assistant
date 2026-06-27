def fuse_predictions(
    sensor,
    eye
):

    rf_focus = sensor["focus"]
    rf_sleepy = sensor["sleepy"]
    rf_distracted = sensor["distracted"]

    eye_open = eye["open"]
    eye_closed = eye["closed"]

    focus_score = (
        0.50 * eye_open +
        0.50 * rf_focus
    )

    sleepy_score = (
        0.70 * eye_closed +
        0.30 * rf_sleepy
    )

    distracted_score = (
        0.40 * eye_open +
        0.60 * rf_distracted
    )

    scores = {
        "focus": focus_score,
        "sleepy": sleepy_score,
        "distracted": distracted_score
    }

    final_state = max(
        scores,
        key=scores.get
    )

    return {
        "state": final_state,
        "focus_score": round(
            focus_score,
            3
        ),
        "sleepy_score": round(
            sleepy_score,
            3
        ),
        "distracted_score": round(
            distracted_score,
            3
        )
    }


if __name__ == "__main__":

    sensor = {
        "focus": 0.53,
        "sleepy": 0.21,
        "distracted": 0.26
    }

    eye = {
        "open": 0.99,
        "closed": 0.01
    }

    result = fuse_predictions(
        sensor,
        eye
    )

    print(result)