from winotify import Notification


def show_notification(title, message):

    toast = Notification(
        app_id="AI Focus Assistant",
        title=title,
        msg=message
    )

    toast.show()