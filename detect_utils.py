from datetime import datetime, timedelta
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib

def send_email(email_sender:str, email_recipient, email_server:str, email_credentials:str, title: str,  body: str, img: str, latest_notification: datetime) -> datetime:
    if latest_notification and datetime.now() - timedelta(seconds=60) < latest_notification:
        return latest_notification
    message = MIMEMultipart('mixed')
    message['From'] = 'Kamera <{}>'.format(email_sender)
    message['To'] = email_recipient
    message['Subject'] = title
    body = MIMEText(body, 'html')
    message.attach(body)

    with open(img, 'rb') as img:
        p = MIMEApplication(img.read(), _subtype='jpg')
        p.add_header('Content-Disposition', 'attachment; filename=camera.jpg')
        message.attach(p)

    server, port = email_server.split(':')
    user, passwd = email_credentials.split(':')
    with smtplib.SMTP(server, port) as server:
        server.starttls()
        server.login(user, passwd)
        server.sendmail(email_sender, email_recipient, message.as_string())
        server.quit()

    return datetime.now()

def get_area_percentage(orig_width: float, orig_height: float, box: list) -> float:
    box_area = (orig_height * box[0] - orig_height * box[2]) * (orig_width * box[1] - orig_width * box[3])
    return box_area / (orig_height * orig_width)

def box_contains_exluded_points(orig_width: float, orig_height: float, box: list, excluded_points: list) -> bool:
    for excluded_point in excluded_points:
        relative_x = excluded_point[0] / orig_width
        relative_y = excluded_point[1] / orig_height
        if relative_x >= box[1] and relative_x <= box[3] and relative_y >= box[0] and relative_y <= box[2]:
            return True
    return False

def get_center_point(orig_width: float, orig_height: float, box: list) -> list:
    return [round(orig_width * ((box[3] + box[1]) / 2)), round(orig_height * ((box[2] + box[0]) / 2))]
