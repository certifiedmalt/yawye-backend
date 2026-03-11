import qrcode
from PIL import Image, ImageDraw, ImageFont
import os

# Config
PLAY_URL = "https://play.google.com/store/apps/details?id=com.youarewhatyoueat.app&pcampaignid=web_share"
WIDTH, HEIGHT = 1280, 720
BG_COLOR = (10, 10, 10)
GREEN = (0, 230, 118)
WHITE = (255, 255, 255)
GRAY = (160, 160, 160)
OUT = "/app/marketing/production/endcard.png"
LOGO_PATH = "/app/marketing/real_icon_watermark.png"

# Generate QR code
qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_H, box_size=8, border=2)
qr.add_data(PLAY_URL)
qr.make(fit=True)
qr_img = qr.make_image(fill_color="white", back_color=(10, 10, 10)).convert("RGBA")
qr_size = 220
qr_img = qr_img.resize((qr_size, qr_size), Image.LANCZOS)

# Create end card
card = Image.new("RGBA", (WIDTH, HEIGHT), BG_COLOR)
draw = ImageDraw.Draw(card)

# Load logo
logo = Image.open(LOGO_PATH).convert("RGBA").resize((100, 100), Image.LANCZOS)

# Position elements - centered layout
center_x = WIDTH // 2

# Logo at top area
logo_y = 120
card.paste(logo, (center_x - 50, logo_y), logo)

# App name below logo
try:
    font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 42)
    font_cta = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    font_url = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    font_scan = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
except:
    font_title = ImageFont.load_default()
    font_cta = font_title
    font_url = font_title
    font_scan = font_title

title_y = logo_y + 115
draw.text((center_x, title_y), "You Are What You Eat", fill=GREEN, font=font_title, anchor="mt")

# CTA text
cta_y = title_y + 55
draw.text((center_x, cta_y), "Download Free on Google Play", fill=WHITE, font=font_cta, anchor="mt")

# QR code centered
qr_y = cta_y + 50
card.paste(qr_img, (center_x - qr_size // 2, qr_y), qr_img)

# "Scan to download" below QR
scan_y = qr_y + qr_size + 12
draw.text((center_x, scan_y), "Scan to download", fill=GRAY, font=font_scan, anchor="mt")

# URL at bottom
url_y = HEIGHT - 45
draw.text((center_x, url_y), "play.google.com/store/apps/details?id=com.youarewhatyoueat.app", fill=GRAY, font=font_url, anchor="mt")

card.save(OUT)
print(f"End card saved: {OUT}")
print(f"Size: {card.size}")
