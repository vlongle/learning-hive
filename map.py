import folium


import folium
from selenium import webdriver
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PIL import Image

def create_english_channel_map():
    lat, lon = 50.0, 0.0
    m = folium.Map(location=[lat, lon], zoom_start=7, tiles='Stamen Terrain')
    folium.Marker([51.127876, 1.313403], tooltip='Dover, UK').add_to(m)
    folium.Marker([50.95129, 1.858686], tooltip='Calais, France').add_to(m)
    
    # Save to an HTML file
    map_file = "english_channel_map.html"
    m.save(map_file)
    return map_file

def html_to_png(html_file):
    # Start a virtual display
    # Open the web page in a browser with selenium and capture a screenshot
    browser = webdriver.Chrome(executable_path='/path/to/chromedriver')  # Update the path
    browser.get('file://' + html_file)
    image_file = "english_channel_map.png"
    browser.save_screenshot(image_file)
    browser.quit()
    return image_file

def png_to_pdf(png_file):
    # Open the PNG image with PIL
    im = Image.open(png_file)
    
    # Get the dimensions of the image
    width, height = im.size
    
    # Convert the dimensions to points (from pixels)
    width, height = width * 0.75, height * 0.75

    # Create a new PDF file
    pdf_file = "english
