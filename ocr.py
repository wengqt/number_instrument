import pytesseract
from PIL import Image

# image = Image.open('./img2/no/image20181212下午14752/3_and1.jpg')
# image = Image.open('/Users/weng/Documents/大四/NumInstrument/img2/no/image20181212上午101234/3_light1.jpg')
image = Image.open('/Users/weng/Documents/大四/NumInstrument/img2/image20181211下午54016/3_numblock2.jpg')
# print(image.size)
image = image.resize((int(image.size[0]/10),int(image.size[1]/10)))
tessdata_config = '--oem 0 digits'
text = pytesseract.image_to_string(image,lang='eng_ome0',config=tessdata_config)
print(text)