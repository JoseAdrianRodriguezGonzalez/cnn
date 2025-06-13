from PIL import Image
import os
def reduce_images(string):
    for i in os.listdir("rotate/"+string):
        foo = Image.open("./rotate/"+string+"/"+i)  # My image is a 200x374 jpeg that is 102kb large
        foo.size  # (200, 374)
        print(f'imagen {i}, renderizada')
        # downsize the image with an ANTIALIAS filter (gives the highest quality)
        foo = foo.resize((76,76))
  # The saved downsized image size is 24.8kb

        foo.save("data_rotate_5/"+string+"/"+i, optimize=True, quality=95)
        print(f'imagen {i}, guardada')
def make_folders():
    os.mkdir("data_rotate_5")
    for i in range(0,12):
        string="state_"+str(i)
        os.mkdir(f"data_rotate_5/{string}")
        reduce_images(string)
#Create the folders
make_folders()
"""
foo = Image.open('./Aprendizaje_1.png')  # My image is a 200x374 jpeg that is 102kb large
foo.size  # (200, 374)
 
 # downsize the image with an ANTIALIAS filter (gives the highest quality)
foo = foo.resize((160,300))

foo.save('./Aprendizaje_1_resized.png', quality=95)  # The saved downsized image size is 24.8kb

foo.save('./Aprendizaje_1_resized_2.png', optimize=True, quality=95)
"""
