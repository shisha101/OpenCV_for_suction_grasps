import os
import time
directory = "Query_images"
current_wd = os.getcwd()
index = 0
file_name_list = os.listdir(directory)
for file in file_name_list:#os.getcwd()
    if file.endswith(".jpg"):
        # extraction of file name without extention to pares to an int
        name_without_extention = file.rsplit('.', 1)
        try:
            file_name_num = int(name_without_extention[0])
        except ValueError:
            file_name_num = ""
        # if the file name has a smaller number than the index then do nothing leave it as it is 
        if (file_name_num != "" and file_name_num <= index):
            continue
        else: # look for a location to store the file
            new_file_name = str(index)+".jpg"
            while (os.path.exists(directory+str("/")+new_file_name)):
#             print "a file with the name :"+new_file_name+" exists "
                index += 1
                new_file_name = str(index)+".jpg"
            print "the file named: "+ file+ " --> "+ new_file_name + " index --> " +str(index)
            os.rename(directory+str("/")+file, directory+str("/")+new_file_name)       