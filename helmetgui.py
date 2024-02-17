from tkinter import ttk
from tkinter import *
import tkinter as tk
from my_functions import *

import os
import cv2
import time 
from tkinter import messagebox
from imutils import face_utils

from matplotlib import pyplot as pl
import numpy as np
# import playsound

from datetime import datetime
from tkinter_webcam import webcam

from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk
from keras.models import load_model

import easyocr
from fpdf import FPDF

win=tk.Tk()
win.title("Helmet Detection System")
win.config(bg='papayawhip')
width= win.winfo_screenwidth()
height= win.winfo_screenheight()
#setting tkinter window size
win.geometry("%dx%d" % (width, height))
win.resizable(False,False)
img1 =Image.open('helmetimg.jpg')
img1 =img1.resize((width,height))
bg = ImageTk.PhotoImage(img1)




def select_image():
	global filename,e1
	f_types = [('Jpg Files', '*.jpg'),('PNG Files','*.png')] 
	filename = tk.filedialog.askopenfilename(filetypes=f_types)
	img=Image.open(filename) # read the image file
	#w=width-160
	#h=height-200
	img=img.resize((500,500)) # new width & height
	img=ImageTk.PhotoImage(img)
	e1 =tk.Label(win)
	e1.place(x=160,y=145)
	e1.image = img # keep a reference! by attaching it to a widget attribute
	e1['image']=img # Sh

def detect_helmet():
	
	global filename,e2,imgs
	imgs=cv2.imread(filename)
	#im=cv2.flip(imgs,1,1) 
	frame = cv2.resize(imgs, frame_size)  # resizing image
	orifinal_frame = frame.copy()
	frame, results = object_detection(frame) 
	print(results)
	rider_list = []
	head_list = []
	number_list = []

	for result in results:
		x1,y1,x2,y2,cnf, clas = result
		if clas == 0:
			rider_list.append(result)
		elif clas == 1:
			head_list.append(result)
		elif clas == 2:
			number_list.append(result)
	for rdr in rider_list:
		time_stamp = str(time.time())
		x1r, y1r, x2r, y2r, cnfr, clasr = rdr
		for hd in head_list:
			x1h, y1h, x2h, y2h, cnfh, clash = hd
			if inside_box([x1r,y1r,x2r,y2r], [x1h,y1h,x2h,y2h]): # if this head inside this rider bbox
				try:
					head_img = orifinal_frame[y1h:y2h, x1h:x2h]
					helmet_present = img_classify(head_img)
				except:
					helmet_present[0] = None
				if  helmet_present[0] == True: # if helmet present
					frame = cv2.rectangle(frame, (x1h, y1h), (x2h, y2h), (0,255,0), 1)
					frame = cv2.putText(frame, f'{round(helmet_present[1],1)}', (x1h, y1h+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
					#frame = cv2.putText(frame, 'With helmet', (x1h, y1h+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
					cv2.putText(frame, 'With helmeet', (x1h, y1h+40),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
				elif helmet_present[0] == None: # Poor prediction
					frame = cv2.rectangle(frame, (x1h, y1h), (x2h, y2h), (0, 255, 255), 1)
					frame = cv2.putText(frame, f'{round(helmet_present[1],1)}', (x1h, y1h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
				elif helmet_present[0] == False: # if helmet absent 
					frame = cv2.rectangle(frame, (x1h, y1h), (x2h, y2h), (0, 0, 255), 1)
					frame = cv2.putText(frame, f'{round(helmet_present[1],1)}', (x1h, y1h+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
					cv2.putText(frame, 'Without Helmet' , (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
					try:
						cv2.imwrite(f'riders_pictures/{time_stamp}.jpg', frame[y1r:y2r, x1r:x2r])
					except:
						print('could not save rider')
					for num in number_list:
						x1_num, y1_num, x2_num, y2_num, conf_num, clas_num = num
						if inside_box([x1r,y1r,x2r,y2r], [x1_num, y1_num, x2_num, y2_num]):
							try:
								num_img = orifinal_frame[y1_num:y2_num, x1_num:x2_num]
								cv2.imwrite(f'number_plates/{time_stamp}_{conf_num}.jpg', num_img)
							except:
								print('could not save number plate')
	#if save_video: # save video
		#out.write(frame)
	#if save_img: #save img
		#cv2.imwrite('saved_frame.jpg', frame)
	
	#if show_video: # show video
		#frame = cv2.resize(frame, (900, 450))  # resizing to fit in screen
		#cv2.imshow('Frame', frame)

	frame = cv2.resize(frame, (900, 450))  # resizing to fit in screen
	cv2.imshow('Frame', frame)
	cv2.waitKey(0)


def detect_helmet_in(source):
	
	save_video = True # want to save video? (when video as source)
	show_video=True # set true when using video file
	save_img=False  # set true when using only image file to save the image
	# when using image as input, lower the threshold value of image classification

	#saveing video as output
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	out = cv2.VideoWriter('output.avi', fourcc, 20.0, frame_size)
	cap = cv2.VideoCapture(source)
	while(cap.isOpened()):
		ret, frame = cap.read()
		if ret == True:
			frame = cv2.resize(frame, frame_size)  # resizing image
			orifinal_frame = frame.copy()
			frame, results = object_detection(frame) 

			rider_list = []
			head_list = []
			number_list = []
			for result in results:
				x1,y1,x2,y2,cnf, clas = result
				if clas == 0:
					rider_list.append(result)
				elif clas == 1:
					head_list.append(result)
				elif clas == 2:
					number_list.append(result)
			for rdr in rider_list:
				time_stamp = str(time.time())
				x1r, y1r, x2r, y2r, cnfr, clasr = rdr
				for hd in head_list:
					x1h, y1h, x2h, y2h, cnfh, clash = hd
					if inside_box([x1r,y1r,x2r,y2r], [x1h,y1h,x2h,y2h]):
						try:
							head_img = orifinal_frame[y1h:y2h, x1h:x2h]
							helmet_present = img_classify(head_img)
						except:
							helmet_present[0] = None
						if  helmet_present[0] == True: # if helmet present
							frame = cv2.rectangle(frame, (x1h, y1h), (x2h, y2h), (0,255,0), 1)
							frame = cv2.putText(frame, f'{round(helmet_present[1],1)}', (x1h, y1h+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
						elif helmet_present[0] == None: # Poor prediction
							frame = cv2.rectangle(frame, (x1h, y1h), (x2h, y2h), (0, 255, 255), 1)
							frame = cv2.putText(frame, f'{round(helmet_present[1],1)}', (x1h, y1h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
						elif helmet_present[0] == False: # if helmet absent 
							frame = cv2.rectangle(frame, (x1h, y1h), (x2h, y2h), (0, 0, 255), 1)
							frame = cv2.putText(frame, f'{round(helmet_present[1],1)}', (x1h, y1h+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
							try:
								cv2.imwrite(f'riders_pictures/{time_stamp}.jpg', frame[y1r:y2r, x1r:x2r])
							except:
								print('could not save rider')

							for num in number_list:
								x1_num, y1_num, x2_num, y2_num, conf_num, clas_num = num
								if inside_box([x1r,y1r,x2r,y2r], [x1_num, y1_num, x2_num, y2_num]):
									try:
										num_img = orifinal_frame[y1_num:y2_num, x1_num:x2_num]
										cv2.imwrite(f'number_plates/{time_stamp}_{conf_num}.jpg', num_img)
									except:
										print('could not save number plate')
			if save_video: # save video
				out.write(frame)
			if save_img: #save img
				cv2.imwrite('saved_frame.jpg', frame)
	
			if show_video: # show video
				frame = cv2.resize(frame, (900, 450))  # resizing to fit in screen
				cv2.imshow('Frame', frame)


			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		else:
			break

	cap.release()
	cv2.destroyAllWindows()
	print('Execution completed')

def detect_helmet_video():
	f_types = [('mp4 Files','*.mp4')] 
	filename1 = tk.filedialog.askopenfilename(filetypes=f_types)
	print(filename1)
	source = filename1
	detect_helmet_in(source)


# def detect_helmet_camera():
# 	#source=0
# 	#detect_helmet_in(source)
# 	save_video = True # want to save video? (when video as source)
# 	show_video=True # set true when using video file
# 	save_img=False  # set true when using only image file to save the image
	
# 	cap = cv2.VideoCapture(0)
# 	while(cap.isOpened()):
# 		ret, frame = cap.read()
# 		if ret == True:
# 			frame = cv2.resize(frame, frame_size)  # resizing image
# 			orifinal_frame = frame.copy()
# 			frame, results = object_detection(frame) 
# 			for result in results:
# 				x1,y1,x2,y2,cnf, clas = result
# 				print(clas)
# 			if show_video: # show video
# 				frame = cv2.resize(frame, (900, 450))  # resizing to fit in screen
# 				cv2.imshow('Frame', frame)


# 			if cv2.waitKey(1) & 0xFF == ord('q'):
# 				break
# 		else:
# 			break

# 	cap.release()
# 	cv2.destroyAllWindows()
# 	print('Execution completed')

def detect_helmet_camera1():
	labels_dict={0:'without helmet',1:'with helmet'}
	color_dict={0:(0,0,255),1:(0,255,0)}
	model=load_model("./model2-002.h5")
	size = 4
	webcam = cv2.VideoCapture(0) #Use camera 0
	classifier = cv2.CascadeClassifier('./haarcascade_helmet.xml')
	while True:
		(rval, im) = webcam.read()
		im=cv2.flip(im,1,1) #Flip to act as a mirror
		mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))
		faces = classifier.detectMultiScale(mini)
		for f in faces:
			(x, y, w, h) = [v * size for v in f] #Scale the shapesize backup
			face_img = im[y:y+h, x:x+w]
			resized=cv2.resize(face_img,(150,150))
			normalized=resized/255.0
			reshaped=np.reshape(normalized,(1,150,150,3))
			reshaped = np.vstack([reshaped])
			result=model.predict(reshaped)
			label=np.argmax(result,axis=1)[0]
			cv2.rectangle(im,(x,y),(x+w,y+h),color_dict[label],2)
			cv2.rectangle(im,(x,y-40),(x+w,y),color_dict[label],-1)
			cv2.putText(im, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
		cv2.imshow('LIVE',   im)
		key = cv2.waitKey(10)
    # if Esc key is press then break out of the loop 
		if key == 27: #The Esc key
			break
# Stop video
	webcam.release()

# Close all started windows
	cv2.destroyAllWindows()

      
def fine_receipt():
    pdf = FPDF('P', 'mm', 'Letter')
    
    f_types = [('Jpg Files', '*.jpg'), ('PNG Files', '*.png')]
    img_file = tk.filedialog.askopenfilename(filetypes=f_types)
    img = cv2.imread(img_file)
    reader = easyocr.Reader(['en'], gpu=False)
    results = reader.readtext(img)
    vehicle_number = ""
    
    for r in range(len(results)):
        vehicle_number += results[r][1]
    
    timestr = time.strftime('%Y-%m-%d-%H-%M-%S')
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    
    # Title
    pdf.cell(190, 10, txt="Fine Receipt", ln=True, align='C')
    pdf.ln(10)  # Add a little space
    
    # Vehicle Number
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f"Vehicle No: {vehicle_number}", ln=True)
    
    # Fine Amount and Reason
    pdf.cell(0, 10, "Fine Amount: $200", ln=True)
    pdf.cell(0, 10, "Reason: Without Helmet", ln=True)
    
    # Save the PDF
    pdf_file_name = f'fine_receipts/{timestr}_{vehicle_number}.pdf'
    pdf.output(pdf_file_name)
    
    # Show a confirmation message
    messagebox.showinfo("Receipt", "Fine Receipt generated")


def winclose():
	win.quit()

	         
  
e1 =tk.Label(win)
e2 =tk.Label(win)
	
label1 = Label( win, image = bg)
label1.place(x = 0, y = 0)
#ttk.Label(win,text="HELMET DETECTION SYSTEM",background="black",foreground="white",font=("Times New Roman",40)).place(relx=0.13,rely=0.02)
btdata=Button(win,text="Select Image",fg="black",bg="white",width=15,font=("Times New Roman",12),command=select_image)
btdata.place(x=900,y=10)

btdata=Button(win,text="Detect Helmet",fg="black",bg="white",width=15,font=("Times New Roman",12),command=detect_helmet)
btdata.place(x=1050,y=10)
btdata=Button(win,text="Detect Helemt in Video",fg="black",bg="white",width=17,font=("Times New Roman",12),command=detect_helmet_video)
btdata.place(x=1200,y=10)
btdata=Button(win,text="Detect Helmet Through Camera",fg="black",bg="white",width=25,font=("Times New Roman",12),command=detect_helmet_camera1)
btdata.place(x=900,y=50)
btdata=Button(win,text="Fine Receipt",fg="black",bg="white",width=10,font=("Times New Roman",12),command=fine_receipt)
btdata.place(x=1140,y=50)

btdata=Button(win,text="Exit",fg="black",bg="white",width=10,font=("Times New Roman",12),command=winclose)
btdata.place(x=1250,y=50)


win.mainloop()
#vid.release() 