# importing the tkinter module and PIL
# that is pillow module
from tkinter import *
from tkinter import filedialog
import os
from PIL import ImageTk, Image
import cv2 as cv
import time

text="Sudoku Solver"

def forward(img_no):
	print(img_no)
	determineText(img_no)
	# GLobal variable so that we can have
	# access and change the variable
	# whenever needed
	global label
	global button_forward
	global button_back
	global button_exit
	label.grid_forget()

	# This is for clearing the screen so that
	# our next image can pop up
	label = Label(image=List_images[img_no-1])

	# as the list starts from 0 so we are
	# subtracting one
	label.grid(row=1, column=2, columnspan=4)
	button_forward = Button(root, text="İleri",
						command=lambda: forward(img_no+1))

	# img_no+1 as we want the next image to pop up
	if img_no == 10:
		button_forward = Button(root, text="Çözdür",
								command=lambda: solver())

	# img_no-1 as we want previous image when we click
	# back button
	button_back = Button(root, text="Geri",
						command=lambda: back(img_no-1))

	# Placing the button in new grid

	button_back.grid(row=5, column=1)
	button_exit.grid(row=0, column=6)
	button_forward.grid(row=5, column=6)


def back(img_no):
	if(img_no>=1):
		print(img_no)
		determineText(img_no)
		# We will have global variable to access these
		# variable and change whenever needed
		global label
		global button_forward
		global button_back
		global button_exit
		label.grid_forget()
		if img_no == 1:
			button_back = Button(root, text="Geri", command=back,
					state=DISABLED)
		else:
			button_back = Button(root, text="Geri",
							command=lambda: back(img_no - 1))
		print(img_no)
		# for clearing the image for new image to pop up
		label = Label(image=List_images[img_no - 1])
		label.grid(row=1, column=0, columnspan=3)
		button_forward = Button(root, text="İleri",
								command=lambda: forward(img_no + 1))
		
		# whenever the first image will be there we will
		# have the back button disabled
		
		label.grid(row=1, column=2, columnspan=4)
		button_back.grid(row=5, column=1)
		button_exit.grid(row=0, column=6)
		button_forward.grid(row=5, column=6)

def showImage():

	fln=filedialog.askopenfilename(initialdir=os.getcwd(),title="Resmi Seçiniz",filetypes=(("JPEG File",".jpeg"),("PNG file",".png")))
	print(fln)
	
	img=Image.open(fln)
	img=ImageTk.PhotoImage(img)
	label.configure(image=img)
	label.image=img
	#time.sleep(3)
	#cv.imshow("Debug" , cv.imread(fln , 0))
	#cv.waitKey()
	
def solver():
	global label
	label.grid_forget()
	# We will make the title of our app as Image Viewer
	root.title("Sudoku Solver")
	label=Label(root,text="",fg="red")
	label.grid(row=15,column=1,columnspan=10,pady=5)

	label.grid(row=0,column=1,columnspan=10)
	solvedLabel=Label(root,text="",fg="green")
	solvedLabel.grid(row=15,column=1,columnspan=10,pady=5)
	
	cells={}
	def validateNum(P):
		out=(P.isdigit() or P=="") and len(P)<2
		return out
	reg=root.register(validateNum)
	def draw3x3Grid(row,column,bgcolor):
		for i in range(3):
			for j in range(3):
				e=Entry(root,width=5,bg=bgcolor,justify="center",validate="key",validatecommand=(reg,"%P"))
				e.grid(row=row+i+1,column=column+j+1, sticky="nsew",padx=1,pady=1,ipady=5)
				cells[(row+i+1,column+j+1)]=e
	def draw9x9Grid():
		color="#D0ffff"
		for rowNo in range(1,10,3):
			for colNo in range(0,9,3):
				draw3x3Grid(rowNo,colNo,color)
				if color=="#D0ffff":
					color="#ffffd0"
				else:
					color="#D0ffff"
	def getValues():
		board=[]	
		label.configure(text="")
		solvedLabel.configure(text="")
		for row in range(2,11):
			rows=[]
			for col in range(1,10):
				val=cells[(row,col)].get()
				if val=="":
					rows.append(0)
				else:
					rows.append(int(val))
			board.append(rows)
		print(board)
		#burda sen board alcan dönen board aşağıda olcak
		board=[[3, 4, 7, 2, 5, 1, 8, 6, 9], [2, 1, 9, 7, 6, 8, 4, 3, 5], [5, 8, 6, 3, 4, 9, 2, 1, 7], [6, 7, 8, 5, 1, 3, 9, 4, 2], [4, 5, 1, 9, 2, 6, 7, 8, 3], [9, 2, 3, 8, 7, 4, 1, 5, 6], [1, 9, 2, 4, 3, 5, 6, 7, 8], [8, 6, 5, 1, 9, 7, 3, 2, 4], [7, 3, 4, 6, 8, 2, 5, 9, 1]]
		fil(board)
	def fil(board):
		
		for rows in range(2,11):
			for col in range(1,10):
				val=board[rows-2][col-1]
				if val!=0:
					cells[(rows,col)].delete(0,"end")
					cells[(rows,col)].insert(0,val)
				

	btn=Button(root,command=getValues,text="Çöz",width=10)
	btn.grid(row=20,column=1,columnspan=5,pady=20)
	draw9x9Grid()
	deneme=[[3,4, 7, 0, 5, 1, 0, 0, 0],[0, 1, 9, 0, 0, 0, 0, 0, 0],[5, 0, 6, 3, 4, 0, 0, 0, 0],[0, 7, 0, 0, 0, 3, 0, 0, 2],[0, 0, 1, 0, 0, 0, 7, 0, 0],[9, 0, 0, 8, 0, 0, 0, 5, 0],[0, 0, 0, 0, 3, 5, 6, 0, 8],[0, 0, 0, 0, 0, 0, 3, 2, 0],[0, 0, 0, 6, 8, 0, 5, 9, 1]]
	fil(deneme)
	
	
			
def determineText(img_no):
	global text
	global root
	if(img_no==1):
		text="Gaussian Blur"
	if(img_no==2):
		text="Adaptive Thresholding"
	if(img_no==3):
		text="Invert"
	if(img_no==4):
		text="Dilate"
	if(img_no==5):
		text="Find the 4 Corners in Image"
	if(img_no==6):
		text="Perspective Transform"
	if(img_no==7):
		text="Grid Marked"
	if(img_no==8):
		text="After Transform Gaussian Blur"
	if(img_no==9):
		text="After Transform Adaptive Thresholding"
	if(img_no==10):
		text="After Transform Invert"
	root.title(text)
	


# Calling the Tk (The initial constructor of tkinter)
root = Tk()

# We will make the title of our app as Image Viewer
root.title(text)

# The geometry of the box which will be displayed
# on the screen
root.geometry("720x720")

# Adding the images using the pillow module which
# has a class ImageTk We can directly add the
# photos in the tkinter folder or we have to
# give a proper path for the images
image_no_1 = ImageTk.PhotoImage(Image.open("1.jpeg"))
image_no_2 = ImageTk.PhotoImage(Image.open("2.jpeg"))
image_no_3 = ImageTk.PhotoImage(Image.open("3.jpeg"))
image_no_4 = ImageTk.PhotoImage(Image.open("4.jpeg"))
image_no_5 = ImageTk.PhotoImage(Image.open("5.jpeg"))
image_no_6 = ImageTk.PhotoImage(Image.open("6.jpeg"))
image_no_7 = ImageTk.PhotoImage(Image.open("7.jpeg"))
image_no_8 = ImageTk.PhotoImage(Image.open("8.jpeg"))
image_no_9 = ImageTk.PhotoImage(Image.open("9.jpeg"))
image_no_10 = ImageTk.PhotoImage(Image.open("10.jpeg"))

# List of the images so that we traverse the list
List_images = [image_no_1, image_no_2, image_no_3, image_no_4,image_no_5,image_no_6,image_no_7,image_no_8,image_no_9,image_no_10]

label = Label(root)

# We have to show the box so this below line is needed
label.grid(row=1, column=2, columnspan=4)

# We will have three button back ,forward and exit
button_back = Button(root, text="Geri", command=back,
					state=DISABLED)

# We will have three button back ,forward and exit
button_browse = Button(root, text="Yükle", command=showImage)
# root.quit for closing the app
button_exit = Button(root, text="Çıkış",
					command=root.quit)

button_forward = Button(root, text="İleri",
						command=lambda: forward(1))

# grid function is for placing the buttons in the frame
button_back.grid(row=5, column=1)
button_browse.grid(row=0, column=1)
button_exit.grid(row=0, column=6)
button_forward.grid(row=5, column=6)

root.mainloop()