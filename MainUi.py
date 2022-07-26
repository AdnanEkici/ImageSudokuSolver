from tkinter import *
from PIL import ImageTk, Image


def forward(img_no):

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
	button_for = Button(root, text="İleri",
						command=lambda: forward(img_no+1))

	# img_no+1 as we want the next image to pop up
	if img_no == 4:
		button_forward = Button(root, text="İleri",
								state=DISABLED)

	# img_no-1 as we want previous image when we click
	# back button
	button_back = Button(root, text="Geri",
						command=lambda: back(img_no-1))

	# Placing the button in new grid

	button_back.grid(row=5, column=1)
	button_exit.grid(row=0, column=6)
	button_forward.grid(row=5, column=6)


def back(img_no):

	# We will have global variable to access these
	# variable and change whenever needed
	global label
	global button_forward
	global button_back
	global button_exit
	label.grid_forget()

	# for clearing the image for new image to pop up
	label = Label(image=List_images[img_no - 1])
	label.grid(row=1, column=0, columnspan=3)
	button_forward = Button(root, text="İleri",
							command=lambda: forward(img_no + 1))
	button_back = Button(root, text="Geri",
						command=lambda: back(img_no - 1))
	print(img_no)

	# whenever the first image will be there we will
	# have the back button disabled
	if img_no == 1:
		button_back = Button(root, Text="Geri", state=DISABLED)

	label.grid(row=1, column=2, columnspan=4)
	button_back.grid(row=5, column=1)
	button_exit.grid(row=0, column=6)
	button_forward.grid(row=5, column=6)


# Calling the Tk (The initial constructor of tkinter)
root = Tk()

# We will make the title of our app as Image Viewer
root.title("Sudoku Solver")

# The geometry of the box which will be displayed
# on the screen
root.geometry("720x720")

# Adding the images using the pillow module which
# has a class ImageTk We can directly add the
# photos in the tkinter folder or we have to
# give a proper path for the images
image_no_1 = ImageTk.PhotoImage(Image.open("input5.jpeg"))
image_no_2 = ImageTk.PhotoImage(Image.open("input5.jpeg"))
image_no_3 = ImageTk.PhotoImage(Image.open("input5.jpeg"))
image_no_4 = ImageTk.PhotoImage(Image.open("input5.jpeg"))

# List of the images so that we traverse the list
List_images = [image_no_1, image_no_2, image_no_3, image_no_4]

label = Label(image=image_no_1)

# We have to show the box so this below line is needed
label.grid(row=1, column=2, columnspan=4)

# We will have three button back ,forward and exit
button_back = Button(root, text="Geri", command=back,
					state=DISABLED)

# root.quit for closing the app
button_exit = Button(root, text="Çıkış",
					command=root.quit)

button_forward = Button(root, text="İleri",
						command=lambda: forward(1))

# grid function is for placing the buttons in the frame
button_back.grid(row=5, column=1)
button_exit.grid(row=0, column=6)
button_forward.grid(row=5, column=6)

root.mainloop()