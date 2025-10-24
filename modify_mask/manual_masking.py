import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from skimage.draw import polygon
from skimage.measure import find_contours, approximate_polygon, subdivide_polygon
import os
import pickle


def get_yes_or_no_answer(question):
    """
    This function asks the user a question and validates their answer
    to be either 'yes' or 'no' (case-insensitive).
    Args:
    question (str): The question to ask the user.

    Returns:
    str: The user's answer (either 'yes' or 'no').
    """
    while True:
        answer = input(question + " (yes/no): ").lower().strip()
        if answer in ('yes', 'y', 'no', 'n'):
            return answer
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")

# --- The main event handler function ---
def onclick(event):
    """
    This function is called whenever a mouse button is pressed on the plot.
    """
    global points, cid, mask, data, mask_orig,left_buttons_initialized

    # We only care about left-clicks on the main axes
    if event.inaxes != ax0:# or event.button != 1:
        return
    
    if event.button == 1:
        if left_buttons_initialized == True:
            # Get the coordinates of the click
            ix, iy = int(round(event.xdata)), int(round(event.ydata))
            points.append((ix, iy))
            # Update the plot to show the new point
            ax0.plot(ix, iy, 'ro', markersize=4) # 'ro' means red 'o'
            fig.canvas.draw()
        if left_buttons_initialized == False:
            left_buttons_initialized = True

    # If we have collected enough points, or if we use right click create and show the mask
    if len(points) == MAX_POINTS or event.button == 3:
        print("All points collected. Creating mask...")
        
        # Disconnect the event handler so no more points can be added
        if cid:
            fig.canvas.mpl_disconnect(cid)
        
        # Get image dimensions
        img_shape = data.shape

        # Subdivide and smooth the polygon
        mask_orig = np.array(points)
        for _ in range(10):
            new_mask = subdivide_polygon(mask_orig, degree=2, preserve_ends=True)

        # Separate the coordinates into rows (y) and columns (x)
        rows = [p[1] for p in new_mask]
        cols = [p[0] for p in new_mask]
        # Create a boolean mask using scikit-image's polygon function
        # This function returns the row and column indices of pixels inside the polygon
        rr, cc = polygon(rows, cols, shape=img_shape)
        mask = np.zeros(img_shape, dtype=bool)
        mask[rr, cc] = True
        
        # Create a semi-transparent RGBA overlay for the mask
        # The shape is (height, width, 4) for R, G, B, Alpha
        overlay = np.zeros((*img_shape, 4))
        overlay[mask] = [1, 0, 0, 0.4] # Red color with 40% opacity

        # Display the mask on top of the image
        ax1.imshow(data, cmap='gray', origin='lower')
        ax1.imshow(overlay, origin='lower')
        ax2.imshow(data, cmap='gray', origin='lower')
        #display the contours of the mask
        contours = find_contours(mask, 0.5)
        for contour in contours:
            ax2.plot(contour[:, 1], contour[:, 0], linewidth=2, color='yellow')
        fig.canvas.draw()
        
        print("Mask displayed. You can now save or close the figure.")

#####main

dir_path     = '/data1/Python/python_projects/modify_mask/'#'/gehme-gpu2/tools/manual_mask/input/'
dir_path_out = '/data1/Python/python_projects/modify_mask/'
#make a list of all pkl files in the directory
#pkl_files = [f for f in os.listdir(dir_path) if f.endswith('.pkl')]

#select a specific event
selected_event = 'inference_base_event_GCS_20101212_A6_DS32fran_file_.pkl'

#open the selected pkl file
with open(os.path.join(dir_path, selected_event), 'rb') as f:
    event_data = pickle.load(f)

string_instrument = ['cor2a', 'cor2b', 'lascoc2']
allmanual_maska = []
allmanual_maskb = []
allmanual_maskl = []
allmanual_pointsa = []
allmanual_pointsb = []
allmanual_pointsl = []

for instrument in range(len(event_data['orig_img'])):
    
    for timeframe in range(len(event_data['orig_img'][instrument])):
        #instrument = 0, 1, 2 cor2a, cor2b, lascoC2 
        # --- Reset global variables ---
        points = []
        MAX_POINTS = 100
        cid = None # Connection ID for the event handler
        mask = ''
        mask_orig = []
        # Display the initial FITS image
        data = event_data['orig_img'][instrument][timeframe]
        fecha = event_data['dates'][instrument][timeframe].split('.')[0]
        #breakpoint()
        if len(data) == 0: #occurs when data = ''
            if instrument == 0:
                allmanual_maska.append(mask)
                allmanual_pointsa.append(mask_orig)
            elif instrument == 1:
                allmanual_maskb.append(mask)
                allmanual_pointsb.append(mask_orig)
            elif instrument == 2:
                allmanual_maskl.append(mask)
                allmanual_pointsl.append(mask_orig)
            continue

        tuneo = True
        while tuneo:
            fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(30, 10))
            ax0.imshow(data, cmap='gray', origin='lower')
            ax1.imshow(data, cmap='gray', origin='lower')
            ax2.imshow(data, cmap='gray', origin='lower')

            ax0.set_title(string_instrument[instrument]+'__'+fecha)
            ax1.set_title('Mask Overlay')
            ax2.set_title('Mask Contours')
            fig.canvas.manager.set_window_title('FITS Image Point Selector')
            points = []
            mask_orig = []
            mask = ''
            
            #initialize left buttons to skip the first left click
            left_buttons_initialized = False
            
            # Connect the onclick function to the 'button_press_event'
            cid = fig.canvas.mpl_connect('button_press_event', onclick)
            # Show the plot and start the interactive session
            plt.show()
            answer = get_yes_or_no_answer("seguimos tuneando la mascara?")
            if answer == 'no' or answer == 'n':
                tuneo = False
            else:
                tuneo = True     

        if instrument == 0:
            allmanual_maska.append(mask)
            allmanual_pointsa.append(mask_orig)
        elif instrument == 1:
            allmanual_maskb.append(mask)
            allmanual_pointsb.append(mask_orig)
        elif instrument == 2:
            allmanual_maskl.append(mask)
            allmanual_pointsl.append(mask_orig)

all_manual_masks = [allmanual_maska, allmanual_maskb, allmanual_maskl]
all_points = [allmanual_pointsa ,allmanual_pointsb, allmanual_pointsl]
event_data['manual_masks'] = [all_manual_masks] 
event_data['manual_points'] = [all_points]
aux = 'Diego_v1_'
output_name = selected_event.split('.')[0]+'_'+aux+'with_manual_masks.pkl'
with open(os.path.join(dir_path_out, output_name), 'wb') as f:
    pickle.dump(event_data, f)
print("Manual masks saved to pickle file. on path:", os.path.join(dir_path_out, output_name))
