import pickle
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
import numpy as np
import matplotlib as mpl
#mpl.use('Agg')
import copy

def read_pickle(file):
    with open(file, "rb") as f:
        data = pickle.load(f)
    return data

#--------------------------------------------------------------------------------------------------------------------
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


def plot_to_png2(ofile, orig_img, event, all_center, mask_threshold, scr_threshold, title=None, plate_scl=1):
    """
    Plot the input images (orig_img) along with the infered masks, labels and scores
    in a single image saved to ofile
    """    
    #mpl.use('TkAgg')
    color=['r','b','g','k','y','m','c','w','r','b','g','k','y','m','c','w']
    obj_labels = ['Back', 'Occ','CME','N/A']
    masks=event['MASK']
    cmap = mpl.colors.ListedColormap(color)  
    nans = np.full(np.shape(orig_img[0]), np.nan)
    masks_modified = copy.deepcopy(masks[0])#masks.copy()
    min_treshold = 0.2 #value set the pixels outside the tuned mask.
    for i in range(len(orig_img)):
        print("Imagen:", title[i])
        answer = get_yes_or_no_answer("Do you want to tune the mask?") 
        if answer == 'no' or answer == 'n':
            tuneo = False
        else:
            tuneo = True
        contador = 0
        while tuneo:
            fig, axs = plt.subplots(1, 2, figsize=(20, 10))
            axs = axs.ravel()
            #plot image and mask as usual
            axs[i].imshow(orig_img[i], vmin=0, vmax=1, cmap='gray')
            axs[i].axis('off')
            axs[i].annotate( str(event["DATE_TIME"][i]),xy=[10,500], fontsize=15, color='w')

            #Incluir event date time
            axs[i+1].imshow(orig_img[i], vmin=0, vmax=1, cmap='gray')        
            axs[i+1].axis('off') 
            for b in range(len(event['LABEL'])):
                scr = event['SCR'][b]
                if scr > scr_threshold:             
                    if contador ==0:
                        masked = nans.copy()            
                        masked[:, :][masks[b] > mask_threshold] = event['CME_ID'][b]     
                        if event['CME_ID'][b] != 0:
                            breakpoint()
                    axs[i+1].imshow(masked, cmap=cmap, alpha=0.4, vmin=0, vmax=1) # add mask
                    box =  mpl.patches.Rectangle(event['BOX'][b][0:2], event['BOX'][b][2]- event['BOX'][b][0], event['BOX'][b][3]- event['BOX'][b][1], linewidth=2, edgecolor=color[int(event['CME_ID'][b])] , facecolor='none') # add box
                    axs[i+1].add_patch(box)
                    axs[i+1].scatter(round(all_center[0][0]), round(all_center[0][1]), color='red', marker='x', s=100)
                    axs[i+1].annotate(obj_labels[event['LABEL'][b]]+':'+'{:.2f}'.format(scr),xy=event['BOX'][b][0:2], fontsize=15, color=color[int(event['CME_ID'][b])])
                    
            points = plt.ginput(n=100, timeout=0)
            #expand the points 2 pixels in all directions
            for point in points:
                masked[int(point[1])-2:int(point[1])+2, int(point[0])-2:int(point[0])+2] = np.nan
                masks_modified[int(point[1])-2:int(point[1])+2, int(point[0])-2:int(point[0])+2] = np.nan
            #for point in points:
            #    masked[int(point[1]), int(point[0])] = np.nan
            #use ginput to select points in the image
            #use the position of points to transform masked elements equal 1 to nan
            axs[i+1].imshow(orig_img[i], vmin=0, vmax=1, cmap='gray') 
            axs[i+1].imshow(masked, cmap=cmap, alpha=0.4, vmin=0, vmax=len(color)-1) # add mask
            plt.tight_layout()
            plt.savefig(ofile)
            plt.close()
            
            answer = get_yes_or_no_answer("Do you want to tune the mask?") 
            if answer == 'no' or answer == 'n':
                tuneo = False
            else:
                tuneo = True        
            contador = contador +1 
        #replace masks[b] elemens where masked is nan with min_treshold
        #breakpoint()
        if contador !=0:
            masks_modified[np.where(np.isnan(masks_modified))] = min_treshold
        #breakpoint()
        #if contador =0 then masks = original masks, nathing happens
    return masks_modified
"""
Este archivo es para modificar las mascaras de las CMEs que se detectaron en el archivo full_parametros_pre_plot2.pkl,
que es una salida del neural_cme_seg_diego.py.
Usualmente las mascaras obtenidas con imageens reales salen con errores, por lo que se necesita modificarlas. Este codigo
permite modificarlas a mano, haciendo click en la imagen y seleccionando los pixeles que se quieren modificar.
"""



save_pickle = True
#ending = '20110215_cor2a'
#data = read_pickle('full_parametros_pre_plot2_20110215_cor2a.pkl')
<<<<<<< HEAD
#ending = '20110215_cor2b_v1'
#data = read_pickle('full_parametros_pre_plot2_20110215_cor2b.pkl')
#opath = '/data_local/python_projects/modify_mask/'
#----------------------------
ending = '20100403_cor2a_v1'
data = read_pickle('full_parametros_pre_plot2_20100403_cor2a_niemela.pkl')
=======
#ending = '20100403_cor2a_v1'
#data = read_pickle('full_parametros_pre_plot2_20100403_cor2a_niemela.pkl')
#ending = '20100403_cor2b_v1'
#data = read_pickle('full_parametros_pre_plot2_20100403_cor2b_niemela.pkl')
ending = '20100403_lascoc2_v1'
data = read_pickle('full_parametros_pre_plot2_20100403_lascoc2_niemela.pkl')
>>>>>>> fea6850f5a5521087706046a77eb81697e77e3d4
ok_dates = data['ok_dates']
df = data['df']
all_center = data['all_center']
ok_orig_img = data['ok_orig_img']
all_plate_scl = data['all_plate_scl']
file_names = data['file_names']
opath = '/data1/Python/python_projects/modify_mask/'
#ofile = 'test.png'
mask_threshold = 0.8
scr_threshold = 0.56
mascaras = []
label = []
scr = []
box = []
cme_id = []
date_time = []
for m in range(0, len(ok_dates)):
    event = df[df['DATE_TIME'] == ok_dates[m]].reset_index(drop=True)
    #breakpoint()
    if len(event) >1:
        event = event[event['SCR'] == event['SCR'].max()] 
    #breakpoint()
    new_masks = plot_to_png2(opath+file_names[m]+"test.png", [ok_orig_img[m]], event,[all_center[m]],mask_threshold=mask_threshold,
            scr_threshold=scr_threshold, title=[file_names[m]], plate_scl=all_plate_scl[m])
    mascaras.append(new_masks)
    label.append(event['LABEL'].tolist())
    scr.append(event['SCR'].tolist())
    box.append(event['BOX'].tolist())
    cme_id.append(event['CME_ID'].tolist())
    date_time.append(event['DATE_TIME'].tolist())
    breakpoint()
output = {'MASK': mascaras, 'LABEL': label, 'SCR': scr, 'BOX': box, 'CME_ID': cme_id, 'OK_DATES': ok_dates,'OK_ORIG_IMG' : ok_orig_img,
            'PLATE_SCL': all_plate_scl,'orig_df': df, 'file_names': file_names, 'DATE_TIME': date_time, 'all_center': all_center}
    #guardar un pickle con las nuevas mascaras que luego debo darle como input al infer2.
breakpoint()
if save_pickle:
    with open(opath+str.lower("new_masks")+ending+'.pkl', 'wb') as write_file:
        pickle.dump(output, write_file)
breakpoint()
#necesito recuperar la nueva mascara.
#breakpoint()


