import pickle
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
import numpy as np
import matplotlib as mpl
#mpl.use('Agg')

def read_pickle(file):
    with open(file, "rb") as f:
        data = pickle.load(f)
    return data

def plot_to_png(ofile, orig_img, masks, scr_threshold=0.15, mask_threshold=0.6 , title=None, labels=None, boxes=None, scores=None):
    """
    Plot the input images (orig_img) along with the infered masks, labels and scores
    in a single image saved to ofile
    """    
    # only detections with score larger than this value are considered
    color=['r','b','g','k','y','m','c','w','r','b','g','k','y','m','c','w','w','w','w','w','w','w','w','w','w','w','w','w','w','w']
    obj_labels = ['Back', 'Occ','CME','N/A']
    #
    cmap = mpl.colors.ListedColormap(color)  
    nans = np.full(np.shape(orig_img[0]), np.nan)
    fig, axs = plt.subplots(1, len(orig_img)*2, figsize=(20, 10))
    axs = axs.ravel()
    for i in range(len(orig_img)): #1 iteracion por imagen?
        axs[i].imshow(orig_img[i], vmin=0, vmax=1, cmap='gray')
        axs[i].axis('off')
        axs[i+1].imshow(orig_img[i], vmin=0, vmax=1, cmap='gray')        
        axs[i+1].axis('off')        
        if boxes is not None:
            nb = 0
            for b in boxes[i]:
                if scores is not None:
                    scr = scores[i][nb]
                else:
                    scr = 0   
                if scr > scr_threshold:             
                    masked = nans.copy()
                    masked[:, :][masks[i][nb] > mask_threshold] = nb              
                    axs[i+1].imshow(masked, cmap=cmap, alpha=0.4, vmin=0, vmax=len(color)-1) # add mask
                    box =  mpl.patches.Rectangle(b[0:2],b[2]-b[0],b[3]-b[1], linewidth=2, edgecolor=color[nb], facecolor='none') # add box
                    axs[i+1].add_patch(box)
                    if labels is not None:
                        axs[i+1].annotate(obj_labels[labels[i][nb]]+':'+'{:.2f}'.format(scr),xy=b[0:2], fontsize=15, color=color[nb])
                print(nb)
                nb+=1
    # axs[0].set_title(f'Cor A: {len(boxes[0])} objects detected') 
    # axs[1].set_title(f'Cor B: {len(boxes[1])} objects detected')               
    # axs[2].set_title(f'Lasco: {len(boxes[2])} objects detected')     
    #if title is not None:
    #    fig.suptitle('\n'.join([title[i]+' ; '+title[i+1] for i in range(0,len(title),2)]) , fontsize=16)   
    plt.tight_layout()
    plt.savefig(ofile)
    plt.close()

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
            fig, axs = plt.subplots(1, len(orig_img)*2, figsize=(20, 10))
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
                    
                    axs[i+1].imshow(masked, cmap=cmap, alpha=0.4, vmin=0, vmax=len(color)-1) # add mask
                    box =  mpl.patches.Rectangle(event['BOX'][b][0:2], event['BOX'][b][2]- event['BOX'][b][0], event['BOX'][b][3]- event['BOX'][b][1], linewidth=2, edgecolor=color[int(event['CME_ID'][b])] , facecolor='none') # add box
                    axs[i+1].add_patch(box)
                    axs[i+1].scatter(round(all_center[0][0]), round(all_center[0][1]), color='red', marker='x', s=100)
                    axs[i+1].annotate(obj_labels[event['LABEL'][b]]+':'+'{:.2f}'.format(scr),xy=event['BOX'][b][0:2], fontsize=15, color=color[int(event['CME_ID'][b])])
                    
            points = plt.ginput(n=100, timeout=0)
            #expand the points 2 pixels in all directions
            for point in points:
                masked[int(point[1])-2:int(point[1])+2, int(point[0])-2:int(point[0])+2] = np.nan
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
        masks[b][np.where(np.isnan(masked))] = min_treshold
        #breakpoint()
    return masks

save_pickle = True
ending = '20110215_cor2a'
data = read_pickle('full_parametros_pre_plot2.pkl')
ok_dates = data['ok_dates']
df = data['df']
all_center = data['all_center']
ok_orig_img = data['ok_orig_img']
all_plate_scl = data['all_plate_scl']
file_names = data['file_names']
opath = '/data_local/python_projects/modify_mask/'
ofile = 'test.png'
mask_threshold = 0.6
scr_threshold = 0.56
mascaras = []
for m in range(0, len(ok_dates)):
    event = df[df['DATE_TIME'] == ok_dates[m]].reset_index(drop=True)
    breakpoint()
    new_masks = plot_to_png2(opath+file_names[m]+"test.png", [ok_orig_img[m]], event,[all_center[m]],mask_threshold=mask_threshold,
            scr_threshold=scr_threshold, title=[file_names[m]], plate_scl=all_plate_scl[m])
    mascaras.append(new_masks)
breakpoint()
#event['MASK'] = new_masks
    #guardar un pickle con las nuevas mascaras que luego debo darle como input al infer2.
if save_pickle:
    with open(opath+'/'+str.lower("new_masks")+ending+'.pkl', 'wb') as write_file:
        pickle.dump(mascaras, write_file)
breakpoint()
#necesito recuperar la nueva mascara.
#breakpoint()


