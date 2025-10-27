from multiprocessing.util import debug
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import numpy as np
from scipy import stats

def bimodality_coefficient(data):
    """
    Calculates the Bimodality Coefficient (BC) for a dataset.
    The BC formula uses Pearson's kurtosis (where a normal 
    distribution has a kurtosis of 3). 
    However, scipy.stats.kurtosis() calculates "excess kurtosis,"
    where a normal distribution is 0. 
    Therefore, we must add 3 to the result from scipy.
    """
    # Calculate skewness
    skewness = stats.skew(data)
    # Calculate excess kurtosis
    excess_kurtosis = stats.kurtosis(data)
    # Convert to Pearson's kurtosis
    kurtosis = excess_kurtosis + 3
    # Calculate the bimodality coefficient
    bc = (skewness**2 + 1) / kurtosis
    return bc

def fun_area_score(mask):
    '''
    Similar to rec2pol, but returns the area of the mask related to the total images size.
    '''
    mask_threshold = 0.54
    area_score=0.
    nans = np.full(mask.shape, np.nan)
    #creates an array with zero value inside the mask and Nan value outside             
    masked = nans.copy()
    masked[:, :][mask > mask_threshold] = 0
    #calculates geometric center of the image
    height, width = masked.shape
    #calculates the amount of pixels corresponding to the mask
    for x in range(width):
        for y in range(height):
            value=masked[x,y]
            if not np.isnan(value):
                area_score = area_score + 1.
    area_score = area_score/(height*width)
    return area_score

def calculate_metrics(mask1, mask2):
    """
    Calculates precision, recall, dice coefficient, and intersection over union (IoU) 
    between two binary masks.
    Args:
        mask1: The first binary mask (numpy array).
        mask2: The second binary mask (numpy array). Ground truth mask.
    """
    # Flatten the masks for efficient calculations
    mask1_flat = mask1#.flatten()
    mask2_flat = mask2#.flatten()
    if len(mask1_flat) != len(mask2_flat):
        breakpoint()
    # Calculate true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN)
    TP = np.sum(np.logical_and(mask1_flat, mask2_flat))
    FP = np.sum(np.logical_and(mask1_flat, np.logical_not(mask2_flat)))
    FN = np.sum(np.logical_and(np.logical_not(mask1_flat), mask2_flat))
    TN = np.sum(np.logical_and(np.logical_not(mask1_flat), np.logical_not(mask2_flat)))
    # Calculate precision, recall, dice coefficient, and IoU
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    dice = (2 * TP) / (2 * TP + FP + FN) if 2 * TP + FP + FN > 0 else 0
    iou = TP / (TP + FP + FN) if TP + FP + FN > 0 else 0
    return precision, recall, dice, iou

def NSD_differecnce(array1,array2):
    """
    Normalized Symmetric Difference (NSD) between two arrays.
    Union - intersection / target
    """
    intersection = np.logical_and(array1, array2)
    union = np.logical_or(array1, array2)
    target = np.sum(array2)
    return (np.sum(union) - np.sum(intersection)) / target if target > 0 else 0


def rec2pol(mask, center=None):
    '''
    Converts the x,y mask to polar coordinates
    Only pixels above the mask_threshold are considered
    TODO: Consider arbitrary image center, usefull in case of Cor-2B images.
    NO FUNCIONA BIEN EN EL CASO DE COR2B y en relacion al plot2. CHEQUEAR
    '''
    mask_threshold = 0.54
    nans = np.full(mask.shape, np.nan)
    pol_mask=[]
    #creates an array with zero value inside the mask and Nan value outside             
    masked = nans.copy()
    breakpoint()
    masked[:, :][mask > mask_threshold] = 0   
    #calculates geometric center of the image
    height, width = masked.shape
    if center is None:
        #case center is not defined. Calculates geometric center of the image
        center_x = width / 2
        center_y = height / 2
    else:
        #case center is defined as input. 
        center_x = center[0]
        #since occ_center is given using images coordinates, we need to invert the y axis.
        center_y = height-center[1]
        #center_y = center[1]
    #calculates distance to the point and the angle for the positive y axis
    for x in range(width):
        for y in range(height):
            value=masked[x,y]
            if not np.isnan(value):
                x_dist = (x-center_x)
                y_dist = (y-center_y)
                distance= np.sqrt(x_dist**2 + y_dist**2)
                #Si y es vertical y x es horizontal, entonces angle positivo calculado con respecto a y positivo, en forma antihoraria.
                angle = np.degrees(np.arctan2(x_dist,y_dist))
                if angle<0:
                    angle+=360
                pol_mask.append([distance,angle])
    return pol_mask

def rec2pol2(mask, center=None):
    mask_threshold = 0.54
    #nans = np.full(mask.shape, np.nan)
    pol_mask=[]
    
    masked = mask.copy()
    #breakpoint()
    masked[:, :][mask > mask_threshold] = 0   

    height, width = masked.shape
    if center is None:
        center_x = width / 2
        center_y = height / 2
    else:
        #case center is defined as input. 
        center_x = center[0]
        center_y = height-center[1]

    for x in range(width):
        for y in range(height):
            #value=masked[x,y]
            x_dist = (x-center_x)
            y_dist = (y-center_y)
            distance= np.sqrt(x_dist**2 + y_dist**2)
            angle = np.degrees(np.arctan2(x_dist,y_dist))
            if angle<0:
                angle+=360
            pol_mask.append([distance,angle])
    return pol_mask

def rec2pol2_mesh(mask, center=None):
    height, width = mask.shape
    if center is None:
        center_x = width / 2
        center_y = height / 2
    else:
        center_x = center[0]
        center_y = height - center[1]
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))

    x_cart = x_coords - center_x
    y_cart = center_y - y_coords

    # 4. Calculate distance (radius) for every pixel
    # This is an ndarray of shape (height, width)
    distance_mesh = np.sqrt(x_cart**2 + y_cart**2)
    return distance_mesh

def get_mask_props(masks, scores=None, plate_scl=1,centerpix=None, percentiles=[5,95],debug =False,filename_debug=None,sat=None):
    """
    Calculate properties of masks.
    """
    scr_threshold = 0.1
    if scores != None:
        scores_out = float(scores)
    if scores == None:
        scores_out = -1
    prop_list=[]
    #breakpoint()
    #box_center = np.array([boxes[0]+(boxes[2]-boxes[0])/2, boxes[1]+(boxes[3]-boxes[1])/2])
    #if centerpix is not None:
    #    distance_to_box_center = np.sqrt((centerpix[0]/2-box_center[0])**2 + (centerpix[1]/2-box_center[1])**2)
    #else:
    #    distance_to_box_center = np.sqrt((masks.shape[0]/2-box_center[0])**2 + (masks.shape[1]/2-box_center[1])**2)
    if centerpix is None:
        centerpix = [masks.shape[0]/2,masks.shape[1]/2]
    #breakpoint()
    pol_mask=rec2pol(masks,center=centerpix)
    if sat == 0:
        min_occ = 2.9 # Rs
    if sat == 1:
        min_occ = 3.9 # Rs
    if sat == 2:
        min_occ = 2.2 # Rs

    if (pol_mask is not None):            
            #takes the min and max angles and calculates cpa and wide angles
        angles_mask   = [s[1] for s in pol_mask]
        distance_mask = [s[0] for s in pol_mask]
        angles   = [angles_mask[j]              for j in range(len(angles_mask)) if distance_mask[j]*plate_scl >= min_occ] #avoids mask errors inside DNN
        distance = [distance_mask[j]*plate_scl  for j in range(len(angles_mask)) if distance_mask[j]*plate_scl >= min_occ]
        if len(angles)>0:
                # checks for the case where the cpa is close to 0 or 2pi
            #if np.max(angles)-np.min(angles) >= 0.9*2*np.pi:
            #    breakpoint()
            #    angles = [s-2*np.pi if s>np.pi else s for s in angles]
            if bimodality_coefficient(angles) <0.65: #unimodal distribution
                angles_centered = [s - np.median(angles) for s in angles]
                angles_centered = [s + 360 if s < -180 else s for s in angles_centered]
                aw_min = np.percentile(angles_centered, percentiles[0])
                aw_max = np.percentile(angles_centered, percentiles[1])
                angles_between_percentiles = [s for s in angles_centered if s >= aw_min and s <= aw_max]
                cpa_ang= np.median(angles_between_percentiles)+ np.median(angles)
                aw_min += np.median(angles)
                aw_max += np.median(angles)

            if bimodality_coefficient(angles) >= 0.65 and bimodality_coefficient(angles) <= 0.7: #transitional distribution
                breakpoint() #is this a Halo?

            if bimodality_coefficient(angles) > 0.7: #bimodal distribution detected. use to happen when the distribution is close to 0 and 360
                angles_centered = [s - np.median(angles) for s in angles]
                angles_centered = [s + 360 if s < 180 else s for s in angles_centered]

                aw_min = np.percentile(angles_centered, percentiles[0]) 
                aw_max = np.percentile(angles_centered, percentiles[1]) 
                angles_between_percentiles = [s for s in angles_centered if s >= aw_min and s <= aw_max]
                cpa_ang = np.median(angles_between_percentiles)
                aw_min += -360 + np.median(angles)
                aw_max += -360 + np.median(angles)
                cpa_ang += -360 + np.median(angles)
                if bimodality_coefficient(angles_centered) > 0.7:
                    angles_centered = [s - np.median(angles) for s in angles]
                    angles_centered = [s + 360 if s < -180 else s for s in angles_centered]
                    if bimodality_coefficient(angles_centered) > 0.7:
                        breakpoint()
                    aw_min = np.percentile(angles_centered, percentiles[0])
                    aw_max = np.percentile(angles_centered, percentiles[1])
                    angles_between_percentiles = [s for s in angles_centered if s >= aw_min and s <= aw_max]
                    cpa_ang= np.median(angles_between_percentiles)+ np.median(angles)
                    aw_min += np.median(angles)
                    aw_max += np.median(angles)
                
                #terminar estas lineas. poner bien el plot. devolver cpa y aw correcto. plot y chequear
            wide_ang=aw_max-aw_min #vale incluso si aw_min es negativo
            if cpa_ang < 0:
                breakpoint()
            if wide_ang <0:
                breakpoint()
            if debug:
                #plot angles histograms in degrees
                plt.hist([s for s in angles], bins=50)
                if aw_min < 0:
                    plt.axvline(x=aw_min+360, color='r', linestyle='--')
                else:
                    plt.axvline(x=aw_min, color='r', linestyle='--')
                plt.axvline(x=aw_max, color='r', linestyle='--')
                plt.axvline(x=cpa_ang, color='g', linestyle='--')
                plt.savefig(filename_debug+'.png')
                plt.close()
                #breakpoint()
            #end debug section

            #calculates the distance to the apex
            #distance = [s[0] for s in pol_mask]
            distance_abs= max(distance, key=abs)
            idx_dist = distance.index(distance_abs)

            #angle corresponding to the apex_dist position
            angulos = angles
            #angulos = [s[1] for s in pol_mask]
            apex_angl = angulos[idx_dist] 
            apex_dist = distance[idx_dist] * plate_scl
            #caclulates the area of the mask in % of the total image area
            area_score = fun_area_score(masks)
            #calculate apex distances as a percentil 98, and the corresponding angles
            apex_dist_percentile = np.percentile(distance, 98) 
            apex_dist_per = [d for d,a in zip(distance,angulos) if d >= apex_dist_percentile-0.5 and d<=apex_dist_percentile+0.5]
            apex_angl_per = [a for d,a in zip(distance,angulos) if d >= apex_dist_percentile-0.5 and d<=apex_dist_percentile+0.5]
    #breakpoint()
    prop_list.append([scores_out,cpa_ang, wide_ang, apex_dist, apex_angl, aw_min, aw_max, area_score,np.median(apex_dist_per),np.median(apex_angl_per)])
    if len(masks) == 0:
        prop_list.append([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, [np.nan], [np.nan]])
    return prop_list

def calculate_metrics_list(mask1, mask2,threshold_mask=True):
    """
    Calculates precision, recall, dice coefficient, and intersection over union (IoU) 
    between two binary masks.
    Args:
        mask1: The first binary mask (numpy array). could be a list of images
        mask2: The second binary mask (numpy array). Ground truth mask. list of 1 image
    """
    precision_list = []
    recall_list = []
    dice_list = []
    iou_list = []
    NSD_list = []

    try:
        zero = np.full(np.shape(mask1[0]), 0)
    except:
        breakpoint()
    # Flatten the masks for efficient calculations
    mask2_flat = mask2
    for j in range(len(mask1)):
        if threshold_mask: 
            mask1_flat = zero.copy()
        #masked[vesMask > 0] = 1
            mask1_flat[:, :][mask1[j] > 0.54] = 1
            mask1_flat[:, :][mask1[j] < 0.54] = 0
        else:
            mask1_flat = mask1[j]
        if len(mask1_flat) != len(mask2_flat):
            breakpoint()
        # Calculate true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN)
        TP = np.sum(np.logical_and(mask1_flat, mask2_flat))
        FP = np.sum(np.logical_and(mask1_flat, np.logical_not(mask2_flat)))
        FN = np.sum(np.logical_and(np.logical_not(mask1_flat), mask2_flat))
        TN = np.sum(np.logical_and(np.logical_not(mask1_flat), np.logical_not(mask2_flat)))
        # Calculate precision, recall, dice coefficient, and IoU
        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        dice = (2 * TP) / (2 * TP + FP + FN) if 2 * TP + FP + FN > 0 else 0
        iou = TP / (TP + FP + FN) if TP + FP + FN > 0 else 0
        NSD = (FP + FN )/ np.sum(mask2_flat) if np.sum(mask2_flat) > 0 else 0
        precision_list.append(precision)
        recall_list.append(recall)
        dice_list.append(dice)
        iou_list.append(iou)
        NSD_list.append(NSD)
    return precision_list, recall_list, dice_list, iou_list, NSD_list


def plot_to_png_contornos(ofile, orig_img, masks, true_mask1=None, true_mask2=None, scr_threshold=0.3, mask_threshold=0.54 , title=None,string=None, 
                labels=None, boxes=None, scores=None, version='v4',anotations=None,plot_boxes=None,not_plot_labels=None):
    """
    Plot the input images (orig_img) along with the infered masks, labels and scores
    in a single image saved to ofile
    """    
    # only detections with score larger than this value are considered
    color=['r','b','g','k','y','m','c','w','r','b','g','k','y','m','c','w']
    if version=='v4':
        obj_labels = ['Back', 'Occ','CME','N/A']
    elif version=='v5':
        obj_labels = ['Back', 'CME']
    elif version=='A4':
        obj_labels = ['Back', 'Occ','CME']
    elif version=='A6':
        obj_labels = ['Back', 'Occ','CME']
    else:
        print(f'ERROR. Version {version} not supported')
        sys.exit()
    cmap = mpl.colors.ListedColormap(color)  
    nans = np.full(np.shape(orig_img[0]), np.nan)
    
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot()
    #fig, axs = plt.subplots(1, len(orig_img)*3, figsize=(30, 10))
    #axs = axs.ravel()
    
    #breakpoint()
    #try:
    for i in range(len(orig_img)):
        ax1.imshow(orig_img[i], vmin=0, vmax=1, cmap='gray', origin='lower')
        ax1.axis('off')
        if string is not None:
            ax1.text(0, 0, string,horizontalalignment='left',verticalalignment='bottom',
            fontsize=15,color='white',transform=ax1.transAxes)
        if true_mask1 is not None:
            masked = nans.copy()
            masked[true_mask1[i] > 0] = 3
            mask_converted = np.nan_to_num(masked, nan=2.0)
            ax1.contour(mask_converted, levels=[2.5], colors='darkblue', alpha=0.99,linewidths=4)
        if true_mask2 is not None:
            masked = nans.copy()
            masked[true_mask2[i] > 0] = 3
            mask_converted = np.nan_to_num(masked, nan=2.0)
            ax1.contour(mask_converted, levels=[2.5], colors='green', alpha=0.99,linewidths=4)
        if boxes is not None:
            nb = 0
            for boxs in boxes[i]:
                if labels[i][nb] == 1: #skip occulter en A4, A6
                    nb+=1
                    continue
                if scores is not None:
                    scr = scores[i][nb]
                else:
                    scr = 0   
                if scr > scr_threshold:             
                    if np.nanmax(masks[i][nb]) < 0.1:
                        breakpoint()
                    if true_mask1 is not None:
                        best_iou, best_dice, best_prec, best_rec,max_iou1, max_dice, max_prec, max_rec = best_mask_treshold(masks[i][nb], orig_img, true_mask1[i],mask_thresholds_list=[mask_threshold])
                        ax1.contour(masks[i][nb], levels=[best_iou], colors='red', alpha=0.9,linewidths=4)
                        if true_mask2 is None: ax1.annotate('IoU : '+'{:.2f}'.format(max_iou1) ,xy=[10,30], fontsize=45, color='white')
                    if true_mask2 is not None:
                        best_iou, best_dice, best_prec, best_rec,max_iou2, max_dice, max_prec, max_rec = best_mask_treshold(masks[i][nb], orig_img, true_mask2[i],mask_thresholds_list=[mask_threshold])
                        ax1.annotate('IoU : ',xy=[10,30], fontsize=45, color='white') 
                        ax1.annotate('{:.2f}'.format(max_iou1),xy=[95,30], fontsize=45, color='blue')     
                        ax1.annotate('{:.2f}'.format(max_iou2),xy=[180,30], fontsize=45, color='green')
                    if true_mask1 is None:
                        ax1.contour(masks[i][nb], levels=[0.5], colors=color[nb], alpha=0.9,linewidths=4)
                    if plot_boxes is None:
                        box =  mpl.patches.Rectangle(boxs[0:2],boxs[2]-boxs[0],boxs[3]-boxs[1], linewidth=4, edgecolor=color[nb], facecolor='none') # add box
                        ax1.add_patch(box)

                    if not_plot_labels is None:
                        ax1.annotate(obj_labels[labels[i][nb]]+':'+'{:.2f}'.format(scr),xy=boxs[0:2], fontsize=35, color=color[nb])

                    if anotations is not None:
                        ax1.annotate(anotations,xy=[10,490], fontsize=30, color='white')
                nb+=1
    plt.tight_layout()
    plt.savefig(ofile)
    plt.close()

    return 

def retocando_manual_mask(mask=None,plate_scl=None,centerpix=None,sat=None):
    '''
    Function to retouch manual masks if needed.
    Currently does nothing, but can be modified to include retouching logic.
    '''
    if centerpix is None:
        centerpix = [mask.shape[0]/2,mask.shape[1]/2]
    #breakpoint()
    #replace False values on mask with 0
    mask_aux = mask.copy()
    #mask = np.nan_to_num(mask, nan=0)
    #pol_mask=rec2pol2(mask,center=centerpix)
    salida = rec2pol2_mesh(mask_aux,center=centerpix)
    #breakpoint()
    if sat == 0:
        min_occ = 2.6 # Rs
    if sat == 1:
        min_occ = 3.9 # Rs
    if sat == 2:
        min_occ = 2.15 # Rs
    
    inside = salida < min_occ/plate_scl
    #breakpoint()
    mask[inside] = False
    return mask

def best_mask_treshold(masks, orig_img, vesMask, mask_thresholds_list=np.arange(0.2, 0.95, 0.05).tolist()):
    #nans = np.full(np.shape(orig_img[0]), np.nan)
    zero = np.full(np.shape(orig_img[0]), 0)
    iou_list = []
    dice_list = []
    prec_list = []
    rec_list = []
    for mask_thresholds in mask_thresholds_list:
        masked = zero.copy()
        #masked[vesMask > 0] = 1
        masked[:, :][masks > mask_thresholds] = 1
        #intersection = np.logical_and(masked, vesMask)
        #union = np.logical_or(masked, vesMask)
        #iou_score = np.sum(intersection) / np.sum(union)
        precision, recall, dice, iou = calculate_metrics(vesMask, masked)
        dice_list.append(dice)
        prec_list.append(precision)
        rec_list.append(recall)
        iou_list.append(iou)
        
    best_mask_threshold_iou  = mask_thresholds_list[np.argmax(iou_list)]
    max_iou = np.max(iou_list)
    best_mask_threshold_dice = mask_thresholds_list[np.argmax(dice_list)]
    max_dice = np.max(dice_list)
    best_mask_threshold_prec = mask_thresholds_list[np.argmax(prec_list)]
    max_prec = np.max(prec_list)
    best_mask_threshold_rec  = mask_thresholds_list[np.argmax(rec_list)]
    max_rec = np.max(rec_list)    
    return best_mask_threshold_iou, best_mask_threshold_dice, best_mask_threshold_prec, best_mask_threshold_rec, max_iou, max_dice, max_prec, max_rec


dir = '/data2/DNN_masks/manual_masking_hebe/'
pkl_files_hebe = [f for f in os.listdir(dir) if f.endswith('.pkl') and 'Hebe' in f]
pkl_files_fer  = [f for f in os.listdir(dir) if f.endswith('.pkl') and 'Fer' in f]
aux_output = '_hebe_manual_mask_new_angles'
model = 'A6DS32'
model_version = 'A6'
mask_threshold = 0.54
#make 2 lists

fechas_hebe = [pkl_files_hebe[j].split('_')[4] for j in range(len(pkl_files_hebe))]
fechas_fer  = [pkl_files_fer[j].split('_')[4]  for j in range(len(pkl_files_fer))]
set_hebe = set(fechas_hebe)
set_fer  = set(fechas_fer)
if len(fechas_hebe) != len(set_hebe):
    breakpoint() #look for repeated events
if set_hebe != set_fer:
    breakpoint()

#sort pkl_files_hebe based on fechas_hebe

pkl_files_hebe = [x for _, x in sorted(zip(fechas_hebe, pkl_files_hebe), key=lambda pair: pair[0])]
pkl_files_fer = [x for _, x in sorted(zip(fechas_fer, pkl_files_fer), key=lambda pair: pair[0])]
files_to_iterate = [[pkl_files_hebe[j], pkl_files_fer[j]] for j in range(len(pkl_files_hebe))]
event_short = {}
iou_manuales = []
for elements in files_to_iterate:
    file1, file2 = elements
    if '20130527' not in file1:
        continue
    breakpoint()
    print(file1, file2)
    #breakpoint()
    with open(dir+file1, 'rb') as f: event1 = pickle.load(f)
    with open(dir+file2, 'rb') as f: event2 = pickle.load(f)
    create_date = 0
    
    for sat in range(3):
        #if sat != 0: #remove
        #    continue
        mask_manual1 = event1['manual_masks'][0][sat]
        mask_manual2 = event2['manual_masks'][0][sat]
        
        for instant in range(len(mask_manual1)):
            #if instant != 6: #remove
            #    continue
            print(f'{instant}_{sat}')
            mask_manual1 = event1['manual_masks'][0][sat][instant]
            mask_manual2 = event2['manual_masks'][0][sat][instant]
            if len(mask_manual1) == 0:
                continue
            if create_date == 0:
                date_event = ''.join(event1['dates'][sat][instant].split('T')[0].split('-'))
                folder = date_event
                event_short[date_event] = {}
                create_date = 1
            center_pix   = event1['centerpix'][sat][instant]
            plt_scale    = event1['plate_scl'][sat][instant]
            labels       = event1['labels'][sat][instant]
            scores       = event1['scores'][sat][instant]
            boxes        = event1['boxes'][sat][instant]
            orig_imagen  = event1['orig_img'][sat][instant]
            mask_dnn1    = event1['masks'][sat][instant]
            mask_dnn2    = event2['masks'][sat][instant]
            #breakpoint()
            for k in range(len(mask_dnn1)):
                if np.sum(mask_dnn1[k] != mask_dnn2[k]) != 0:
                    breakpoint()
            time1        = event1['dates'][sat][instant]
            time2        = event2['dates'][sat][instant]
            if time1 != time2:
                breakpoint()
            if len(mask_dnn1) == 0:
                print(f'No DNN masks for event {date_event}, sat {sat}, instant {instant}')
                continue
            date_str    = event1['dates'][sat][instant][:-7]
            #breakpoint()
            #retocamos las macascas manuales, forzamos el occ interno.
            mask1_retoque = retocando_manual_mask(mask=mask_manual1, plate_scl=plt_scale, centerpix=center_pix, sat=sat)
            mask2_retoque = retocando_manual_mask(mask=mask_manual2, plate_scl=plt_scale, centerpix=center_pix, sat=sat)
            mask_manual1 = mask1_retoque
            mask_manual2 = mask2_retoque

            a1,b1,c1,all_iou1,all_nsd1 = calculate_metrics_list(mask_dnn1, mask_manual1)
            a2,b2,c2,all_iou2,all_nsd2 = calculate_metrics_list(mask_dnn1, mask_manual2)
            iou_selected = max( (iou, index) for index, iou in enumerate(all_iou1) if labels[index] == 2)
            imax_old     = iou_selected[1]
            imax_old2 = np.argmax(all_iou1)
            if all_iou1[imax_old2] != iou_selected[0]:
                #may happen if there is trully no dnn mask intersecting the manual mask
                print(f'Warning: Different max iou selected for event {date_event}, sat {sat}, instant {instant}')
            iou1      = all_iou1[imax_old]
            iou2      = all_iou2[imax_old]
            nsd       = all_nsd1[imax_old]
            #breakpoint()
            TP = np.sum(np.logical_and(mask_manual1, mask_manual2))
            FP = np.sum(np.logical_and(mask_manual1, np.logical_not(mask_manual2)))
            FN = np.sum(np.logical_and(np.logical_not(mask_manual1), mask_manual2))
            iou_manual = TP / (TP + FP + FN) if TP + FP + FN > 0 else 0
            #_,_,_,iou_manual,_ = calculate_metrics_list(mask_manual1, mask_manual2)
            iou_manuales.append(iou_manual)
            ofile = os.path.join(dir+folder,date_str+'_sat_'+str(sat)+'_doble_retoque.png')
            if not os.path.exists(dir+folder):
                os.makedirs(dir+folder)
            orig_imagen = event1['orig_img'][sat][instant]
            plot_to_png_contornos(ofile, [orig_imagen], [[mask_dnn1[imax_old]]], [mask_manual1], [mask_manual2],scores=[[scores[imax_old]]], labels=[[labels[imax_old]]], boxes=[[boxes[imax_old]]],
                                    mask_threshold=mask_threshold, scr_threshold=0.1, version=model_version,string=None,anotations=date_str,not_plot_labels=True)
    #breakpoint()

#create histogram of iou_manuales
plt.hist(iou_manuales, bins=20)
mediana = np.median(iou_manuales)
plt.axvline(x=mediana, color='r', linestyle='--', label='Median: {:.2f}'.format(mediana))
plt.xlabel('IoU between manual masks')
plt.ylabel('Number of cases')
plt.title('Histogram of IoU between manual masks')
plt.savefig('/data2/DNN_masks/iou_manuals_histogram'+aux_output+'.png')
plt.close()
breakpoint()


