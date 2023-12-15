#---Programa para ajustar el modelo distorted de Teresa a imágenes de HI-1, en el plano x-z. Se abre una imagen fits, luego se hace clicks sobre la imagen y ajusta la mejor funcion
#   posible, hay que modificar la función F para ver qué forma tiene a priori antes.
import cv2
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

#Código para abrir la imagen
filename = '/data1/Python/python_projects/fit_on_image_franco/20110326_140901_24h1A.fts'
img = fits.getdata(filename)
m=np.nanmean(img)
st=np.nanstd(img)
scl=2
vmin=m-st*scl
vmax=m+st*scl
img = (((img - vmin) ) / (vmax - vmin))#.astype('uint8')
coordinates = []   # Lista de coordenadas de los clicks del usuario

#Por alguna razon el opencv da vuelta la imagen, esto es para darla vuelta en y
#img = cv2.flip(img, 0)

#---Función que transforma las coordenadas del toroide a coordenadas en 2D
    #Pongo las coordenadas del paper, pero por el momento voy a pararme para hacer Ro=0 y como me interesa solo 2D por el momento puedo obviar la coord y el cos(theta) en x
    #x = (Ro + r*F(theta)*np.cos(phi)) * np.cos(theta)
    #y = (Ro + r*F(theta)*np.cos(phi)) * np.sin(theta)
    #z = r*np.sin(phi)
#---Entonces mi sistema de coordenadas queda:
    #x = r*F(delta,theta)*np.cos(phi)
    #z = r*np.sin(phi)     #Hago ro=0, pero atento que luego hay que sumarlo, se podria calcular la distancia al Sol usando XCEN y el tamaño de los pixeles quizás? Sumar con la distancia de la nave al sol y triangular?

def torus_coordinates(phi,params):
   #global x0,y0
   #print(params)
   phi = np.radians(phi)
   x = params[0]*F(phi,params)*np.cos(phi)#+params[1]                  #aca solo phi es la variable independiente
   y = params[0]*np.sin(phi)              #+params[2]    #tener en cuenta que escribo 'y' pero en el paper de teresa ella lo llama 'z', es la sección transversal
   return (x,y)

#---Definir la función F ----> 
# params[0]=r, params[1]=x_center, params[2]=y_center, params[3]=delta, params[4]=lambda
def F(phi,params):  
    #return  1                                                             #Caso circular
    #return params[3]                                                        # Caso elíptico
    #return params[3]*(1-params[4]*np.cos(phi))                                  # Caso papa (?) el caso b de la Fig.2 de Teresa  
    #return params[3]*(1-params[4]*((np.cos(phi))**2))                           # Caso peanut shape 
    return params[3]*(1-(params[4]*np.cos(phi))+3*params[4]*np.sin(phi/4))      # Caso forma corazon teresa, la dejaré para el último por ser la más complicada

#---Defino funcion de error para computar el error medio y hacer el minimo cuadrado
def err(params):   
    global coordinates
        
    f=[]
    angulo=[]
    cord_cent=[]
    
    for cor in coordinates: #hago un for para diferenciar por cuadrantes en phi, ya que si calculo el ángulo como atan y no le advierto del cuadrante, lo grafica mal
        
        y_cent=(cor[1]-params[2]) #Llevo las coordenadas al sistema de referencia en el centro 
        x_cent=(cor[0]-params[1])
        phi=np.arctan(abs(y_cent/x_cent))
        #phi=np.arctan2(y_cent,x_cent)
        phi = np.degrees(phi)
        if (x_cent>0 and y_cent>0): 
            phi=phi       
        if (x_cent<0 and y_cent>0):
            phi=180-phi
        if (x_cent<0 and y_cent<0):
            phi=phi+180
        if (x_cent>0 and y_cent<0):
            phi=360-phi
        
        angulo.append(phi)
        cord_cent.append([x_cent,y_cent])
        #print(params)
        f.append(torus_coordinates(phi,params)) 
        
    f=np.array(f)
    coord= np.array(cord_cent)
    
    #plt.plot(c[:,0],c[:,1],color='red',marker='o',linestyle='None')
        #plt.show()

    #plt.plot(f[:,0],f[:,1],color='green',marker='x',linestyle='None')
    #breakpoint()
    #colores=["red","blue","yellow","green","purple","black"]
    #for i in range(coord.shape[0]):
        #plt.subplot(2, 1, 1)  # 2 filas, 1 columna, primer subplot
    #    plt.plot(coord[:,0][i],coord[:,1][i],color=colores[i],marker='o',linestyle='None')
        #plt.subplot(2, 1, 2)  # 2 filas, 1 columna, primer subplot
        #plt.gca().invert_yaxis()
    #    plt.plot(f[:,0][i],f[:,1][i],color=colores[i],marker='x',linestyle='None')
        #elemx=torus_coordinates(angulo[i],r[0])[0]+x0
        #elemy=torus_coordinates(angulo[i],r[0])[1]+y0
        #print(elemx,elemy)

    #plt.show()
    

    dist=np.sqrt((f[:,0]-coord[:,0])**2+(f[:,1]-coord[:,1])**2)
    #OBS:Si params tiene mas de 1 variable, entonces dist debe ser 1D array. 
    #dist=np.mean(dist)   

    return dist

#---Función que se llama cada vez que el usuario hace clic en la imagen
def onclick(event, x, y, flags, params):

    if event == cv2.EVENT_LBUTTONDOWN:  # Click izquierdo, primero clickear el centro
        y_shift = y
        coordinates.append((x, y_shift))
        print('Click izquierdo',x,y_shift)
        cv2.circle(img, (x, y_shift), 5, (0, 255, 0),-1)
        
        
    elif event == cv2.EVENT_RBUTTONDOWN:  # Click derecho
        #left bottom corner equal 0,0
        #Estimate of center position of the F function that we want to fit ->x0,y0
        vec_x=[cor[0] for cor in coordinates]
        vec_y=[cor[1] for cor in coordinates]
        x0=np.mean(vec_x) #coordenadas aproximadas del centro desde donde quiero que ajuste la imagen, habria que ajustarlo
        y0=np.mean(vec_y)
        
        #Estimate of the radius
        r_ini=np.mean(np.sqrt(([cor[0] for cor in coordinates]-x0)**2+([cor[1]for cor in coordinates]-y0)**2))
        #-----------------------
        #Initial parameters that depends on the F function
        #F = circle case
        #params_ini=[r_ini,x0,y0]    
        
        #F = elipse in x direction. opcion a) Fig 2 paper Teresa
        #delta_ini = 0.8          #This parameter requires a better estimate of initial value
        #params_ini=[r_ini,x0,y0,delta_ini]
        
        #F = Papa. opcion b) Fig 2 paper Teresa
        #delta_ini = 0.8
        #lambda_ini = 0.4
        #params_ini=[r_ini,x0,y0,delta_ini,lambda_ini]

        #F = Peanut shape. opcion c) Fig 2 paper Teresa
        #delta_ini = 0.5
        #lambda_ini = 0.4
        #params_ini=[r_ini,x0,y0,delta_ini,lambda_ini]

        #F = Heart shape. opcion d) Fig 2 paper Teresa
        delta_ini = 0.9
        lambda_ini = 0.4
        params_ini=[r_ini,x0,y0,delta_ini,lambda_ini]
        #--------------------------------------------
        print('El r inicial es:',params_ini)
        
        fit=least_squares(err,params_ini,method='lm')
        #fit.x son los parametros, asi los guarda la variable fit
        pts=[]
        #pts2=[]                         #armo otro set para graficar la funcion con los parametros iniciales y comparar con los puntos que yo le doy
        phi= np.linspace(0,360, 1000)
        print("solucion es")
        print(fit.x)
        print(type(fit.x))
        for phi_value in phi:
            pts.append(np.array(torus_coordinates(phi_value,fit.x)) + [fit.x[1],fit.x[2]])#+ [[fit.x[1]],[fit.x[2]]])
        
        pts=np.array(pts , np.int32)
                #pts2=np.array(pts2 , np.int32)
        
        cv2.polylines(img,[pts],False,(0,255,255),2)
        #cv2.polylines(img,[pts2],False,(255,255,255),2)
        
        cv2.imshow('image',img)

# Mostrar la imagen
#cv2.namedWindow('image',cv2.WINDOW_NORMAL)
#cv2.resizeWindow('image', 512, 512)

cv2.imshow('image', img)#,plt.gca().invert_yaxis()


# Configurar la función para manejar eventos del mouse
#params = {'img': img, 'coordinates': []}
cv2.setMouseCallback('image', onclick)#, params)
# Esperar a que se presione la tecla 'q' para salir
while cv2.waitKey(1) != ord('q'):
    pass

#cv2.waitKey(0)
cv2.destroyAllWindows() #cierra todas las ventanas

#Que se puede hacer, que una con polylines todos los puntos clickeados y que luego tome todos esos puntos