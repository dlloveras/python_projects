#---Programa para ajustar el modelo distorted de Teresa a imágenes de HI-1, en el plano x-z. Se abre una imagen fits, luego se hace clicks sobre la imagen y ajusta la mejor funcion
#   posible, hay que modificar la función F para ver qué forma tiene a priori antes.
import cv2
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

#Código para abrir la imagen
filename = r'C:\Users\franc\OneDrive\Escritorio\Franco\Datos\Eventos_FR_para_analizar\20110326\20110326_140901_24h1A.fts'
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
    #z = r*np.sin(phi) 
    #Hago ro=0, pero atento que luego hay que sumarlo, se podria calcular la distancia al Sol usando XCEN y el tamaño de los pixeles quizás? Sumar con la distancia de la nave al sol y triangular?

def torus_coordinates(phi,par):
    x = par[0]*F(phi+par[5],par)*np.cos(phi+par[5]) + par[3] #aca solo phi es la variable independiente
    y = par[0]*np.sin(phi+par[5]) + par[4]                  #tener en cuenta que escribo 'y' pero en el paper de teresa ella lo llama 'z', es la sección transversal
    return (x,y)

#---Definir la función F ----> par[0]=r, par[1]=delta, par[2]=landa, par[3]=offset x, par[4]=offset y, par[5]=offset_phi
def F(phi,par):  
    return par[1]                                                         # Caso elíptico
    #return par[1]*(1-par[1]*np.cos(phi))                                  # Caso papa (?) el caso b de la Fig.2 de Teresa  
    #return par[1]*(1-par[2]*((np.cos(phi))**2))                              # Caso peanut shape 
    #return par[1]*(1 - (par[2]*np.cos(phi)) + 3*par[1]*np.sin(phi/4))     # Caso forma corazon teresa, la dejaré para el último por ser la más complicada

#---Defino funcion de error para computar el error medio y hacer el minimo cuadrado
def err(par):   
    global coordinates
    f=[]
    for cor in coordinates: #hago un for para diferenciar por cuadrantes en phi, ya que si calculo el ángulo como atan y no le advierto del cuadrante, lo grafica mal
        #r=np.sqrt(cor[0]**2+cor[1]**2)
        dy=(cor[1]-par[4])
        dx=(cor[0]-par[3])
        if (dx>0 and dy>0) or (dx>0 and dy<0):
            off=0       
        if (dx<0 and dy<0) or (dx<0 and dy>0):
            off=np.pi
        phi=np.arctan(dy/dx)+off
        f.append(torus_coordinates(phi,par)) 
    f=np.array(f)
    c= np.array(coordinates)
    dist=np.sqrt((f[:,0]-c[:,0])**2+(f[:,1]-c[:,1])**2)
    #print(f,c,dist)
    return dist

#---Función que se llama cada vez que el usuario hace clic en la imagen
def onclick(event, x, y, flags, params):

    if event == cv2.EVENT_LBUTTONDOWN:  # Click izquierdo, primero clickear el centro
        coordinates.append((x, y))
        print('Click izquierdo',x,y)
        cv2.circle(img, (x, y), 5, (0, 255, 0),-1)
        
    elif event == cv2.EVENT_RBUTTONDOWN:  # Click derecho
        x0=800 
        y0=550 
        #x0=np.mean(coordinates[:][0]) #coordenadas aproximadas del centro desde donde quiero que ajuste la imagen, habria que ajustarlo
        #y0=np.mean(coordinates[:][1])
        #x0 = coordinates[0][0]
        #y0 = coordinates[0][1]
        par_ini=[200,0.5,0.5,x0,y0,np.pi/4] #parametros iniciales del ajuste: [r, delta, landa,  offset x, offset y, phi]
        #print(np.shape(coordinates))
        print('Las cond iniciales son [r, delta, landa,  offset x, offset y, phi]', par_ini)
        fit=least_squares(err,par_ini,method='lm')
        print(fit.x)                   #fit.x son los parametros, asi los guarda la variable fit
        pts=[]
        pts2=[]                         #armo otro set para graficar la funcion con los parametros iniciales y comparar con los puntos que yo le doy
        phi= np.linspace(0,2*np.pi, 1000)
        for phi_value in phi:
            pts.append(torus_coordinates(phi_value,fit.x)) #f[0]=x, f[1]=y
            pts2.append(torus_coordinates(phi_value,par_ini))
        pts=np.array(pts , np.int32)
        pts2=np.array(pts2 , np.int32)
        cv2.polylines(img,[pts],False,(0,255,255),2)
        #cv2.polylines(img,[pts2],False,(255,255,255),2)
        #img=img[::-1, :] #Invertir la imagen
        cv2.imshow('image',img)

# Mostrar la imagen
#img=img[::-1, :] #Invertir la imagen
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 512, 512)
cv2.imshow('image', img)

# Configurar la función para manejar eventos del mouse
params = {'img': img, 'coordinates': []}
cv2.setMouseCallback('image', onclick)#, params)

# Esperar a que se presione la tecla 'q' para salir
while cv2.waitKey(1) != ord('q'):
    pass

#cv2.waitKey(0)
cv2.destroyAllWindows() #cierra todas las ventanas

#Que se puede hacer, que una con polylines todos los puntos clickeados y que luego tome todos esos puntos