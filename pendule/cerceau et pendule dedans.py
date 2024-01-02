

import numpy as np
from math import pi,cos,sin
import matplotlib.pyplot as plt
import matplotlib.animation as animation


R,l,m1,m2,g=3,0.5,0.1,1,9.81
j1,j2=m1*l*l/3,m2*R*R
frott1,frott2=0,0
# schéma simple pour reprendre les bases
# theta 1 = angle du pendule interne

# tout écrire en vectoriel avec numpy

def rk4vect_dim4(F,vectinit,h,n):
    L,z=np.zeros((n,5)),vectinit
    L[0,:]=z
    for j in range(1,n):
        k1=F(z)
        k2=F(z+k1*h/2)
        k3=F(z+k2*h/2)
        k4=F(z+k3*h)
        z=z+(k1+2*k2+2*k3+k4)*h/6
        L[j]=z
    return L

def afficher_positionsvitesse(F,a,h,n): # graphe des fonctions positions et vitesse
    import matplotlib.pyplot as plt
    import numpy as np
    from math import cos,sin
    ax=plt.subplots()[1]
    ax.set(xlabel='x', ylabel='y')
    ax.grid()
    L=rk4vect_dim4(F,a,h,n)
    plt.plot(L[:,0],L[:,1],color='green')
    plt.plot(L[:,0],L[:,2],color='blue')
    plt.plot(L[:,0],L[:,3],color='green')
    plt.plot(L[:,0],L[:,4],color='blue')
    plt.show()
    return L[:,1],L[:,2],L[:,3],L[:,4]

def cerceau_pendule(z): #  z le vecteur en question (t,theta,thetapoint)
    theta12point=-g*sin(z[1])/(6*l*j1*(1+l*R*m1*j1/j2))
    return np.array([1,z[2],theta12point , z[4],-j1/j2*theta12point])







# portrait de phase serait pas mal ici i think









theta1,theta1point,theta2,theta2point=afficher_positionsvitesse(cerceau_pendule,np.array([0,1,1,0,0]),0.001,200000)



fig,ax= plt.subplots() # initialise la figure
ax.grid()
line, = plt.plot([],[]) 
plt.xlim(-10,10)
plt.ylim(-10,10)
plt.plot([-10,10],[0,0],c='black') # sol

X2=-2*pi*R*theta2
X1,Y1=X2+l*np.sin(theta1),R-l*np.cos(theta1)
# fonction à définir quand blit=True

defn=1000 # nombre de points pour la création du cercle

def cercle(i):
    return [X2[i]+ R*cos(2*pi*k/defn-pi/2) for k in range(defn+1)],[R*(1+sin(2*pi*k/defn-pi/2)) for k in range(defn +1)]


# crée l'arrière de l'animation qui sera présent sur chaque image
def init():
    line.set_data([],[])
    return line,

def animate(i):
    a,b=cercle(10*i)
    line.set_data(a+[X2[10*i]]+[X1[10*i]],b+[R]+[Y1[10*i]])
    return line,
 
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=1000, blit=True, interval=10, repeat=False)
plt.show()



# approf direct, faire tourner le bail




# foutre des ressors aussi un peu partout (voir les positions stables etc ..)
# penser à faire un vrai truc où on peut tout combiner et faire ce que l'on veut avec la mécanique
# introduire les chocs aussi, ça a l'air cool
