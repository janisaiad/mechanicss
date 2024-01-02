# pendule kapitza simple
import numpy as np
from math import pi,cos,sin
import matplotlib.pyplot as plt
import matplotlib.animation as animation

longueur,l0,g=0.5,1,9.81
w=3
frott=0
# schéma simple pour reprendre les bases


# tout écrire en vectoriel avec numpy

def rk4vect_dim4(F,vectinit,h,n):
    L,z=np.zeros((n,3)),vectinit
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
    plt.show()
    T=l0*np.sin(w*L[:,0])
    X=longueur*np.sin(L[:,1])+T
    Y=-longueur*np.cos(L[:,1])+T
    return T,X,Y

def kapitza1(z): #  z le vecteur en question (t,theta,thetapoint)
    return np.array([1,z[2],-3*l0/longueur*w*w*sin(w*z[0])*sin(z[1])-3*g/longueur*sin(z[1])-frott*z[2]])


"""def portrait_de_phase(F,a,h,n): # portrait de phase simple (la flemme)
    import matplotlib.pyplot as plt
    import numpy as np
    ax=plt.subplots()[1]
    ax.grid()
    l,lpoint,theta,thetapoint=rk4vect_dim4(F,a,h,n)
    plt.plot(l,lpoint)
    plt.show()
    ax=plt.subplots()[1]
    ax.grid()
    plt.plot(theta,thetapoint)
    plt.show()"""



T,X,Y=afficher_positionsvitesse(kapitza1,np.array([0,pi-0.01,0]),0.001,200000)



fig,ax= plt.subplots() # initialise la figure
ax.grid()
line, = plt.plot([],[]) 
plt.xlim(-5,5)
plt.ylim(-4,4)

# fonction à définir quand blit=True

# crée l'arrière de l'animation qui sera présent sur chaque image
def init():
    line.set_data([],[])
    return line,

def animate(i):
    line.set_data(np.append(X[10*i-10:10*i],0),np.append(Y[10*i-10:10*i],T[10*i]))
    return line,
 
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=1000, blit=True, interval=10, repeat=False)
plt.show()
















# idée : rajouter un ressort au lieu du pendule ptdrrrr