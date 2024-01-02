# pendule kapitza simple
import numpy as np
from math import pi,cos,sin,sqrt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# fixe et inférieure reliée à un objet ponctuel M de masse m pouvant se déplacer sa
# schéma simple pour reprendre les bases

lambd=0.3 # hauteur du point
lo=1 # longueur à vide
k=10 # période
m=1 # masse
alpha=1 # angle de pivot autour du centre
gg=9.81 # accélération de la pesanteur
frottements=0

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




# g est inutile ici pour l'instant, on rajoutera des perturbations style du vent périodique etc.. ou des fonctions aléatoires
def g(t):
    return 

def f(x,y,t): # version en fonction de x et xpoint
    u=sqrt(x**2+lambd**2)
    return y,(-1/m)*(x/u)*k*(u-lo)-frottements*y-gg*sin(alpha)

# système assez sensible aussi, faire attentoin à garder un pas assez petit sinon ça diverge grossièrement

X,Y=


fig,ax= plt.subplots() # initialise la figure
ax.grid()
line, = plt.plot([],[]) 
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.plot([0,x],[0,y],color="black")
# fonction à définir quand blit=True, toujours pas compris cette remarque

# crée l'arrière de l'animation qui sera présent sur chaque image
def init():
    line.set_data([],[])
    return line,

def animate(i): # fonction majeure
    if i == 0:
        line.set_data(X[0:1]+[x],Y[0:1]+[y])
    else:

        line.set_data(X[900*i:1000*i]+[0]+[X[1000*i-1]]+[x],Y[900*i:1000*i]+[0]+[Y[1000*i-1]]+[y])
    return line,
 
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=1000, blit=True, interval=10, repeat=True)

plt.show()




# voir 01/05/2022 page 1 pour des approfondissements