
import numpy as np
from math import pi,cos,sin
import matplotlib.pyplot as plt
import matplotlib.animation as animation



# faire n trucs pour rigoler
a,g,m=1,9.81,1


# forçage ensuite




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
    return L[:,0],L[:,1]



def evolue(z):
    return np.array([1,z[2],sin(z[1])*(2*a+0.5*cos(z[1])*z[2]*z[2]/(4/3+21/4*cos(z[1])*cos(z[1])))])


def étend_le_bail(theta):
    return [0,2*a*sin(theta),0],[0,2*a*cos(theta),4*a*cos(theta)]


T,theta=afficher_positionsvitesse(evolue,np.array([0,1,0.1]),0.001,200000)



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
    a,b=étend_le_bail(theta[20*i])
    line.set_data(a,b)
    return line,
 
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=1000, blit=True, interval=10, repeat=False)
plt.show()





